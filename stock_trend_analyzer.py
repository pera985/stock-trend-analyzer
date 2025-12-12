#!/usr/bin/env python3
"""
Stock Trend Analyzer - Massive.com Edition
Identifies stocks that are trending up based on multiple technical indicators
Supports custom stock lists or full exchange scanning (NYSE, NASDAQ, AMEX)
Uses Massive.com (Polygon.io) API for professional-grade data with WebSocket support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
from typing import List, Dict, Optional
import argparse
import requests
import os
from collections import deque
import logging
import sys
from polygon import RESTClient  # Updated from 'massive' to 'polygon' (Polygon.io API client)
import pytz
import config  # Import configuration module

warnings.filterwarnings('ignore')


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs only to console
    """
    # Create logger
    logger = logging.getLogger('StockTrendAnalyzer')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


# Initialize logger (will be reconfigured in main())
logger = logging.getLogger('StockTrendAnalyzer')


class RateLimiter:
    """Rate limiter to stay within API limits"""
    def __init__(self, max_requests_per_minute: int = None):
        if max_requests_per_minute is None:
            max_requests_per_minute = config.DEFAULT_RATE_LIMIT
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        logger.info(f"RateLimiter initialized with {max_requests_per_minute} requests/minute limit")
        
    def wait_if_needed(self):
        """Wait if we're approaching rate limit"""
        now = time.time()
        
        # Remove requests older than 60 seconds
        initial_count = len(self.request_times)
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        removed_count = initial_count - len(self.request_times)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} expired requests from tracking window")
        
        # If we're at the limit, wait until the oldest request expires
        if len(self.request_times) >= self.max_requests:
            sleep_time = 60 - (now - self.request_times[0]) + 0.1
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching ({len(self.request_times)}/{self.max_requests}), "
                             f"waiting {sleep_time:.1f}s...")
                print(f"Rate limit approaching, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                # Clear old requests after waiting
                now = time.time()
                while self.request_times and now - self.request_times[0] > 60:
                    self.request_times.popleft()
                logger.info("Rate limit wait completed, resuming requests")
        
        # Record this request
        self.request_times.append(time.time())
        logger.debug(f"Request logged. Current window: {len(self.request_times)}/{self.max_requests} requests")


class MassiveClient:
    """Client for Massive.com (Polygon.io) API with REST and WebSocket support"""
    def __init__(self, api_key: str, rate_limiter: RateLimiter, aggregate_seconds: int = 30):
        self.api_key = api_key
        self.rest_client = RESTClient(api_key)
        self.rate_limiter = rate_limiter
        self.aggregate_seconds = aggregate_seconds
        logger.info(f"MassiveClient initialized with REST client (1sec aggregation: {aggregate_seconds}s)")
        
    def get_daily_data(self, ticker: str, outputsize: str = 'full') -> Optional[pd.DataFrame]:
        """
        Get daily time series data using Massive.com REST API
        outputsize: 'compact' (100 days) or 'full' (5 years)
        """
        logger.info(f"Fetching daily data for {ticker} (outputsize: {outputsize})")
        self.rate_limiter.wait_if_needed()

        try:
            # Calculate date range based on outputsize
            to_date = datetime.now().strftime('%Y-%m-%d')
            if outputsize == 'compact':
                from_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
            else:  # full
                from_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

            logger.debug(f"Making API request to Massive.com for {ticker} from {from_date} to {to_date}")
            start_time = time.time()

            # Use Massive REST client to fetch aggregate bars
            aggs = []
            for agg in self.rest_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=from_date,
                to=to_date,
                adjusted=True,
                limit=50000
            ):
                aggs.append(agg)

            elapsed_time = time.time() - start_time
            logger.debug(f"API response received in {elapsed_time:.2f}s ({len(aggs)} bars)")

            if not aggs:
                logger.warning(f"No daily data found for {ticker}")
                return None

            # Convert to DataFrame
            data_list = []
            for agg in aggs:
                data_list.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume
                })

            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
            logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            logger.error(f"Unexpected error fetching daily data for {ticker}: {e}", exc_info=True)
            return None

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the most recent price for a ticker including extended hours (pre-market/after-hours).
        Uses 1-minute bars from the current day to capture the latest trading activity.
        """
        logger.debug(f"Fetching latest price for {ticker} (including extended hours)")
        self.rate_limiter.wait_if_needed()

        try:
            # Get today's date and fetch 1-minute bars for today
            today = datetime.now().strftime('%Y-%m-%d')

            aggs = []
            for agg in self.rest_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='minute',
                from_=today,
                to=today,
                adjusted=True,
                limit=1000  # Should cover full extended hours day
            ):
                aggs.append(agg)

            if not aggs:
                logger.debug(f"No intraday data found for {ticker} today, falling back to daily")
                return None

            # Return the close price of the most recent bar
            latest_price = aggs[-1].close
            latest_time = pd.to_datetime(aggs[-1].timestamp, unit='ms', utc=True).tz_convert('US/Eastern')
            logger.debug(f"{ticker}: Latest price ${latest_price:.2f} at {latest_time}")
            return latest_price

        except Exception as e:
            logger.warning(f"Error fetching latest price for {ticker}: {e}")
            return None

    def get_earnings_calendar(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get earnings calendar data for a ticker using Massive.com REST API
        Returns DataFrame with earnings dates
        Note: Massive.com provides ticker events which include earnings
        """
        logger.info(f"Fetching earnings calendar for {ticker}")
        self.rate_limiter.wait_if_needed()

        try:
            logger.debug(f"Making API request to Massive.com for earnings calendar: {ticker}")
            start_time = time.time()

            # Fetch ticker events for the last year
            # Massive.com uses ticker events API for corporate actions including earnings
            base_url = f"https://api.polygon.io/vX/reference/tickers/{ticker}/events"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(base_url, headers=headers, timeout=10)

            elapsed_time = time.time() - start_time
            logger.debug(f"API response received in {elapsed_time:.2f}s (status: {response.status_code})")

            if response.status_code != 200:
                logger.debug(f"No earnings data available for {ticker} (status: {response.status_code})")
                return None

            data = response.json()
            if 'results' not in data or not data['results']:
                logger.debug(f"No earnings events found for {ticker}")
                return None

            # Extract earnings events
            earnings_list = []
            results = data.get('results', {})
            events = results.get('events', [])

            for event in events:
                if event.get('type') == 'earnings':
                    earnings_list.append({
                        'symbol': ticker,
                        'reportDate': pd.to_datetime(event.get('date')),
                        'fiscalQuarter': event.get('fiscal_quarter', ''),
                        'fiscalYear': event.get('fiscal_year', '')
                    })

            if not earnings_list:
                logger.debug(f"No earnings data found for {ticker}")
                return None

            df = pd.DataFrame(earnings_list)
            df = df.dropna(subset=['reportDate'])
            df.sort_values('reportDate', inplace=True)

            logger.debug(f"Successfully fetched {len(df)} earnings records for {ticker}")
            return df

        except Exception as e:
            logger.warning(f"Could not fetch earnings data for {ticker}: {e}")
            return None

    def get_earnings_dates(self, ticker: str, from_date: str = None,
                           to_date: str = None) -> List[datetime]:
        """
        Get earnings report dates for a ticker from Polygon.io using financials API.
        Same approach as inflection_detector for consistency.

        Args:
            ticker: Stock symbol
            from_date: Start date (YYYY-MM-DD). Default: 2 years ago.
            to_date: End date (YYYY-MM-DD). Default: today.

        Returns:
            List of datetime objects representing earnings dates.
        """
        logger.info(f"Fetching earnings dates for {ticker}")
        self.rate_limiter.wait_if_needed()

        # Default date range
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        # Use the stock financials endpoint (same as inflection_detector)
        url = "https://api.polygon.io/vX/reference/financials"
        params = {
            "ticker": ticker,
            "filing_date.gte": from_date,
            "filing_date.lte": to_date,
            "timeframe": "quarterly",
            "order": "asc",
            "limit": 50,
            "apiKey": self.api_key
        }

        earnings_dates = []

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for result in data.get("results", []):
                # Use filing_date as the earnings announcement date
                filing_date = result.get("filing_date")
                if filing_date:
                    try:
                        dt = datetime.strptime(filing_date, "%Y-%m-%d")
                        earnings_dates.append(dt)
                    except ValueError:
                        pass

            logger.debug(f"Successfully fetched {len(earnings_dates)} earnings dates for {ticker}")
            return earnings_dates

        except Exception as e:
            # Silently fail - earnings dates are optional
            logger.debug(f"Could not fetch earnings dates for {ticker}: {e}")
            return []

    def get_intraday_data(self, ticker: str, interval: str = '5min',
                          outputsize: str = 'full') -> Optional[pd.DataFrame]:
        """
        Get intraday time series data using Massive.com REST API
        interval: '1sec', '1min', '3min', '5min', '15min', '30min', '60min'
        outputsize: 'compact' (latest 100 data points) or 'full' (trailing 30 days)
        Note: 1sec interval is automatically limited to 'compact' mode
        """
        # Force compact mode for 1-second data to avoid excessive data requests
        if interval == '1sec' and outputsize == 'full':
            logger.warning(f"1-second interval requested with 'full' outputsize - forcing to 'compact' mode")
            outputsize = 'compact'

        logger.info(f"Fetching intraday data for {ticker} (interval: {interval}, outputsize: {outputsize})")
        self.rate_limiter.wait_if_needed()

        try:
            # Map interval to Massive.com timespan and multiplier
            interval_map = {
                '1sec': (1, 'second'),
                '1min': (1, 'minute'),
                '3min': (3, 'minute'),
                '5min': (5, 'minute'),
                '15min': (15, 'minute'),
                '30min': (30, 'minute'),
                '60min': (1, 'hour')
            }

            if interval not in interval_map:
                logger.error(f"Unsupported interval: {interval}")
                return None

            multiplier, timespan = interval_map[interval]

            # Calculate date range based on outputsize
            to_date = datetime.now().strftime('%Y-%m-%d')

            # Special handling for 1-second data: limit to current day only
            if interval == '1sec':
                # For 1-second data, always use current day regardless of outputsize
                from_date = datetime.now().strftime('%Y-%m-%d')
                logger.info(f"1-second interval: limiting data to current trading day only")
            elif outputsize == 'compact':
                # Get last 2 trading days for compact
                from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            else:  # full
                # Get last 30 days for full
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            logger.debug(f"Making API request to Massive.com for {ticker} from {from_date} to {to_date}")
            start_time = time.time()

            # Use Massive REST client to fetch aggregate bars
            aggs = []
            for agg in self.rest_client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                adjusted=True,
                limit=50000
            ):
                aggs.append(agg)

            elapsed_time = time.time() - start_time
            logger.debug(f"API response received in {elapsed_time:.2f}s ({len(aggs)} bars)")

            if not aggs:
                logger.warning(f"No intraday data found for {ticker}")
                return None

            # Convert to DataFrame
            data_list = []
            for agg in aggs:
                data_list.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms', utc=True).tz_convert('US/Eastern'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume
                })

            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # If 1-second data, aggregate to N-second bars for better visualization
            if interval == '1sec':
                agg_window = f'{self.aggregate_seconds}S'
                logger.info(f"Aggregating {len(df)} 1-second bars into {self.aggregate_seconds}-second bars for {ticker}")

                # Resample to N-second intervals
                df_resampled = pd.DataFrame()
                df_resampled['Open'] = df['Open'].resample(agg_window).first()
                df_resampled['High'] = df['High'].resample(agg_window).max()
                df_resampled['Low'] = df['Low'].resample(agg_window).min()
                df_resampled['Close'] = df['Close'].resample(agg_window).last()
                df_resampled['Volume'] = df['Volume'].resample(agg_window).sum()

                # Remove rows with NaN (periods with no data)
                df_resampled.dropna(inplace=True)

                logger.info(f"Successfully aggregated to {len(df_resampled)} {self.aggregate_seconds}-second bars for {ticker}")
                logger.debug(f"Date range: {df_resampled.index[0]} to {df_resampled.index[-1]}")

                return df_resampled
            else:
                logger.info(f"Successfully fetched {len(df)} {interval} data points for {ticker}")
                logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")

                return df

        except Exception as e:
            logger.error(f"Unexpected error fetching intraday data for {ticker}: {e}", exc_info=True)
            return None


class StockTrendAnalyzer:
    def __init__(self, api_key: str, interval='1d', period='6mo', max_requests_per_minute: int = None, aggregate_seconds: int = 30):
        """
        Initialize the analyzer

        Args:
            api_key: Massive.com API key
            interval: Data interval ('1d' for daily, '1sec', '1min', '3min', '5min', '15min', '30min', '60min' for intraday)
            period: Historical period for daily ('compact'=100 days, 'full'=5 years)
                    For intraday: 'compact'=100 data points, 'full'=30 days
            max_requests_per_minute: Rate limit (default from config.DEFAULT_RATE_LIMIT)
            aggregate_seconds: Aggregation window for 1sec interval (15 or 30 seconds, default 30)
        """
        if max_requests_per_minute is None:
            max_requests_per_minute = config.DEFAULT_RATE_LIMIT
        self.api_key = api_key
        self.interval = interval
        self.period = period
        self.aggregate_seconds = aggregate_seconds
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.client = MassiveClient(api_key, self.rate_limiter, aggregate_seconds)

        # Map period to outputsize for Massive.com
        if period in ['compact', 'full']:
            self.outputsize = period
        else:
            # Default to full for maximum data
            self.outputsize = 'full'

        logger.info(f"StockTrendAnalyzer initialized:")
        logger.info(f"  - Interval: {interval}")
        logger.info(f"  - Period: {period} (outputsize: {self.outputsize})")
        logger.info(f"  - Rate limit: {max_requests_per_minute} req/min")

    def validate_ticker_format(self, ticker: str) -> tuple[bool, str]:
        """
        Validate ticker format and provide warnings for potentially invalid tickers

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (is_valid, warning_message)
        """
        # Common patterns for invalid/problematic tickers
        warnings = []

        # Check for preferred stock suffixes that might not work with intraday
        if ticker.endswith(('-', 'P', 'PR')) or any(c in ticker for c in ['.', '-']):
            if self.interval != '1d':
                warnings.append(f"Preferred stock/warrant '{ticker}' may not have intraday data")

        # Check for unusual length
        if len(ticker) > 5:
            warnings.append(f"Ticker '{ticker}' is unusually long - may be invalid")

        # Check for lowercase (should be uppercase)
        if ticker != ticker.upper():
            warnings.append(f"Ticker '{ticker}' contains lowercase - should be uppercase")

        # Check for special characters that indicate bonds/options
        if any(char in ticker for char in ['/', '=']):
            warnings.append(f"Ticker '{ticker}' contains special characters - may not be a regular stock")

        return (len(warnings) == 0, '; '.join(warnings) if warnings else '')
    
    def get_client(self):
        """Get the MassiveClient instance for external use"""
        return self.client
        
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data['Close'].rolling(window=period).mean()
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed +DI and -DI
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_momentum(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10, 30], latest_price: float = None) -> Dict:
        """Calculate price momentum over multiple periods

        For 1-day period, uses time-based lookback (24 hours earlier)
        For other periods, uses trading-day lookback (index-based)

        Args:
            data: DataFrame with OHLCV data
            periods: List of periods in days to calculate momentum for
            latest_price: Optional real-time price (including extended hours) to use instead of last close
        """
        momentum = {}

        # Use latest_price if provided (includes extended hours), otherwise use last close from data
        current_price = latest_price if latest_price is not None else data['Close'].iloc[-1]

        for period in periods:
            if period == 1:
                # For 1-day, compare current price to previous day's close
                # Use the second-to-last close as the reference point (previous trading day)
                try:
                    if len(data) >= 2:
                        # Previous day's close
                        prev_close = data['Close'].iloc[-2]
                        momentum['1d'] = ((current_price / prev_close) - 1) * 100
                    else:
                        momentum['1d'] = None
                except Exception as e:
                    logger.warning(f"Error calculating 1d momentum: {e}")
                    momentum['1d'] = None
            else:
                # For other periods, use trading-day lookback (index-based)
                if len(data) >= period:
                    momentum[f'{period}d'] = ((current_price / data['Close'].iloc[-period]) - 1) * 100
                else:
                    momentum[f'{period}d'] = None

        return momentum
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume trends"""
        if len(data) < 20:
            return {'volume_trend': None, 'avg_volume': None}
        
        recent_volume = data['Volume'].iloc[-10:].mean()
        previous_volume = data['Volume'].iloc[-20:-10].mean()
        
        volume_trend = 'Increasing' if recent_volume > previous_volume else 'Decreasing'
        
        return {
            'volume_trend': volume_trend,
            'avg_volume': recent_volume,
            'volume_change': ((recent_volume / previous_volume) - 1) * 100 if previous_volume > 0 else None
        }
    
    def is_trending_up(self, ticker: str) -> Optional[Dict]:
        """
        Analyze if a stock is trending up based on all four criteria
        Returns a dictionary with analysis results or None if data unavailable
        """
        logger.info(f"Analyzing {ticker}...")
        try:
            # Download data based on interval type
            if self.interval == '1d':
                logger.debug(f"{ticker}: Fetching daily data")
                data = self.client.get_daily_data(ticker, self.outputsize)
                daily_data = None  # Not needed for daily interval
            else:
                # Intraday intervals - fetch both intraday and daily data
                logger.debug(f"{ticker}: Fetching intraday data ({self.interval})")
                data = self.client.get_intraday_data(ticker, self.interval, self.outputsize)

                # Fetch daily data for momentum calculations
                logger.debug(f"{ticker}: Fetching daily data for momentum calculation")
                daily_data = self.client.get_daily_data(ticker, outputsize='full')

            if data is None or data.empty or len(data) < 50:
                logger.warning(f"{ticker}: Insufficient data (got {len(data) if data is not None else 0} points, need ≥50)")
                return None

            logger.debug(f"{ticker}: Data validated - {len(data)} data points available")

            # Fetch latest real-time price (includes pre-market/after-hours)
            logger.debug(f"{ticker}: Fetching latest real-time price (including extended hours)")
            latest_price = self.client.get_latest_price(ticker)
            if latest_price is not None:
                logger.debug(f"{ticker}: Using real-time price ${latest_price:.2f} (includes extended hours)")
            else:
                logger.debug(f"{ticker}: No real-time price available, using last close from data")

            # 1. MOVING AVERAGE CROSSOVERS
            logger.debug(f"{ticker}: Calculating moving averages")
            data['SMA_50'] = self.calculate_sma(data, 50) if len(data) >= 50 else None
            data['SMA_200'] = self.calculate_sma(data, 200) if len(data) >= 200 else None
            data['SMA_20'] = self.calculate_sma(data, 20) if len(data) >= 20 else None

            # Use latest real-time price if available, otherwise use last close from data
            current_price = latest_price if latest_price is not None else data['Close'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1] if data['SMA_50'].iloc[-1] is not None else 0
            sma_200 = data['SMA_200'].iloc[-1] if len(data) >= 200 and data['SMA_200'].iloc[-1] is not None else 0

            ma_bullish = False
            if sma_50 > 0:
                ma_bullish = current_price > sma_50
                if sma_200 > 0:
                    ma_bullish = ma_bullish and (sma_50 > sma_200)

            logger.debug(f"{ticker}: MA Analysis - Price: ${current_price:.2f}, SMA50: ${sma_50:.2f}, "
                        f"SMA200: ${sma_200:.2f}, Bullish: {ma_bullish}")

            # 2. PRICE MOMENTUM
            # For intraday intervals, use daily data for momentum calculations if available
            # Pass latest_price to include extended hours data in momentum calculation
            logger.debug(f"{ticker}: Calculating momentum")
            if self.interval != '1d' and daily_data is not None and not daily_data.empty:
                logger.debug(f"{ticker}: Using daily data for momentum ({len(daily_data)} days)")
                momentum = self.calculate_momentum(daily_data, latest_price=latest_price)
            else:
                momentum = self.calculate_momentum(data, latest_price=latest_price)
            momentum_positive = all(v > 0 for v in momentum.values() if v is not None)
            logger.debug(f"{ticker}: Momentum - {momentum}, All Positive: {momentum_positive}")
            
            # 3. TECHNICAL INDICATORS
            logger.debug(f"{ticker}: Calculating technical indicators (RSI, MACD, ADX)")
            rsi = self.calculate_rsi(data)
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            macd_data = self.calculate_macd(data)
            macd_bullish = macd_data['histogram'].iloc[-1] > 0 if not pd.isna(macd_data['histogram'].iloc[-1]) else False
            
            adx = self.calculate_adx(data)
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
            
            rsi_favorable = 50 <= current_rsi <= 70
            adx_strong = current_adx > 25
            
            logger.debug(f"{ticker}: Indicators - RSI: {current_rsi:.2f} (Favorable: {rsi_favorable}), "
                        f"MACD Bullish: {macd_bullish}, ADX: {current_adx:.2f} (Strong: {adx_strong})")
            
            # 4. VOLUME ANALYSIS
            logger.debug(f"{ticker}: Analyzing volume trends")
            volume_data = self.analyze_volume(data)
            volume_increasing = volume_data['volume_trend'] == 'Increasing'
            logger.debug(f"{ticker}: Volume - Trend: {volume_data['volume_trend']}, "
                        f"Change: {volume_data.get('volume_change', 'N/A')}%")
            
            # OVERALL SCORING
            logger.debug(f"{ticker}: Calculating final score")
            score = 0
            max_score = 6

            # Calculate individual score components
            ma_bullish_score = 1.5 if ma_bullish else 0
            momentum_score = 1.5 if momentum_positive else 0
            rsi_score = 1.0 if rsi_favorable else 0
            macd_score = 1.0 if macd_bullish else 0
            adx_score = 0.5 if adx_strong else 0
            volume_score = 0.5 if volume_increasing else 0

            if ma_bullish:
                score += 1.5
                logger.debug(f"{ticker}: +1.5 points for bullish moving averages")
            if momentum_positive:
                score += 1.5
                logger.debug(f"{ticker}: +1.5 points for positive momentum")
            if rsi_favorable:
                score += 1
                logger.debug(f"{ticker}: +1.0 point for favorable RSI")
            if macd_bullish:
                score += 1
                logger.debug(f"{ticker}: +1.0 point for bullish MACD")
            if adx_strong:
                score += 0.5
                logger.debug(f"{ticker}: +0.5 points for strong ADX")
            if volume_increasing:
                score += 0.5
                logger.debug(f"{ticker}: +0.5 points for increasing volume")

            # Consider it trending up if score is >= 4 out of 6
            is_trending = score >= 4.0

            logger.info(f"{ticker}: Analysis complete - Score: {score:.1f}/{max_score}, "
                       f"Trending: {is_trending}")

            return {
                'ticker': ticker,
                'is_trending': is_trending,
                'score': score,
                'max_score': max_score,
                'current_price': round(current_price, 2),
                'sma_50': round(sma_50, 2) if sma_50 > 0 else 'N/A',
                'sma_200': round(sma_200, 2) if sma_200 > 0 else 'N/A',
                'ma_bullish': ma_bullish,
                'momentum': {k: round(v, 2) if v is not None else 'N/A' for k, v in momentum.items()},
                'momentum_positive': momentum_positive,
                'rsi': round(current_rsi, 2),
                'rsi_favorable': rsi_favorable,
                'macd_bullish': macd_bullish,
                'adx': round(current_adx, 2),
                'adx_strong': adx_strong,
                'volume_trend': volume_data['volume_trend'],
                'volume_change': round(volume_data['volume_change'], 2) if volume_data['volume_change'] else 'N/A',
                'volume_increasing': volume_increasing,
                # Individual score components for detailed breakdown
                'ma_bullish_score': ma_bullish_score,
                'momentum_score': momentum_score,
                'rsi_score': rsi_score,
                'macd_score': macd_score,
                'adx_score': adx_score,
                'volume_score': volume_score
            }
            
        except Exception as e:
            logger.error(f"{ticker}: Error during analysis - {e}", exc_info=True)
            return None
    
    def scan_stocks(self, tickers: List[str], show_progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scan multiple stocks for trending patterns
        
        Args:
            tickers: List of stock tickers to analyze
            show_progress: Whether to show progress during scanning
            
        Returns:
            Tuple of (trending_df, all_results_df):
                - trending_df: DataFrame with only trending stocks (score >= 4.0)
                - all_results_df: DataFrame with ALL analyzed stocks
        """
        trending_results = []
        all_results = []
        total = len(tickers)
        
        logger.info(f"Starting scan of {total} stocks")
        logger.info(f"Configuration: interval={self.interval}, outputsize={self.outputsize}")
        print(f"\nScanning {total} stocks for uptrending patterns...")
        print(f"Using interval: {self.interval}, Massive.com API\n")
        
        scan_start_time = time.time()
        trending_count = 0
        skipped_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            if show_progress:
                if i % 10 == 0:
                    elapsed = time.time() - scan_start_time
                    avg_time = elapsed / i
                    remaining = (total - i) * avg_time
                    logger.info(f"Progress: {i}/{total} ({(i/total)*100:.1f}%) - "
                              f"Trending: {trending_count}, Skipped: {skipped_count}, "
                              f"ETA: {remaining:.0f}s")
                    print(f"Progress: {i}/{total} ({(i/total)*100:.1f}%)")
                elif i % 50 == 0:
                    print(f"Progress: {i}/{total} ({(i/total)*100:.1f}%) - "
                          f"Rate limiter is managing API calls...")
            
            result = self.is_trending_up(ticker)
            
            if result is None:
                skipped_count += 1
                logger.debug(f"Skipped {ticker} (no valid data)")
            else:
                # Add to all results
                all_results.append(result)
                
                if result['is_trending']:
                    trending_count += 1
                    trending_results.append(result)
                    logger.info(f"✓ {ticker} is trending (score: {result['score']:.1f})")
                else:
                    logger.debug(f"✗ {ticker} not trending (score: {result['score']:.1f})")
        
        scan_elapsed = time.time() - scan_start_time
        logger.info(f"Scan complete: {total} stocks in {scan_elapsed:.1f}s "
                   f"({scan_elapsed/total:.2f}s per stock)")
        logger.info(f"Results: {trending_count} trending, {skipped_count} skipped, "
                   f"{len(all_results) - trending_count} not trending")
        
        # Convert to DataFrames
        trending_df = pd.DataFrame(trending_results) if trending_results else pd.DataFrame()
        all_results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        
        if trending_df.empty:
            logger.warning("No stocks found matching the trending criteria")
            print("\nNo stocks found matching the trending criteria.")
        else:
            trending_df = trending_df.sort_values('score', ascending=False)
            logger.info(f"Returning {len(trending_df)} trending stocks")
        
        if not all_results_df.empty:
            all_results_df = all_results_df.sort_values('score', ascending=False)
        
        return trending_df, all_results_df
    
    def print_results(self, df: pd.DataFrame, detailed: bool = False):
        """Print results in a formatted way"""
        if df.empty:
            print("\nNo trending stocks found.")
            return
        
        print(f"\n{'='*80}")
        print(f"STOCKS TRENDING UP - Found {len(df)} stocks")
        print(f"{'='*80}\n")
        
        if detailed:
            for _, row in df.iterrows():
                print(f"\n{row['ticker']} - Score: {row['score']:.1f}/{row['max_score']}")
                print(f"  Price: ${row['current_price']}")
                print(f"  Moving Averages: SMA50=${row['sma_50']}, SMA200=${row['sma_200']} | Bullish: {row['ma_bullish']}")
                print(f"  Momentum: {row['momentum']} | Positive: {row['momentum_positive']}")
                print(f"  RSI: {row['rsi']} | Favorable: {row['rsi_favorable']}")
                print(f"  MACD: Bullish={row['macd_bullish']}")
                print(f"  ADX: {row['adx']} | Strong Trend: {row['adx_strong']}")
                print(f"  Volume: {row['volume_trend']} ({row['volume_change']}%)")
                print(f"  {'-'*60}")
        else:
            # Summary table
            summary = df[['ticker', 'score', 'current_price', 'rsi', 'adx', 'volume_trend']].copy()
            summary.columns = ['Ticker', 'Score', 'Price', 'RSI', 'ADX', 'Volume Trend']
            print(summary.to_string(index=False))
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, df: pd.DataFrame, filename: str = 'trending_stocks.csv'):
        """Save results to CSV file"""
        if df.empty:
            logger.warning("No results to save (empty DataFrame)")
            print("No results to save.")
            return
        
        try:
            logger.info(f"Saving {len(df)} results to {filename}")
            
            # Flatten momentum dictionary for CSV
            df_save = df.copy()
            momentum_df = pd.json_normalize(df_save['momentum'])
            df_save = df_save.drop('momentum', axis=1)
            df_save = pd.concat([df_save, momentum_df], axis=1)
            
            df_save.to_csv(filename, index=False)
            logger.info(f"Results successfully saved to {filename}")
            print(f"\nResults saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {e}", exc_info=True)
            print(f"\nError saving results: {e}")
    
    def save_all_results(self, df: pd.DataFrame, filename: str = 'all_stocks_analyzed.csv'):
        """Save all analyzed stocks (trending and non-trending) to CSV file with detailed scoring breakdown"""
        if df.empty:
            logger.warning("No results to save (empty DataFrame)")
            print("No all-results to save.")
            return

        try:
            logger.info(f"Saving {len(df)} all-results with detailed breakdown to {filename}")

            # Flatten momentum dictionary for CSV
            df_save = df.copy()
            momentum_df = pd.json_normalize(df_save['momentum'])
            df_save = df_save.drop('momentum', axis=1)
            df_save = pd.concat([df_save, momentum_df], axis=1)

            # Reorder columns to put scoring breakdown near the score
            # Get all columns
            cols = df_save.columns.tolist()

            # Define desired column order: basic info, total score, score breakdown, then technical indicators
            priority_cols = ['ticker', 'score', 'ma_bullish_score', 'momentum_score', 'rsi_score',
                           'macd_score', 'adx_score', 'volume_score', 'is_trending', 'max_score',
                           'current_price']

            # Build final column list: priority columns first, then remaining columns
            ordered_cols = [col for col in priority_cols if col in cols]
            remaining_cols = [col for col in cols if col not in priority_cols]
            final_cols = ordered_cols + remaining_cols

            df_save = df_save[final_cols]

            df_save.to_csv(filename, index=False)
            logger.info(f"All results with detailed breakdown successfully saved to {filename}")
            print(f"All analyzed stocks (with detailed scoring breakdown) saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving all results to {filename}: {e}", exc_info=True)
            print(f"\nError saving all results: {e}")


def wait_until_time(target_hour: int, target_minute: int = 0, timezone_str: str = 'US/Central'):
    """
    Wait until a specific time of day in the specified timezone

    Args:
        target_hour: Target hour (0-23)
        target_minute: Target minute (0-59)
        timezone_str: Timezone name (default: 'US/Central')
    """
    tz = pytz.timezone(timezone_str)

    while True:
        now = datetime.now(tz)
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

        # If target time has passed today, schedule for tomorrow
        if now >= target_time:
            target_time += timedelta(days=1)

        wait_seconds = (target_time - now).total_seconds()

        print(f"\n{'='*80}")
        print(f"SCHEDULED START TIME: {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Waiting {wait_seconds / 3600:.2f} hours ({wait_seconds / 60:.1f} minutes) until execution...")
        print(f"{'='*80}\n")

        # Wait until target time
        time.sleep(wait_seconds)

        # Verify we've reached the target time (in case of system clock changes)
        now = datetime.now(tz)
        if now.hour == target_hour and now.minute >= target_minute:
            print(f"\n{'='*80}")
            print(f"TARGET TIME REACHED: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Starting execution...")
            print(f"{'='*80}\n")
            break
        else:
            # Clock drift or system time change, recalculate
            print("System time changed, recalculating wait time...")
            continue


###############################
def main():
    # Base directory for input files (embedded in code)
    input_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_files')

    parser = argparse.ArgumentParser(
        description='Stock Trend Analyzer - Massive.com Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Usage
  # ------------
  # Analyze using default file (sample_ticker.txt)
  python stock_trend_analyzer.py

  # Analyze specific tickers
  python stock_trend_analyzer.py --tickers AAPL,MSFT,GOOGL

  # Load from different file
  python stock_trend_analyzer.py --file my_watchlist.txt

  # Intraday Analysis
  # -----------------
  # Intraday 5-minute analysis
  python stock_trend_analyzer.py --interval 5min

  # 1-second interval (aggregated to 30-second bars)
  python stock_trend_analyzer.py --interval 1sec --aggregate 30

  # Output Options
  # --------------
  # Custom output filename
  python stock_trend_analyzer.py --output my_analysis.csv

  # Enable debug logging (log files are created automatically in logs/ directory)
  python stock_trend_analyzer.py --log-level DEBUG

  # Loop Mode (Live Dashboard)
  # --------------------------
  # Enable continuous loop mode with live dashboard (no file outputs)
  python stock_trend_analyzer.py --loop --tickers AAPL,MSFT,GOOGL,NVDA,TSLA,META

  # Loop mode with 5-minute intraday analysis
  python stock_trend_analyzer.py --interval 5min --loop --tickers SPY,QQQ,IWM

  # Loop mode with custom watchlist file
  python stock_trend_analyzer.py --interval 5min --loop --file my_watchlist.txt

  # Scheduled Execution
  # -------------------
  # Schedule execution to start at 6:00 AM CT
  python stock_trend_analyzer.py --start-at 06:00 --file my_watchlist.txt

  # Schedule with loop mode (starts at 6:00 AM, then loops continuously)
  python stock_trend_analyzer.py --start-at 06:00 --interval 5min --loop --file my_watchlist.txt

  # Schedule daily analysis at market open (8:30 AM CT) with 1-minute updates
  python stock_trend_analyzer.py --start-at 08:30 --interval 1min --loop --tickers AAPL,MSFT,GOOGL

Note:
  Standard Mode (without --loop):
    - Detailed output and all plots are generated automatically
    - Output directories (csv/, plots/, trending_charts/, logs/) are created automatically
    - csv/trending/: Top 20 trending stocks (score >= 4.0)
    - csv/all/: All analyzed stocks with detailed scoring breakdown
    - Log files are automatically generated with timestamps and file prefixes

  Loop Mode (with --loop):
    - Displays live dashboard with top 6 tickers
    - Shows last 2 days of data only
    - No CSV or chart files generated (dashboard only)
    - Continuous updates after each analysis cycle
    - Press Ctrl+C to exit

  Scheduling (--start-at):
    - Use HH:MM format (24-hour, US/Central timezone)
    - Can be combined with --loop for automated daily monitoring
    - Example: --start-at 08:30 starts at market open (8:30 AM CT)
        """
    )
    
    parser.add_argument('--api-key', type=str, 
                        help='Massive.com API key (or set MASSIVE_API_KEY env variable)')
    parser.add_argument('--tickers', type=str, 
                        help='Comma-separated list of tickers (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--file', type=str,
                        default='sample_ticker.txt',
                        help='Ticker file name in input_files/ directory (default: sample_ticker.txt)')
    parser.add_argument('--interval', type=str, default='1d',
                        choices=['1d', '1sec', '1min', '3min', '5min', '15min', '30min', '60min'],
                        help='Data interval (default: 1d). Use 1sec for 1-second, 3min for 3-minute, 5min for 5-minute intraday')
    parser.add_argument('--period', type=str, default='full',
                        choices=['compact', 'full'],
                        help='Historical period: compact (100 days/points) or full (5 years/30 days) (default: full)')
    parser.add_argument('--aggregate', type=int, default=30,
                        choices=[15, 30],
                        help='Aggregation window in seconds for 1sec interval (default: 30)')
    parser.add_argument('--rate-limit', type=int, default=150,
                        help='API requests per minute (default: 150 for premium)')
    parser.add_argument('--output', type=str, 
                        help='Output CSV filename (default: auto-generated with timestamp)')
    parser.add_argument('--start-date', type=str, default='2025-01-01',
                        help='Start date for individual charts in YYYY-MM-DD format (default: 2025-01-01)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top trending stocks to create individual charts for (default: 10, use 0 for all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str,
                        help='Log file path (default: auto-generated in logs/ directory with timestamp and file prefix)')
    parser.add_argument('--loop', action='store_true',
                        help='Enable continuous looping mode with live dashboard (shows top 6 tickers, last 2 days, no file outputs)')
    parser.add_argument('--start-at', type=str,
                        help='Start execution at specific time in HH:MM format (24-hour, US/Central timezone). Example: --start-at 06:00')

    args = parser.parse_args()

    # Handle scheduled start time if specified
    if args.start_at:
        try:
            # Parse the time string (HH:MM format)
            time_parts = args.start_at.split(':')
            if len(time_parts) != 2:
                raise ValueError("Time must be in HH:MM format")

            target_hour = int(time_parts[0])
            target_minute = int(time_parts[1])

            # Validate hour and minute
            if not (0 <= target_hour <= 23):
                raise ValueError("Hour must be between 0 and 23")
            if not (0 <= target_minute <= 59):
                raise ValueError("Minute must be between 0 and 59")

            # Wait until the specified time
            wait_until_time(target_hour, target_minute)

        except ValueError as e:
            print(f"Error: Invalid start time format '{args.start_at}': {e}")
            print("Please use HH:MM format (24-hour), e.g., --start-at 06:00")
            return

    # Create output and logs directories
    output_dir = config.OUTPUT_DIR_MAIN
    logs_dir = os.path.join(output_dir, config.OUTPUT_DIR_LOGS)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # Generate log filename with timestamp (will be updated with file_prefix later if needed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine log filename
    if args.log_file:
        # User specified a log file
        log_filename = args.log_file
    else:
        # Auto-generate log filename in logs/ directory
        log_filename = os.path.join(logs_dir, f'stock_analyzer_{timestamp}.log')

    # Setup logging
    global logger
    logger = setup_logging(args.log_level, log_filename)
    logger.info("="*60)
    logger.info("Stock Trend Analyzer - Massive.com Edition")
    logger.info("="*60)

    # Validate and parse start date
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        logger.info(f"Start date for individual charts: {args.start_date}")
    except ValueError:
        logger.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD format.")
        print(f"Error: Invalid start date '{args.start_date}'. Please use YYYY-MM-DD format (e.g., 2025-01-01)")
        return
    
    # Log top-n setting
    if args.top_n == 0:
        logger.info("Will generate individual charts for ALL trending stocks")
    else:
        logger.info(f"Will generate individual charts for top {args.top_n} trending stocks")
    
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get('MASSIVE_API_KEY')
    
    if not api_key:
        logger.error("No API key provided")
        print("Error: Massive.com API key is required.")
        print("\nProvide it using:")
        print("  --api-key YOUR_API_KEY")
        print("  or set environment variable: export MASSIVE_API_KEY=your_key_here")
        return
    
    logger.info(f"API key found: {api_key[:8]}...")
    
    # Initialize analyzer
    logger.info("Initializing analyzer")
    analyzer = StockTrendAnalyzer(
        api_key=api_key,
        interval=args.interval,
        period=args.period,
        max_requests_per_minute=args.rate_limit,
        aggregate_seconds=args.aggregate
    )

    # Create display interval for file naming
    # For 1sec interval, use the aggregate time (e.g., "15sec" or "30sec")
    if args.interval == '1sec':
        display_interval = f'{args.aggregate}sec'
    else:
        display_interval = args.interval

    # Determine which tickers to analyze
    tickers = []
    
    if args.tickers:
        # User provided comma-separated tickers (overrides file)
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from command line: {', '.join(tickers)}")
        print(f"Analyzing custom ticker list: {', '.join(tickers)}")
    else:
        # Read tickers from file (default or user-specified)
        # Build full path from input_files directory + filename
        input_file_path = os.path.join(input_files_dir, args.file)
        try:
            logger.info(f"Reading tickers from file: {input_file_path}")
            with open(input_file_path, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith('#')]
            logger.info(f"Loaded {len(tickers)} tickers from {input_file_path}")
            print(f"Loaded {len(tickers)} tickers from {input_file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {input_file_path}")
            print(f"Error: File not found: {input_file_path}")
            print(f"Please create the file in input_files/ or use --tickers to specify stocks directly.")
            print(f"Example: python3 stock_trend_analyzer.py --tickers AAPL,MSFT,GOOGL")
            return
        except Exception as e:
            logger.error(f"Error reading file {input_file_path}: {e}")
            print(f"Error reading file: {e}")
            return
    
    if not tickers:
        logger.error("No tickers to analyze")
        print("No tickers to analyze.")
        print("The ticker file is empty or all lines are commented out.")
        return
    
    logger.info(f"Total tickers to analyze: {len(tickers)}")

    # Validate ticker formats (especially important for intraday)
    problematic_tickers = []
    for ticker in tickers:
        is_valid, warning = analyzer.validate_ticker_format(ticker)
        if not is_valid:
            problematic_tickers.append((ticker, warning))
            logger.warning(f"Ticker validation: {warning}")

    if problematic_tickers and args.interval != '1d':
        print(f"\n⚠️  Warning: Found {len(problematic_tickers)} potentially problematic ticker(s) for intraday data:")
        for ticker, warning in problematic_tickers[:5]:  # Show first 5
            print(f"  - {ticker}: {warning}")
        if len(problematic_tickers) > 5:
            print(f"  ... and {len(problematic_tickers) - 5} more")
        print("These tickers may fail during analysis. Consider reviewing your ticker list.\n")

    # Extract prefix from input file for naming
    file_prefix = None
    if not args.tickers:
        # Extract filename without extension
        input_filename = os.path.basename(args.file)
        file_prefix_raw = os.path.splitext(input_filename)[0]

        # Special handling for sample_ticker.txt
        if file_prefix_raw.lower() == 'sample_ticker':
            file_prefix = 'Sample_Ticker'
        else:
            # Use the prefix as-is for other files
            file_prefix = file_prefix_raw

        logger.info(f"Using file prefix: {file_prefix}")

        # Rename log file to include file_prefix (only if auto-generated, not user-specified)
        if not args.log_file:
            new_log_filename = os.path.join(logs_dir, f'stock_analyzer_{timestamp}_{file_prefix}.log')

            # Update the file handler to use the new filename
            # Close and remove the old file handler
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)

            # Rename the physical log file
            if os.path.exists(log_filename):
                os.rename(log_filename, new_log_filename)

            # Add new file handler with updated filename
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler = logging.FileHandler(new_log_filename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Update log_filename variable for reference
            log_filename = new_log_filename
            logger.info(f"Log file renamed to include file prefix: {new_log_filename}")

    # Check if loop mode is enabled
    if args.loop:
        # Launch live dashboard instead of single-pass analysis
        logger.info("Loop mode enabled - starting live dashboard")
        print("\n" + "="*80)
        print("LOOP MODE ENABLED")
        print("="*80)
        print("Starting live dashboard with continuous updates...")
        print("No CSV, plots, or trending_charts files will be created.")
        print("="*80 + "\n")

        # Import live dashboard module
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from live_dashboard import LiveDashboard

        # Create and run live dashboard
        # Pass both actual interval (for data fetching) and display_interval (for file naming)
        dashboard = LiveDashboard(
            analyzer=analyzer,
            tickers=tickers,
            interval=args.interval,
            outputsize=analyzer.outputsize,
            loop_delay=0,  # No delay - update immediately after each cycle
            file_prefix=file_prefix,  # Pass file prefix for naming output files
            display_interval=display_interval  # Shows aggregate time for 1sec data
        )
        dashboard.run()

        # After dashboard exits (user pressed Ctrl+C)
        logger.info("Live dashboard stopped")
        return

    # Run the analysis (normal single-pass mode)
    logger.info("Starting stock analysis")
    script_start_time = time.time()
    scan_start_time = time.time()
    trending_df, all_results_df = analyzer.scan_stocks(tickers)
    scan_elapsed_time = time.time() - scan_start_time

    logger.info(f"Analysis completed in {scan_elapsed_time:.1f} seconds")
    logger.info(f"Average time per stock: {scan_elapsed_time/len(tickers):.2f}s")

    # Print results (detailed output is always enabled)
    analyzer.print_results(trending_df, detailed=True)

    print(f"\nAnalysis completed in {scan_elapsed_time:.1f} seconds")
    print(f"Average time per stock: {scan_elapsed_time/len(tickers):.2f}s")
    print(f"Log file: {log_filename}")

    # Create organized directory structure for outputs (all under output/)
    output_dir = config.OUTPUT_DIR_MAIN
    csv_dir = os.path.join(output_dir, config.OUTPUT_DIR_CSV)  # Parent CSV directory
    csv_trending_dir = os.path.join(csv_dir, 'trending')  # For top 20 trending stocks only
    csv_all_dir = os.path.join(csv_dir, 'all')  # For all stocks with detailed scoring breakdown
    plots_dir = os.path.join(output_dir, config.OUTPUT_DIR_PLOTS)
    charts_dir = os.path.join(output_dir, config.OUTPUT_DIR_CHARTS)

    # Create directories if they don't exist
    for directory in [csv_trending_dir, csv_all_dir, plots_dir, charts_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}/")
            print(f"Created directory: {directory}/")
    
    # Save trending stocks to csv_trending/ directory - Top 20 only
    output_filename = None
    if args.output:
        # Use custom filename in csv_trending/ directory
        csv_filename = os.path.join(csv_trending_dir, os.path.basename(args.output))
        # Save only top 20 trending stocks
        top_20_df = trending_df.head(20) if len(trending_df) > 20 else trending_df
        logger.info(f"Saving top {len(top_20_df)} trending stocks to {csv_filename}")
        analyzer.save_results(top_20_df, csv_filename)
        output_filename = csv_filename
    elif not trending_df.empty:
        # Auto-save with timestamp in csv_trending/ directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Include file prefix and interval in CSV filename if available
        if file_prefix:
            csv_filename = os.path.join(csv_trending_dir, f'trending_stocks_{timestamp}_{args.interval}_{file_prefix}.csv')
        else:
            csv_filename = os.path.join(csv_trending_dir, f'trending_stocks_{timestamp}_{args.interval}.csv')

        # Save only top 20 trending stocks
        top_20_df = trending_df.head(20) if len(trending_df) > 20 else trending_df
        logger.info(f"Auto-saving top {len(top_20_df)} trending stocks to {csv_filename}")
        analyzer.save_results(top_20_df, csv_filename)
        output_filename = csv_filename

    # Save ALL stocks (trending and non-trending) with detailed scoring breakdown to csv_all/ directory
    if not all_results_df.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Include file prefix and interval in CSV filename if available
        if file_prefix:
            csv_all_filename = os.path.join(csv_all_dir, f'all_stocks_detailed_{timestamp}_{args.interval}_{file_prefix}.csv')
        else:
            csv_all_filename = os.path.join(csv_all_dir, f'all_stocks_detailed_{timestamp}_{args.interval}.csv')

        logger.info(f"Saving all {len(all_results_df)} analyzed stocks with detailed breakdown to {csv_all_filename}")
        analyzer.save_all_results(all_results_df, csv_all_filename)
    
    # Generate visualization plots (always enabled)
    if not all_results_df.empty:
        try:
            logger.info("Generating visualization plots")
            print("\nGenerating visualization plots...")
            
            # Import plotting module
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plot_results import plot_score_comparison
            
            # Determine plot filename (save to plots/ directory)
            if output_filename:
                # Extract base name and put in plots/ directory
                base_name = os.path.splitext(os.path.basename(output_filename))[0]
                plot_file = os.path.join(plots_dir, f'{base_name}_plot.png')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Include file prefix and interval in plot filename if available
                if file_prefix:
                    plot_file = os.path.join(plots_dir, f'stock_analysis_plot_{timestamp}_{display_interval}_{file_prefix}.png')
                else:
                    plot_file = os.path.join(plots_dir, f'stock_analysis_plot_{timestamp}_{display_interval}.png')
            
            # Create plot
            plot_score_comparison(all_results_df, tickers, plot_file)
            logger.info(f"Dashboard plot saved to {plot_file}")
            
        except ImportError as e:
            logger.error(f"Could not import plotting module: {e}")
            print(f"Error: Could not generate plots. Make sure matplotlib is installed.")
            print("Run: pip install matplotlib")
        except Exception as e:
            logger.error(f"Error generating plots: {e}", exc_info=True)
            print(f"Error generating plots: {e}")
    else:
        logger.warning("No data available for plotting")
        print("\nNo data available for plotting.")
    
    # Generate individual technical analysis charts (always enabled)
    # Create two folders: one for uptrending stocks and one for non-trending stocks
    if not all_results_df.empty:
        try:
            # Import individual plotting module
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from plot_individual_tickers import create_individual_plots_for_all

            # Create base subdirectory under trending_charts based on file prefix with timestamp
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            if file_prefix:
                base_dir = os.path.join(charts_dir, f'{file_prefix}_{display_interval}_{timestamp_str}')
            else:
                # Default subdirectory if using --tickers command line option
                base_dir = os.path.join(charts_dir, f'Custom_{display_interval}_{timestamp_str}')

            # Create the base subdirectory
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Created charts base directory: {base_dir}/")

            # Separate stocks into uptrending and non-trending
            # Non-trending stocks are those with score < 4.0
            non_trending_df = all_results_df[all_results_df['score'] < 4.0].copy()

            total_charts_created = 0

            # 1. Create charts for UPTRENDING stocks (score >= 4.0)
            if not trending_df.empty:
                logger.info("Generating individual technical analysis charts for uptrending stocks")

                # Filter to top N trending stocks
                if args.top_n > 0 and len(trending_df) > args.top_n:
                    stocks_to_plot = trending_df.head(args.top_n)
                    logger.info(f"Creating charts for top {args.top_n} trending stocks (out of {len(trending_df)})")
                    print(f"\nGenerating individual technical analysis charts for top {args.top_n} uptrending stocks...")
                else:
                    stocks_to_plot = trending_df
                    logger.info(f"Creating charts for all {len(trending_df)} trending stocks")
                    print(f"\nGenerating individual technical analysis charts for all {len(trending_df)} uptrending stocks...")

                # Create uptrending subdirectory
                uptrending_dir = os.path.join(base_dir, 'uptrending')
                os.makedirs(uptrending_dir, exist_ok=True)
                logger.info(f"Created uptrending charts subdirectory: {uptrending_dir}/")

                # Create individual plots for uptrending stocks
                plot_files = create_individual_plots_for_all(
                    stocks_to_plot,
                    analyzer.get_client(),
                    args.interval,
                    analyzer.outputsize,
                    uptrending_dir,
                    start_date=args.start_date,
                    display_interval=display_interval
                )

                if plot_files:
                    total_charts_created += len(plot_files)
                    logger.info(f"Uptrending charts saved to {uptrending_dir}/ directory")
                    print(f"\n✓ {len(plot_files)} uptrending charts saved to '{uptrending_dir}/' directory")
            else:
                logger.warning("No trending stocks found for individual plotting")
                print("\nNo trending stocks found (score >= 4.0) for uptrending charts.")

            # 2. Create charts for NON-TRENDING stocks (score < 4.0)
            if not non_trending_df.empty:
                logger.info("Generating individual technical analysis charts for non-trending stocks")

                # Sort non-trending by score (highest first)
                non_trending_df = non_trending_df.sort_values('score', ascending=False)

                logger.info(f"Creating charts for all {len(non_trending_df)} non-trending stocks")
                print(f"\nGenerating individual technical analysis charts for {len(non_trending_df)} non-trending stocks...")

                # Create non-trending subdirectory
                non_trending_dir = os.path.join(base_dir, 'non_trending')
                os.makedirs(non_trending_dir, exist_ok=True)
                logger.info(f"Created non-trending charts subdirectory: {non_trending_dir}/")

                # Create individual plots for non-trending stocks
                plot_files = create_individual_plots_for_all(
                    non_trending_df,
                    analyzer.get_client(),
                    args.interval,
                    analyzer.outputsize,
                    non_trending_dir,
                    start_date=args.start_date,
                    display_interval=display_interval
                )

                if plot_files:
                    total_charts_created += len(plot_files)
                    logger.info(f"Non-trending charts saved to {non_trending_dir}/ directory")
                    print(f"\n✓ {len(plot_files)} non-trending charts saved to '{non_trending_dir}/' directory")
            else:
                logger.info("No non-trending stocks to plot")
                print("\nNo non-trending stocks to plot.")

            # Summary
            if total_charts_created > 0:
                print(f"\n✓ Total: {total_charts_created} charts created in '{base_dir}/' directory")
                print(f"   Charts show data from {args.start_date} to present")

        except ImportError as e:
            logger.error(f"Could not import individual plotting module: {e}")
            print(f"Error: Could not generate individual plots. Make sure matplotlib is installed.")
            print("Run: pip install matplotlib")
        except Exception as e:
            logger.error(f"Error generating individual plots: {e}", exc_info=True)
            print(f"Error generating individual plots: {e}")
    else:
        logger.warning("No analyzed stocks available for individual plotting")
        print("\nNo analyzed stocks available for individual charts.")

    # Calculate and log total execution time
    total_elapsed_time = time.time() - script_start_time
    print(f"\n{'='*60}")
    print(f"Total execution time: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")

    logger.info("="*60)
    logger.info("Program completed successfully")
    logger.info(f"Total execution time: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.2f} minutes)")
    logger.info("="*60)


if __name__ == "__main__":
    main()
