#!/usr/bin/env python3
"""
Live Dashboard for Stock Trend Analyzer
Displays top 6 scoring tickers with real-time updates
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import pytz
from scipy.ndimage import gaussian_filter1d

# Import from the main analyzer
from stock_trend_analyzer import StockTrendAnalyzer, logger
import config

# RSI Color coding (same as plot_individual_tickers.py)
RSI_COLOR_OVERSOLD = '#FF4444'    # Red - oversold (< 30)
RSI_COLOR_BEARISH = '#FFA500'     # Orange - bearish (30-50)
RSI_COLOR_BULLISH = '#4169E1'     # Royal blue - bullish (50-80)
RSI_COLOR_OVERBOUGHT = '#FFD700'  # Gold - overbought (> 80)

# Velocity/Acceleration colors (same as inflection_detector)
SMOOTHED_COLOR = 'blue'           # Blue for smoothed price line (more visible)
VELOCITY_COLOR = 'blue'           # Blue for velocity line
ACCEL_COLOR = '#C49821'           # Gold/amber color for acceleration

# Quadrant colors for velocity/acceleration shading
COLOR_VEL_POS_ACC_POS = '#00C853'  # Bright green - rising & steepening
COLOR_VEL_POS_ACC_NEG = '#69F0AE'  # Medium green - rising but flattening
COLOR_VEL_NEG_ACC_POS = '#FF8A80'  # Medium red - falling but flattening
COLOR_VEL_NEG_ACC_NEG = '#D50000'  # Bright red - falling & steepening


def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return pd.DataFrame({
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    })


def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_rsi_color(rsi_value):
    """Get color based on RSI value"""
    if pd.isna(rsi_value):
        return 'gray'
    elif rsi_value < 30:
        return RSI_COLOR_OVERSOLD
    elif 30 <= rsi_value < 50:
        return RSI_COLOR_BEARISH
    elif 50 <= rsi_value <= 80:
        return RSI_COLOR_BULLISH
    else:  # > 80
        return RSI_COLOR_OVERBOUGHT


def calculate_smoothed_velocity_acceleration(data, sigma=3):
    """
    Calculate smoothed price, velocity (1st derivative), and acceleration (2nd derivative)
    using Gaussian smoothing - same approach as inflection_detector.

    Args:
        data: DataFrame with 'Close' column
        sigma: Gaussian smoothing parameter (default: 3)

    Returns:
        Dict with 'smoothed', 'velocity', 'acceleration' Series
    """
    prices = data['Close'].values.astype(float)

    # Smooth prices using Gaussian filter
    smoothed = gaussian_filter1d(prices, sigma=sigma)

    # Calculate derivatives
    velocity = np.gradient(smoothed)       # First derivative (rate of change)
    acceleration = np.gradient(velocity)   # Second derivative

    return {
        'smoothed': pd.Series(smoothed, index=data.index),
        'velocity': pd.Series(velocity, index=data.index),
        'acceleration': pd.Series(acceleration, index=data.index)
    }


def detect_swing_points(data, window=5):
    """
    Detect swing highs and lows and classify them as HH, HL, LH, LL.

    HH (Higher High): A swing high that is higher than the previous swing high
    LH (Lower High): A swing high that is lower than the previous swing high
    HL (Higher Low): A swing low that is higher than the previous swing low
    LL (Lower Low): A swing low that is lower than the previous swing low

    Args:
        data: DataFrame with 'High' and 'Low' columns
        window: Lookback window for swing detection (bars on each side)

    Returns:
        DataFrame with swing point classifications added
    """
    df = data.copy()
    highs = df['High'].values
    lows = df['Low'].values

    # Initialize columns
    df['swing_high'] = False
    df['swing_low'] = False
    df['swing_label'] = ''
    df['is_major_swing'] = False  # For distinguishing major vs minor

    swing_highs = []  # List of (index, price)
    swing_lows = []   # List of (index, price)

    # Detect swing highs and lows
    for i in range(window, len(df) - window):
        # Swing high: highest point in the window
        if highs[i] == max(highs[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('swing_high')] = True
            swing_highs.append((i, highs[i]))

        # Swing low: lowest point in the window
        if lows[i] == min(lows[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('swing_low')] = True
            swing_lows.append((i, lows[i]))

    # Classify swing highs as HH or LH
    prev_swing_high = None
    for idx, price in swing_highs:
        if prev_swing_high is not None:
            if price > prev_swing_high:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'HH'
            else:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'LH'
        prev_swing_high = price

    # Classify swing lows as HL or LL
    prev_swing_low = None
    for idx, price in swing_lows:
        if prev_swing_low is not None:
            if price > prev_swing_low:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'HL'
            else:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'LL'
        prev_swing_low = price

    # Mark major swings (larger window = more significant)
    major_window = window * 2
    for i in range(major_window, len(df) - major_window):
        if df.iloc[i]['swing_high']:
            if highs[i] == max(highs[i-major_window:i+major_window+1]):
                df.iloc[i, df.columns.get_loc('is_major_swing')] = True
        if df.iloc[i]['swing_low']:
            if lows[i] == min(lows[i-major_window:i+major_window+1]):
                df.iloc[i, df.columns.get_loc('is_major_swing')] = True

    return df


# Swing label colors (same as inflection_detector)
SWING_LABEL_COLORS = {
    'HH': 'darkgreen',   # Higher High - bullish
    'HL': 'green',       # Higher Low - bullish
    'LH': 'darkred',     # Lower High - bearish
    'LL': 'red'          # Lower Low - bearish
}


class LiveDashboard:
    """Live dashboard displaying top 6 scoring tickers with real-time updates"""

    def __init__(self, analyzer, tickers, interval='1d', outputsize='full', loop_delay=0, file_prefix=None, display_interval=None):
        """
        Initialize the live dashboard

        Args:
            analyzer: StockTrendAnalyzer instance
            tickers: List of ticker symbols to analyze
            interval: Data interval ('1d' or intraday like '5min')
            outputsize: 'compact' or 'full'
            loop_delay: Seconds to wait between update cycles (default: 0 for immediate updates)
            file_prefix: Prefix for output files (typically from ticker file name)
            display_interval: Display interval for file naming (defaults to interval if not provided)
        """
        self.analyzer = analyzer
        self.tickers = tickers
        self.interval = interval
        self.display_interval = display_interval if display_interval is not None else interval
        self.outputsize = outputsize
        self.loop_delay = loop_delay
        self.file_prefix = file_prefix
        self.client = analyzer.get_client()

        # State variables
        self.top_stocks = []  # List of (ticker, score, data) tuples
        self.top_6_results = []  # List of result dictionaries for top 6 stocks
        self.all_results = []  # List of ALL result dictionaries (for plotting)
        self.last_update = None
        self.cycle_count = 0

        # Create output directories (all under output/)
        output_dir = config.OUTPUT_DIR_MAIN
        self.charts_dir = os.path.join(output_dir, config.OUTPUT_DIR_CHARTS)
        self.csv_dir = os.path.join(output_dir, config.OUTPUT_DIR_CSV)
        self.plots_dir = os.path.join(output_dir, config.OUTPUT_DIR_PLOTS)

        # Add timestamp to filenames
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create all necessary directories
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Set filenames (will be reused for all updates)
        if file_prefix:
            # Live Dashboard goes directly to trending_charts/ with timestamp
            self.dashboard_filename = os.path.join(self.charts_dir, f'Live_Dashboard_{timestamp_str}_{self.display_interval}_{file_prefix}.png')
            # CSV goes to csv/ directory - will be updated with cycle number during each update
            self.csv_filename_prefix = f'Top_Stocks_{self.display_interval}_{file_prefix}'
            # Dashboard Plot goes to plots/ directory with timestamp
            self.plot_filename = os.path.join(self.plots_dir, f'stock_analysis_plot_{timestamp_str}_{self.display_interval}_{file_prefix}.png')
        else:
            self.dashboard_filename = os.path.join(self.charts_dir, f'Live_Dashboard_{timestamp_str}_{self.display_interval}.png')
            self.csv_filename_prefix = f'Top_Stocks_{self.display_interval}'
            self.plot_filename = os.path.join(self.plots_dir, f'stock_analysis_plot_{timestamp_str}_{self.display_interval}.png')

        # Create figure with heatmap on left + 2x3 grid for 6 stocks
        self.fig = plt.figure(figsize=(15, 7.5), dpi=150)

        # Create title with file prefix if available
        title = 'Live Stock Trend Dashboard - Top 6 Stocks by Score'
        if file_prefix:
            title += f' from {file_prefix}'

        self.fig.suptitle(title,
                         fontsize=8, fontweight='bold', y=0.995)

        # Adjust figure margins (heatmap at left edge)
        self.fig.subplots_adjust(top=0.94, bottom=0.06, left=0.02, right=0.96)

        # Main grid: heatmap on left (1 col), charts on right (6 cols)
        self.main_gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[1, 6], wspace=0.12)

        # Heatmap (left side, full height)
        self.heatmap_ax = self.fig.add_subplot(self.main_gs[0, 0])

        # Charts area - create 2x3 grid for 6 stocks
        charts_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=self.main_gs[0, 1],
                                                      hspace=0.35, wspace=0.25)

        # Initialize subplots (will be populated during updates)
        self.axes = []
        for row in range(2):
            for col in range(3):
                # Each stock gets 4 vertical panels (price, volume, RSI, velocity/acceleration)
                gs_sub = gridspec.GridSpecFromSubplotSpec(
                    4, 1, subplot_spec=charts_gs[row, col],
                    height_ratios=[3, 1, 1, 1], hspace=0.02
                )
                ax1 = self.fig.add_subplot(gs_sub[0])  # Price
                ax2 = self.fig.add_subplot(gs_sub[1], sharex=ax1)  # Volume
                ax3 = self.fig.add_subplot(gs_sub[2], sharex=ax1)  # RSI
                ax4 = self.fig.add_subplot(gs_sub[3], sharex=ax1)  # Velocity/Acceleration

                self.axes.append((ax1, ax2, ax3, ax4))

        plt.ion()  # Enable interactive mode

    def analyze_all_tickers(self):
        """Analyze all tickers and return top 6 by score"""
        cycle_start_time = time.time()

        # Log and print cycle header
        header = f"Dashboard Update Cycle {self.cycle_count + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        logger.info("="*80)
        logger.info(header)
        logger.info("="*80)
        print(f"\n{'='*80}")
        print(header)
        print(f"{'='*80}\n")

        results = []

        for i, ticker in enumerate(self.tickers, 1):
            status_msg = f"  [{i:3d}/{len(self.tickers)}] Analyzing {ticker}... "
            print(status_msg, end='', flush=True)

            result = self.analyzer.is_trending_up(ticker)
            if result:
                results.append(result)
                result_msg = f"✓ Score: {result['score']:.1f}/6.0"
                print(result_msg)
                logger.info(status_msg + result_msg)
            else:
                fail_msg = "❌ Failed"
                print(fail_msg)
                logger.info(status_msg + fail_msg)

            # Add empty line between tickers
            print()
            logger.info("")

        # Sort by score (highest first) and get top 6 regardless of filter status
        # This shows the top 6 stocks by score, whether trending or not
        results.sort(key=lambda r: r.get('score', 0), reverse=True)
        top_6 = results[:6]

        # Store ALL results and top 6 results for later use in plotting
        self.all_results = results  # All analyzed stocks
        self.top_6_results = top_6  # Just top 6 for the dashboard

        # Log and print top 6 header
        logger.info("")
        logger.info("="*80)
        logger.info("Top 6 Stocks by Score:")
        logger.info("="*80)
        print(f"\n{'='*80}")
        print(f"Top 6 Stocks by Score:")
        print(f"{'='*80}")

        # Calculate and display detailed metrics for each stock
        for i, result in enumerate(top_6, 1):
            ticker = result['ticker']
            score = result['score']
            price = result['current_price']

            # Extract momentum data (contains gains over different periods)
            momentum = result.get('momentum', {})
            gain_1d = momentum.get('1d', 'N/A')
            gain_5d = momentum.get('5d', 'N/A')
            gain_10d = momentum.get('10d', 'N/A')

            # Calculate rate of gain (5-day gain / 5 = average daily rate)
            if gain_5d != 'N/A' and gain_5d is not None:
                rate_of_gain = gain_5d / 5  # Average daily rate over 5 days
                rate_str = f"{rate_of_gain:+.2f}%/day"
            else:
                rate_str = "N/A"

            # Format daily gain
            if gain_1d != 'N/A' and gain_1d is not None:
                gain_1d_str = f"{gain_1d:+.2f}%"
            else:
                gain_1d_str = "N/A"

            # Calculate volatility from momentum data (std dev of gains)
            gains_list = [v for v in [gain_1d, gain_5d, gain_10d] if v != 'N/A' and v is not None]
            if len(gains_list) >= 2:
                import numpy as np
                volatility = np.std(gains_list)
                vol_str = f"{volatility:.2f}%"
            else:
                vol_str = "N/A"

            # Log and print stock metrics
            line1 = f"  {i}. {ticker:6s} - Score: {score:.1f}/6.0 | Price: ${price:.2f}"
            line2 = f"      Daily Gain: {gain_1d_str:>8s} | Rate: {rate_str:>12s} | Volatility: {vol_str:>8s}"
            print(line1)
            print(line2)
            logger.info(line1)
            logger.info(line2)

            # Add empty line between top 6 stocks (but not after the last one)
            if i < len(top_6):
                print()
                logger.info("")

        logger.info("="*80)
        print(f"{'='*80}\n")

        # Calculate and display cycle duration with requests per minute
        cycle_duration = time.time() - cycle_start_time
        cycle_minutes = cycle_duration / 60

        # Calculate requests per minute for this cycle
        # Each ticker analysis makes approximately 1 API request
        requests_this_cycle = len(self.tickers)
        requests_per_minute = (requests_this_cycle / cycle_duration) * 60 if cycle_duration > 0 else 0

        logger.info("#####################")
        logger.info(f"Cycle duration: {cycle_minutes:.2f} minutes ({cycle_duration:.1f} seconds)")
        logger.info(f"Requests per minute: {requests_per_minute:.1f} req/min")
        logger.info("#####################")
        logger.info("")
        print("#####################")
        print(f"Cycle duration: {cycle_minutes:.2f} minutes ({cycle_duration:.1f} seconds)")
        print(f"Requests per minute: {requests_per_minute:.1f} req/min")
        print("#####################\n")

        # Fetch data for top 6 stocks
        self.top_stocks = []
        for result in top_6:
            ticker = result['ticker']
            score = result['score']

            # Fetch price data
            if self.interval == '1d':
                data = self.client.get_daily_data(ticker, self.outputsize)
            else:
                data = self.client.get_intraday_data(ticker, self.interval, self.outputsize)

            if data is not None and not data.empty:
                # For intraday, convert to CT first
                if self.interval != '1d' and len(data) > 0:
                    # Convert from ET to CT
                    data.index = data.index - pd.Timedelta(hours=1)

                # Filter to last 1 day for display, but use previous day's data for indicator calculations
                # This ensures Bollinger Bands (20), RSI (14), Volume MA (50) etc. have enough data
                # to display values from the very first candle of the current day
                if self.interval != '1d':
                    # For intraday: Find the most recent pre-market start (3:00 AM CT)

                    # Get the most recent timestamp
                    latest_time = data.index[-1]

                    # Find the start of the current trading day (3:00 AM CT)
                    # If current time is before 3 AM, go back to previous day's 3 AM
                    if latest_time.hour < 3:
                        # Before 3 AM, so current trading day starts yesterday at 3 AM
                        current_day_start = latest_time.replace(hour=3, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
                    else:
                        # After 3 AM, current trading day starts today at 3 AM
                        current_day_start = latest_time.replace(hour=3, minute=0, second=0, microsecond=0)

                    # Filter data from current trading day's 3 AM onward (1 day display)
                    # but store previous day's data for indicator calculations
                    previous_day_start = current_day_start - pd.Timedelta(days=1)
                    if previous_day_start.dayofweek == 6:  # Sunday
                        previous_day_start = previous_day_start - pd.Timedelta(days=2)
                    elif previous_day_start.dayofweek == 5:  # Saturday
                        previous_day_start = previous_day_start - pd.Timedelta(days=1)

                    # Keep 2 days of data - previous day for indicator warmup, current day for display
                    full_data = data[data.index >= previous_day_start].copy()

                    # Filter to current day only for display
                    data = data[data.index >= current_day_start]

                    # Store full data for indicator calculations
                    data.attrs['full_data'] = full_data
                else:
                    # For daily: last 1 trading day display, but keep 50 days for calculations
                    full_data = data.tail(50).copy()
                    data = data.tail(1)
                    data.attrs['full_data'] = full_data

                # Filter weekends if needed
                if len(data) > 0:
                    data = data[data.index.dayofweek < 5]

                # For intraday, filter to extended market hours (3 AM to 7 PM CT)
                if self.interval != '1d' and len(data) > 0:
                    # Filter to extended market hours (3 AM to 7 PM CT)
                    data = data[(data.index.hour >= 3) & (data.index.hour < 19)]

                if not data.empty:
                    # Store ticker, score, data, and the full result dictionary for momentum access
                    self.top_stocks.append((ticker, score, data, result))

        self.cycle_count += 1
        self.last_update = datetime.now()

    def plot_stock_panel(self, ax_tuple, ticker, score, data, result=None):
        """
        Plot a single stock's 4-panel technical analysis

        Args:
            ax_tuple: Tuple of (ax1, ax2, ax3, ax4) for price, volume, RSI, velocity/acceleration
            ticker: Stock ticker symbol
            score: Trend score
            data: DataFrame with OHLCV data (1 day for display, full_data in attrs for calculations)
            result: Result dictionary containing momentum data (optional)
        """
        ax1, ax2, ax3, ax4 = ax_tuple

        # Clear all axes and any twin axes
        for ax in ax_tuple:
            # Clear any twin axes first (secondary y-axes)
            for twin in ax.figure.axes:
                if twin is not ax and twin.bbox.bounds == ax.bbox.bounds:
                    twin.remove()
            ax.clear()

        # Use full_data (2 days) for indicator calculations if available
        # This ensures indicators have enough warmup data to display from the start
        if hasattr(data, 'attrs') and 'full_data' in data.attrs:
            full_data = data.attrs['full_data']
            display_data = data  # 1 day for display

            # Calculate indicators on FULL data (2 days) for accurate values
            bb_full = calculate_bollinger_bands(full_data, period=20, num_std=2)
            rsi_full = calculate_rsi(full_data, period=14)
            sma_5_full = full_data['Close'].rolling(window=5).mean()
            sma_20_full = full_data['Close'].rolling(window=20).mean()
            volume_ma_50_full = full_data['Volume'].rolling(window=50).mean()

            # Calculate smoothed price, velocity, acceleration on FULL data
            derivatives_full = calculate_smoothed_velocity_acceleration(full_data, sigma=3)

            # Slice to display period only (align with display_data index)
            bb = {k: v.loc[display_data.index] for k, v in bb_full.items()}
            rsi = rsi_full.loc[display_data.index]
            sma_5 = sma_5_full.loc[display_data.index]
            sma_20 = sma_20_full.loc[display_data.index]
            volume_ma_50 = volume_ma_50_full.loc[display_data.index]
            smoothed = derivatives_full['smoothed'].loc[display_data.index]
            velocity = derivatives_full['velocity'].loc[display_data.index]
            acceleration = derivatives_full['acceleration'].loc[display_data.index]
            data = display_data
        else:
            # Fallback: calculate on available data
            bb = calculate_bollinger_bands(data, period=20, num_std=2)
            rsi = calculate_rsi(data, period=14)
            sma_5 = data['Close'].rolling(window=5).mean()
            sma_20 = data['Close'].rolling(window=20).mean()
            derivatives = calculate_smoothed_velocity_acceleration(data, sigma=3)
            smoothed = derivatives['smoothed']
            velocity = derivatives['velocity']
            acceleration = derivatives['acceleration']
            volume_ma_50 = data['Volume'].rolling(window=50).mean()

        # Detect swing points for HH/HL/LH/LL labels
        swing_window = 5
        data = detect_swing_points(data, window=swing_window)

        # Create sequential x-axis positions for display data
        x_positions = np.arange(len(data))

        # Fetch earnings data
        earnings_positions = []
        try:
            earnings_df = self.client.get_earnings_calendar(ticker)
            if earnings_df is not None and not earnings_df.empty:
                data_start = data.index.min()
                data_end = data.index.max()
                earnings_in_range = earnings_df[
                    (earnings_df['reportDate'] >= data_start) &
                    (earnings_df['reportDate'] <= data_end)
                ]

                for _, earnings_row in earnings_in_range.iterrows():
                    earnings_date = earnings_row['reportDate']
                    time_diffs = abs(data.index - earnings_date)
                    closest_idx = time_diffs.argmin()

                    if self.interval == '1d':
                        if time_diffs.iloc[closest_idx] <= pd.Timedelta(days=1):
                            earnings_positions.append(closest_idx)
                    else:
                        if time_diffs.iloc[closest_idx] <= pd.Timedelta(hours=4):
                            earnings_positions.append(closest_idx)
        except Exception:
            pass  # Silently skip earnings if unavailable

        # Find pre-market start positions for intraday
        premarket_start_positions = []
        market_open_positions = []
        if self.interval != '1d':
            current_date = None
            for idx, timestamp in enumerate(data.index):
                date_only = timestamp.date()
                if date_only != current_date:
                    current_date = date_only
                    if timestamp.hour >= 3:
                        premarket_start_positions.append(idx)
                    else:
                        for future_idx in range(idx, len(data)):
                            future_timestamp = data.index[future_idx]
                            if future_timestamp.date() != date_only:
                                break
                            if future_timestamp.hour >= 3:
                                premarket_start_positions.append(future_idx)
                                break

            # Find market open positions (8:30 AM CT)
            current_date = None
            for idx, timestamp in enumerate(data.index):
                date_only = timestamp.date()
                if date_only != current_date:
                    current_date = date_only
                    # Look for 8:30 AM CT (hour=8, minute>=30) or later
                    if timestamp.hour > 8 or (timestamp.hour == 8 and timestamp.minute >= 30):
                        market_open_positions.append(idx)
                    else:
                        # Look ahead for the 8:30 AM mark on this day
                        for future_idx in range(idx, len(data)):
                            future_timestamp = data.index[future_idx]
                            if future_timestamp.date() != date_only:
                                break
                            if future_timestamp.hour > 8 or (future_timestamp.hour == 8 and future_timestamp.minute >= 30):
                                market_open_positions.append(future_idx)
                                break

            # Find market close positions (3:00 PM CT = 15:00)
            market_close_positions = []
            current_date = None
            for idx, timestamp in enumerate(data.index):
                date_only = timestamp.date()
                if date_only != current_date:
                    current_date = date_only
                    # Look for 3:00 PM CT (hour=15, minute=0) or later
                    if timestamp.hour >= 15:
                        market_close_positions.append(idx)
                    else:
                        # Look ahead for the 3:00 PM mark on this day
                        for future_idx in range(idx, len(data)):
                            future_timestamp = data.index[future_idx]
                            if future_timestamp.date() != date_only:
                                break
                            if future_timestamp.hour >= 15:
                                market_close_positions.append(future_idx)
                                break

        # ============================================
        # PANEL 1: PRICE WITH BOLLINGER BANDS (CANDLESTICKS)
        # ============================================

        # Draw candlesticks (same setup as inflection_detector)
        candle_width = 0.8
        wick_width = 1.0  # Thicker wick for solid black appearance

        # FIRST PASS: Draw ALL wicks (behind bodies)
        for i in range(len(data)):
            high_price = data['High'].iloc[i]
            low_price = data['Low'].iloc[i]
            ax1.plot([x_positions[i], x_positions[i]], [low_price, high_price],
                     color='black', linewidth=wick_width, alpha=1.0, zorder=2)

        # SECOND PASS: Draw ALL bodies (on top of wicks)
        for i in range(len(data)):
            open_price = data['Open'].iloc[i]
            close_price = data['Close'].iloc[i]

            # Determine color: green for up, red for down
            if close_price >= open_price:
                body_color = '#56B05C'  # Green
            else:
                body_color = '#F77272'  # Light red

            body_bottom = min(open_price, close_price)
            body_height = abs(close_price - open_price)

            # If open == close, draw a small horizontal line (doji)
            if body_height == 0:
                ax1.plot([x_positions[i] - candle_width/2, x_positions[i] + candle_width/2],
                         [close_price, close_price], color=body_color, linewidth=1, zorder=3)
            else:
                rect = plt.Rectangle((x_positions[i] - candle_width/2, body_bottom),
                                      candle_width, body_height,
                                      facecolor=body_color, edgecolor=body_color,
                                      linewidth=0.5, alpha=1.0, zorder=3)
                ax1.add_patch(rect)

        # Bollinger Bands
        ax1.plot(x_positions, bb['upper'], 'g--', linewidth=0.8, alpha=0.7)
        ax1.plot(x_positions, bb['middle'], 'b-', linewidth=0.5)
        ax1.plot(x_positions, bb['lower'], 'r--', linewidth=0.8, alpha=0.7)
        ax1.fill_between(x_positions, bb['lower'], bb['upper'], alpha=0.1, color='gray')

        # Smoothed price line (solid blue)
        ax1.plot(x_positions, smoothed.values, label='Smoothed', color=SMOOTHED_COLOR,
                 linewidth=0.7, alpha=0.8)

        # Add pre-market, market open, and market close lines to legend if intraday
        if self.interval != '1d':
            if premarket_start_positions:
                ax1.plot([], [], color='green', linestyle='-', linewidth=1, label='Pre-Market 3:00', alpha=0.75)
            if market_open_positions:
                ax1.plot([], [], color='green', linestyle='-', linewidth=1, label='Market Open 8:30', alpha=1.0)
            if market_close_positions:
                ax1.plot([], [], color='red', linestyle='-', linewidth=1, label='Market Close 3:00 PM', alpha=1.0)

        ax1.set_ylabel('Price ($)', fontsize=5, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=4)
        ax1.tick_params(axis='y', labelsize=4)
        ax1.grid(True, alpha=0.5)

        # Add "Times are CT" label for intraday
        if self.interval != '1d':
            ax1.text(0.02, 0.02, 'Times are CT', transform=ax1.transAxes,
                    fontsize=4, style='italic', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Add latest price with timestamp at top right
        if result is not None:
            current_price = result.get('current_price', None)
            if current_price is not None:
                # Get current time in CT (Central Time)
                ct_tz = pytz.timezone('US/Central')
                current_time_ct = datetime.now(ct_tz)
                time_str = current_time_ct.strftime('%H:%M:%S CT')

                price_label = f'${current_price:.1f} @ {time_str}'
                ax1.text(0.98, 0.98, price_label, transform=ax1.transAxes,
                        fontsize=5, fontweight='bold', ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.9))

        # Title with score and daily gain
        title = f'{score:.1f} | {ticker}'
        if self.interval != '1d':
            title += f' ({self.display_interval})'

        # Add daily gain to title if available
        if result is not None:
            momentum = result.get('momentum', {})
            gain_1d = momentum.get('1d', 'N/A')
            if gain_1d != 'N/A' and gain_1d is not None:
                title += f' | {gain_1d:+.2f}%'

        ax1.set_title(title, fontsize=6, fontweight='bold', pad=2)

        # ============================================
        # PANEL 2: VOLUME WITH MOVING AVERAGES
        # ============================================

        # Volume bars - color based on price movement (close >= open = green, else red)
        # Same setup as inflection_detector
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
                  for i in range(len(data))]
        ax2.bar(x_positions, data['Volume'], color=colors, alpha=0.65, width=0.8)
        ax2.plot(x_positions, volume_ma_50, 'purple', linewidth=1.0, alpha=0.8, label='Vol MA50')

        ax2_right = ax2.twinx()
        ax2_right.plot(x_positions, sma_5, 'orange', linewidth=1.0, alpha=0.8, label='SMA5')
        ax2_right.plot(x_positions, sma_20, 'blue', linewidth=1.0, alpha=0.8, label='SMA20')

        ax2.set_ylabel('Volume', fontsize=5, fontweight='bold')
        ax2_right.set_ylabel('Price ($)', fontsize=5, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=4)
        ax2_right.tick_params(axis='y', labelsize=4)
        ax2.grid(True, alpha=0.5)
        # Format volume axis in K (thousands)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

        # Add legend for volume panel (combine both axes)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=4)

        # ============================================
        # PANEL 3: RSI
        # ============================================

        for i in range(len(rsi) - 1):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i + 1]):
                continue
            x = [x_positions[i], x_positions[i + 1]]
            y = [rsi.iloc[i], rsi.iloc[i + 1]]
            color = get_rsi_color(rsi.iloc[i])
            ax3.plot(x, y, color=color, linewidth=1.0)

        # RSI horizontal lines (same as inflection_detector)
        ax3.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax3.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax3.axhline(y=50, color='magenta', linestyle='-', linewidth=1, alpha=0.8)

        # Shade overbought/oversold regions (same as inflection_detector)
        ax3.fill_between(x_positions, 70, 100, alpha=0.1, color='red')
        ax3.fill_between(x_positions, 0, 30, alpha=0.1, color='green')

        ax3.set_ylabel('RSI', fontsize=5, fontweight='bold')
        ax3.tick_params(axis='y', labelsize=4)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.5)

        # Current RSI display
        if not pd.isna(rsi.iloc[-1]):
            current_rsi = rsi.iloc[-1]
            rsi_color = get_rsi_color(current_rsi)
            ax3.text(0.98, 0.95, f'RSI: {current_rsi:.1f}',
                    transform=ax3.transAxes, fontsize=5, fontweight='bold',
                    ha='right', va='top', color=rsi_color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ============================================
        # PANEL 4: VELOCITY & ACCELERATION
        # ============================================

        # Velocity line (left Y axis) - blue, same as smoothed price
        ax4.plot(x_positions, velocity.values, color=VELOCITY_COLOR,
                 linewidth=1, label='Velocity', alpha=1.0)
        ax4.axhline(y=0, color=VELOCITY_COLOR, linestyle='-', linewidth=0.8, alpha=1.0)
        ax4.set_ylabel('Vel', color=VELOCITY_COLOR, fontsize=4)
        ax4.tick_params(axis='y', labelcolor=VELOCITY_COLOR, labelsize=4)
        ax4.grid(True, alpha=0.3)

        # Acceleration line (right Y axis) - gold/amber
        ax4_right = ax4.twinx()
        ax4_right.plot(x_positions, acceleration.values, color=ACCEL_COLOR,
                       linewidth=1, label='Acceleration', alpha=0.8)
        ax4_right.axhline(y=0, color=ACCEL_COLOR, linestyle='-', linewidth=0.8, alpha=1.0)
        ax4_right.set_ylabel('Acc', color=ACCEL_COLOR, fontsize=4)
        ax4_right.tick_params(axis='y', labelcolor=ACCEL_COLOR, labelsize=4)

        # Shaded regions based on velocity/acceleration sign combinations
        # Same colors as inflection_detector
        vel_values = velocity.values
        acc_values = acceleration.values

        for i in range(len(vel_values)):
            vel_pos = vel_values[i] > 0
            acc_pos = acc_values[i] > 0

            if vel_pos and acc_pos:
                color = COLOR_VEL_POS_ACC_POS  # Bright green - rising & steepening
            elif vel_pos and not acc_pos:
                color = COLOR_VEL_POS_ACC_NEG  # Medium green - rising but flattening
            elif not vel_pos and acc_pos:
                color = COLOR_VEL_NEG_ACC_POS  # Medium red - falling but flattening
            else:
                color = COLOR_VEL_NEG_ACC_NEG  # Bright red - falling & steepening

            # Shade a vertical bar for this data point
            if i < len(x_positions):
                width = 1 if i == len(x_positions) - 1 else x_positions[min(i+1, len(x_positions)-1)] - x_positions[i]
                ax4.axvspan(x_positions[i], x_positions[i] + width,
                           facecolor=color, alpha=0.35, zorder=0)

        # =================================================================
        # Draw vertical lines (same as inflection_detector)
        # SOLID lines at velocity zero crossings
        # DASHED lines at quadrant changes (excluding velocity crossings)
        # =================================================================
        axes_for_vlines = [ax1, ax2, ax3, ax4]

        # Detect velocity zero crossings
        velocity_crossings = []  # List of (index, is_bullish) tuples
        for i in range(1, len(vel_values)):
            if vel_values[i-1] * vel_values[i] < 0:  # Sign change = zero crossing
                is_bullish = vel_values[i-1] < 0 and vel_values[i] > 0
                velocity_crossings.append((i, is_bullish))

        velocity_crossing_indices = set(idx for idx, _ in velocity_crossings)

        # Draw SOLID vertical lines at velocity zero crossings
        for idx, is_bullish in velocity_crossings:
            line_color = RSI_COLOR_BULLISH if is_bullish else 'darkorange'
            for ax in axes_for_vlines:
                ax.axvline(x=x_positions[idx], color=line_color, linestyle='-', alpha=1.0, linewidth=0.5)

        # Note: Dashed vertical lines at quadrant changes removed - only velocity zero crossings shown

        ax4.set_xlabel('Time', fontsize=5, fontweight='bold')

        # ============================================
        # X-AXIS FORMATTING (ALL PANELS)
        # ============================================

        # Calculate tick positions
        num_ticks = min(6, len(data))
        tick_positions = np.linspace(0, len(data) - 1, num_ticks, dtype=int)

        if self.interval != '1d':
            tick_labels = [data.index[i].strftime('%m/%d\n%H:%M') for i in tick_positions]
        else:
            tick_labels = [data.index[i].strftime('%m/%d') for i in tick_positions]

        # Calculate bars for 1-hour gap on the right based on interval
        if self.interval == '1d':
            bars_for_gap = 1  # 1 day for daily charts
        elif self.interval in ['5min', '3min']:
            bars_for_gap = 12  # 12 bars = 1 hour for 5min
        elif self.interval == '15min':
            bars_for_gap = 4  # 4 bars = 1 hour for 15min
        elif self.interval == '30min':
            bars_for_gap = 2  # 2 bars = 1 hour for 30min
        elif self.interval == '60min' or self.interval == '1hour':
            bars_for_gap = 1  # 1 bar = 1 hour for 60min
        else:
            bars_for_gap = 12  # Default to 1 hour worth
        x_max = len(data) - 1 + bars_for_gap

        for ax in ax_tuple:
            ax.set_xlim(0, x_max)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5)

            # Add earnings markers
            if earnings_positions:
                for earnings_pos in earnings_positions:
                    ax.axvline(x=earnings_pos, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
                    y_min, y_max = ax.get_ylim()
                    y_range = y_max - y_min
                    marker_y = y_min + (y_range * 0.05)
                    ax.text(earnings_pos, marker_y, 'E',
                           ha='center', va='center',
                           fontsize=6, fontweight='bold', color='white',
                           bbox=dict(boxstyle='circle,pad=0.3', facecolor='purple',
                                   edgecolor='yellow', linewidth=1, alpha=0.95),
                           zorder=1000)

            # Add pre-market markers
            if premarket_start_positions:
                for premarket_pos in premarket_start_positions:
                    ax.axvline(x=premarket_pos, color='green', linestyle='-', linewidth=1, alpha=0.75, zorder=1)

            # Add market open markers (8:30 AM CT) - green vertical line with linewidth=1
            if market_open_positions:
                for market_open_pos in market_open_positions:
                    ax.axvline(x=market_open_pos, color='green', linestyle='-', linewidth=1, alpha=1.0, zorder=2)

            # Add market close markers (3:00 PM CT) - red vertical line with linewidth=1
            if market_close_positions:
                for market_close_pos in market_close_positions:
                    ax.axvline(x=market_close_pos, color='red', linestyle='-', linewidth=1, alpha=1.0, zorder=2)

        # Hide x-axis labels for top three panels (ax4 shows labels at bottom)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)

    def update_heatmap(self):
        """Update the score heatmap on the left side"""
        self.heatmap_ax.clear()

        if not self.all_results:
            self.heatmap_ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                fontsize=8, transform=self.heatmap_ax.transAxes)
            self.heatmap_ax.axis('off')
            return

        # Sort by score (descending)
        sorted_results = sorted(self.all_results, key=lambda x: x['score'], reverse=True)

        # Get tickers and scores
        tickers = [r['ticker'] for r in sorted_results]
        scores = [r['score'] for r in sorted_results]

        # Get daily gains (1d momentum)
        daily_gains = []
        for r in sorted_results:
            momentum = r.get('momentum', {})
            gain = momentum.get('1d', 0)
            if isinstance(gain, str):
                gain = 0
            daily_gains.append(gain)

        # Create color map based on daily gain (green for positive, red for negative)
        max_abs_gain = max(abs(min(daily_gains)) if daily_gains else 1,
                          abs(max(daily_gains)) if daily_gains else 1, 0.1)

        colors = []
        for gain in daily_gains:
            if gain >= 0:
                # Green intensity based on gain magnitude
                intensity = min(gain / max_abs_gain, 1.0)
                colors.append((0.2, 0.6 + 0.4 * intensity, 0.2))  # Green shades
            else:
                # Red intensity based on loss magnitude
                intensity = min(abs(gain) / max_abs_gain, 1.0)
                colors.append((0.8 + 0.2 * intensity, 0.2, 0.2))  # Red shades

        # Create horizontal bars
        y_positions = np.arange(len(tickers))
        bars = self.heatmap_ax.barh(y_positions, scores, color=colors, edgecolor='black', linewidth=0.5, height=0.8)

        # Add ticker labels and gain percentages
        for i, (ticker, score, gain) in enumerate(zip(tickers, scores, daily_gains)):
            # Ticker label on the left
            self.heatmap_ax.text(-0.3, i, ticker, ha='right', va='center', fontsize=7, fontweight='bold')
            # Score value inside the bar
            self.heatmap_ax.text(score - 0.1, i, f'{score:.1f}', ha='right', va='center',
                                fontsize=6, color='white', fontweight='bold')
            # Daily gain on the right
            gain_color = 'green' if gain >= 0 else 'red'
            self.heatmap_ax.text(6.2, i, f'{gain:+.1f}%', ha='left', va='center',
                                fontsize=6, color=gain_color, fontweight='bold')

        # Formatting
        self.heatmap_ax.set_xlim(-0.5, 7)
        self.heatmap_ax.set_ylim(-0.5, len(tickers) - 0.5)
        self.heatmap_ax.invert_yaxis()  # Highest score at top
        self.heatmap_ax.set_yticks([])
        self.heatmap_ax.set_xlabel('Score (out of 6)', fontsize=6, fontweight='bold')
        self.heatmap_ax.set_title('All Stocks by Score\n(Color = Daily Gain)', fontsize=6, fontweight='bold')
        self.heatmap_ax.axvline(x=4.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Threshold')
        self.heatmap_ax.grid(axis='x', alpha=0.3)

    def update_dashboard(self):
        """Update the dashboard with latest data"""
        # Refresh the figure canvas to ensure clean redraw
        self.fig.canvas.flush_events()

        # Analyze all tickers and get top 6
        self.analyze_all_tickers()

        # Update the heatmap
        self.update_heatmap()

        # Plot each of the top 6 stocks
        for i, (ticker, score, data, result) in enumerate(self.top_stocks):
            if i < 6:  # Safety check
                self.plot_stock_panel(self.axes[i], ticker, score, data, result)

        # Clear any unused panels (if less than 6 stocks)
        for i in range(len(self.top_stocks), 6):
            for ax in self.axes[i]:
                ax.clear()
                ax.axis('off')

        # Update main title with timestamp
        if self.last_update:
            title = 'Live Stock Trend Dashboard - Top 6 Stocks by Score'
            if self.file_prefix:
                title += f' from {self.file_prefix}'
            title += f' | Last Update: {self.last_update.strftime("%Y-%m-%d %H:%M:%S")} | Cycle: {self.cycle_count}'

            self.fig.suptitle(title,
                             fontsize=6, fontweight='bold', y=0.995)

        # Save top 6 stocks to CSV file (overwrites previous version)
        # Use same format as trending_stocks CSV from main analyzer
        if self.top_6_results:
            csv_data = []
            for result in self.top_6_results:
                # Extract all fields from result dictionary
                ticker = result['ticker']
                is_trending = result.get('is_trending', True)
                score = result['score']
                max_score = result.get('max_score', 6)
                current_price = result.get('current_price', 'N/A')
                sma_50 = result.get('sma_50', 'N/A')
                sma_200 = result.get('sma_200', 'N/A')
                ma_bullish = result.get('ma_bullish', False)
                momentum_positive = result.get('momentum_positive', False)
                rsi = result.get('rsi', 'N/A')
                rsi_favorable = result.get('rsi_favorable', False)
                macd_bullish = result.get('macd_bullish', False)
                adx = result.get('adx', 'N/A')
                adx_strong = result.get('adx_strong', False)
                volume_trend = result.get('volume_trend', 'N/A')
                volume_change = result.get('volume_change', 'N/A')
                volume_increasing = result.get('volume_increasing', False)

                # Extract momentum gains (1d, 5d, 10d, 30d)
                momentum = result.get('momentum', {})
                gain_1d = momentum.get('1d', 'N/A')
                gain_5d = momentum.get('5d', 'N/A')
                gain_10d = momentum.get('10d', 'N/A')
                gain_30d = momentum.get('30d', 'N/A')

                csv_data.append({
                    'ticker': ticker,
                    'is_trending': is_trending,
                    'score': score,
                    'max_score': max_score,
                    'current_price': current_price,
                    'sma_50': sma_50,
                    'sma_200': sma_200,
                    'ma_bullish': ma_bullish,
                    'momentum_positive': momentum_positive,
                    'rsi': rsi,
                    'rsi_favorable': rsi_favorable,
                    'macd_bullish': macd_bullish,
                    'adx': adx,
                    'adx_strong': adx_strong,
                    'volume_trend': volume_trend,
                    'volume_change': volume_change,
                    'volume_increasing': volume_increasing,
                    '1d': gain_1d,
                    '5d': gain_5d,
                    '10d': gain_10d,
                    '30d': gain_30d
                })

            csv_df = pd.DataFrame(csv_data)
            try:
                # Overwrite the same CSV file each cycle
                csv_filename = os.path.join(self.csv_dir, f'{self.csv_filename_prefix}.csv')
                csv_df.to_csv(csv_filename, index=False)
                logger.info(f"CSV file saved to: {csv_filename} (Cycle {self.cycle_count})")
            except Exception as e:
                logger.error(f"Error saving CSV file: {e}")

        # Save the dashboard to file (overwrites previous version)
        try:
            self.fig.savefig(self.dashboard_filename, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Dashboard saved to: {self.dashboard_filename}")
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")

        # Create and save summary plot using the same format as main analyzer (overwrites previous version)
        if self.all_results:
            try:
                import matplotlib.patches as mpatches
                import matplotlib.gridspec as gridspec_plot

                # Color scheme (same as plot_results.py)
                COLOR_PASS = '#4CAF50'  # Green for passing stocks
                COLOR_FAIL = '#F44336'  # Red for failing stocks
                COLOR_THRESHOLD = '#FF9800'  # Orange for threshold line
                COLOR_GRID = '#E0E0E0'  # Light gray for grid

                # Create figure with multiple subplots (3x2 grid)
                # Use figure number 2 to keep it separate from the main dashboard (figure 1)
                # Reduced size from (16, 12) to (10, 7.5) for better readability
                # Set DPI to 150 to match saved file appearance in interactive display
                plot_fig = plt.figure(num=2, figsize=(10, 7.5), dpi=150)
                plot_fig.clear()  # Clear previous content if figure already exists
                gs = plot_fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

                # Extract data from ALL results (not just top 6)
                tickers_list = [r['ticker'] for r in self.all_results]
                scores_list = [r['score'] for r in self.all_results]
                colors = [COLOR_PASS if s >= 4.0 else COLOR_FAIL for s in scores_list]

                # 1. MAIN PLOT: Score Bar Chart (ALL stocks, sorted by daily gain)
                ax1 = plot_fig.add_subplot(gs[0, :])
                bars = ax1.bar(range(len(tickers_list)), scores_list, color=colors, edgecolor='black', linewidth=1.0)
                ax1.axhline(y=4.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=1.5, label='Threshold (4.0)')
                ax1.set_xlabel('Stock Ticker (Sorted by Daily Gain)', fontsize=7, fontweight='bold')
                ax1.set_ylabel('Trend Score (out of 6.0)', fontsize=7, fontweight='bold')
                ax1.set_title(f'Stock Trend Analysis: All {len(tickers_list)} Stocks (Sorted by Daily Gain)', fontsize=9, fontweight='bold', pad=10)
                ax1.set_xticks(range(len(tickers_list)))
                ax1.set_xticklabels(tickers_list, rotation=45, ha='right')
                ax1.grid(axis='y', alpha=0.3, color=COLOR_GRID)
                ax1.set_ylim(0, 6.5)

                # Add score labels on bars
                for i, (bar, score) in enumerate(zip(bars, scores_list)):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{score:.1f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

                # Legend
                pass_count = len([s for s in scores_list if s >= 4.0])
                fail_count = len([s for s in scores_list if s < 4.0])
                pass_patch = mpatches.Patch(color=COLOR_PASS, label=f'Trending ({pass_count} stocks)')
                fail_patch = mpatches.Patch(color=COLOR_FAIL, label=f'Not Trending ({fail_count} stocks)')
                threshold_line = mpatches.Patch(color=COLOR_THRESHOLD, label='Threshold (4.0)')
                ax1.legend(handles=[pass_patch, fail_patch, threshold_line], loc='upper right', fontsize=6)

                # Reduce tick label sizes
                ax1.tick_params(axis='both', which='major', labelsize=6)

                # 2. SCORE DISTRIBUTION
                ax2 = plot_fig.add_subplot(gs[1, 0])
                score_ranges = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6']
                score_counts = [
                    len([s for s in scores_list if 0 <= s < 1]),
                    len([s for s in scores_list if 1 <= s < 2]),
                    len([s for s in scores_list if 2 <= s < 3]),
                    len([s for s in scores_list if 3 <= s < 4]),
                    len([s for s in scores_list if 4 <= s < 5]),
                    len([s for s in scores_list if 5 <= s <= 6])
                ]
                colors_dist = [COLOR_FAIL] * 4 + [COLOR_PASS] * 2
                ax2.bar(score_ranges, score_counts, color=colors_dist, edgecolor='black', linewidth=1.0)
                ax2.set_xlabel('Score Range', fontsize=7, fontweight='bold')
                ax2.set_ylabel('Number of Stocks', fontsize=7, fontweight='bold')
                ax2.set_title('Score Distribution', fontsize=8, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3, color=COLOR_GRID)
                ax2.tick_params(axis='both', which='major', labelsize=6)

                # Add count labels
                for i, count in enumerate(score_counts):
                    if count > 0:
                        ax2.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=6, fontweight='bold')

                # 3. CRITERIA SUCCESS RATE (Trending Stocks Only - score >= 4.0)
                ax3 = plot_fig.add_subplot(gs[1, 1])

                # Filter for trending stocks only (score >= 4.0)
                trending_stocks = [r for r in self.all_results if r['score'] >= 4.0]

                if trending_stocks:
                    # Count how many trending stocks pass each criterion
                    total_trending = len(trending_stocks)

                    criteria_counts = {
                        'MA Bullish': sum(1 for r in trending_stocks if r.get('ma_bullish', False)),
                        'Momentum+': sum(1 for r in trending_stocks if r.get('momentum_positive', False)),
                        'RSI Favorable': sum(1 for r in trending_stocks if r.get('rsi_favorable', False)),
                        'MACD Bullish': sum(1 for r in trending_stocks if r.get('macd_bullish', False)),
                        'ADX Strong': sum(1 for r in trending_stocks if r.get('adx_strong', False)),
                        'Volume↑': sum(1 for r in trending_stocks if r.get('volume_increasing', False))
                    }

                    # Calculate success rate as percentage
                    criteria_names = list(criteria_counts.keys())
                    success_rates = [(criteria_counts[c] / total_trending) * 100 for c in criteria_names]

                    # Create horizontal bar chart
                    y_pos = np.arange(len(criteria_names))
                    bars = ax3.barh(y_pos, success_rates, color=COLOR_PASS, edgecolor='black', linewidth=1.0, alpha=0.8)

                    ax3.set_yticks(y_pos)
                    ax3.set_yticklabels(criteria_names, fontsize=6)
                    ax3.set_xlabel('Success Rate (%)', fontsize=7, fontweight='bold')
                    ax3.set_title(f'Criteria Success Rate (Trending, n={total_trending})', fontsize=8, fontweight='bold')
                    ax3.grid(axis='x', alpha=0.3, color=COLOR_GRID)
                    ax3.set_xlim(0, 105)
                    ax3.invert_yaxis()
                    ax3.tick_params(axis='x', which='major', labelsize=6)

                    # Add percentage labels
                    for i, (rate, count) in enumerate(zip(success_rates, [criteria_counts[c] for c in criteria_names])):
                        ax3.text(rate + 2, i, f'{rate:.1f}% ({count}/{total_trending})',
                                va='center', fontsize=6, fontweight='bold')
                else:
                    ax3.text(0.5, 0.5, 'No trending stocks (score ≥ 4.0)',
                            ha='center', va='center', transform=ax3.transAxes, fontsize=8)

                # 4. STATISTICS TABLE
                ax4 = plot_fig.add_subplot(gs[2, 0])
                ax4.axis('off')

                total_stocks = len(scores_list)
                passing_count = len([s for s in scores_list if s >= 4.0])
                failing_count = total_stocks - passing_count
                avg_score = np.mean(scores_list)
                median_score = np.median(scores_list)
                max_score = max(scores_list)
                min_score = min(scores_list)

                stats_text = f"""
    STATISTICS
    {'='*25}

    Total:      {total_stocks}
    Trending:   {passing_count} ({passing_count/total_stocks*100:.1f}%)
    Not Trend:  {failing_count} ({failing_count/total_stocks*100:.1f}%)

    Avg Score:  {avg_score:.2f}
    Median:     {median_score:.2f}
    Max:        {max_score:.2f}
    Min:        {min_score:.2f}

    Threshold:  4.0/6.0
    """

                ax4.text(0.05, 0.5, stats_text, fontsize=6.5, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round',
                        facecolor='wheat', alpha=0.3))

                # 5. DAILY PRICE MOVEMENT (Top 10 by Daily Gain)
                ax5 = plot_fig.add_subplot(gs[2, 1])

                # Get top 10 stocks by daily gain (1d momentum) for Daily Price Movement panel
                # Sort by 1d gain in descending order (highest gain first)
                def get_daily_gain(result):
                    gain = result.get('momentum', {}).get('1d', None)
                    if gain is None or gain == 'N/A':
                        return -float('inf')
                    return gain

                top_10_results = sorted(self.all_results, key=get_daily_gain, reverse=True)[:10]
                top_10_tickers = [r['ticker'] for r in top_10_results]

                gains_list = []
                for result in top_10_results:
                    momentum = result.get('momentum', {})
                    gain_1d = momentum.get('1d', None)
                    if gain_1d is not None and gain_1d != 'N/A':
                        gains_list.append(gain_1d)
                    else:
                        gains_list.append(0)

                gain_colors = ['green' if g >= 0 else 'red' for g in gains_list]
                y_pos = np.arange(len(top_10_tickers))
                ax5.barh(y_pos, gains_list, color=gain_colors, alpha=0.7, edgecolor='black', linewidth=1.0)
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels(top_10_tickers, fontsize=6)
                ax5.set_xlabel('Daily Gain (%)', fontsize=7, fontweight='bold')
                ax5.set_title('Daily Price Movement (Top 10)', fontsize=8, fontweight='bold')
                ax5.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax5.grid(axis='x', alpha=0.3, color=COLOR_GRID)
                ax5.invert_yaxis()
                ax5.tick_params(axis='x', which='major', labelsize=6)

                # Add gain labels on bars
                for i, (ticker, gain) in enumerate(zip(top_10_tickers, gains_list)):
                    if gain != 0:
                        label_x = gain + (0.2 if gain > 0 else -0.2)
                        ax5.text(label_x, i, f'{gain:+.2f}%', va='center',
                                ha='left' if gain > 0 else 'right', fontsize=6, fontweight='bold')

                # Add overall title with timestamp and cycle number
                title = f'Stock Trend Analysis Dashboard'
                if self.file_prefix:
                    title += f' - {self.file_prefix}'
                title += f' | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Cycle: {self.cycle_count}'
                plot_fig.suptitle(title, fontsize=10, fontweight='bold', y=0.995)

                plt.tight_layout()
                plot_fig.savefig(self.plot_filename, dpi=150, bbox_inches='tight', facecolor='white')

                # Close the figure without displaying (save only)
                plt.close(plot_fig)

                logger.info(f"Summary plot saved to: {self.plot_filename}")
            except Exception as e:
                logger.error(f"Error saving summary plot: {e}")

        # Draw and update both figure windows
        plt.draw()
        plt.pause(0.1)

    def run(self):
        """Run the live dashboard in continuous loop"""
        print("\n" + "="*80)
        print("STARTING LIVE DASHBOARD")
        print("="*80)
        print(f"Mode: Continuous looping (updates immediately after each analysis cycle)")
        print(f"Displaying: Top 6 stocks by score")
        print(f"Data interval: {self.display_interval}")
        print(f"Time window: Last 1 day")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")

        try:
            while True:
                # Update dashboard
                self.update_dashboard()

                # Optional delay between updates (0 = immediate, or set to N seconds if desired)
                if self.loop_delay > 0:
                    print(f"\nWaiting {self.loop_delay} seconds before next update...")
                    time.sleep(self.loop_delay)
                else:
                    # No delay - start next cycle immediately
                    print(f"\nStarting next analysis cycle...")
                    time.sleep(0.5)  # Small pause for readability

        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("Dashboard stopped by user")
            print(f"Total cycles completed: {self.cycle_count}")

            # Save both dashboards before closing
            try:
                self.fig.savefig(self.dashboard_filename, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"Live dashboard saved to: {self.dashboard_filename}")
            except Exception as e:
                print(f"Error saving live dashboard: {e}")

            # Save stock trend analysis dashboard if it exists
            try:
                fig2 = plt.figure(2)
                if fig2.get_axes():  # Only save if it has content
                    fig2.savefig(self.plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"Stock trend analysis dashboard saved to: {self.plot_filename}")
            except Exception as e:
                print(f"Error saving stock trend analysis dashboard: {e}")

            print("="*80)
            # Close both figures
            plt.close(self.fig)  # Main dashboard (figure 1)
            plt.close(2)  # Stock analysis plot (figure 2)
