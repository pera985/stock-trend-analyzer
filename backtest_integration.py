#!/usr/bin/env python3
"""
Walk-Forward Backtesting Integration for Stock Trend Analyzer

This script connects the Stock Trend Analyzer's 6-point scoring system with
walk-forward backtesting to evaluate historical performance and optimize
the TRENDING_THRESHOLD parameter.

Usage:
    python3 backtest_integration.py --tickers AAPL,MSFT,GOOGL --years 3
    python3 backtest_integration.py --file input_files/watchlist.txt --years 5
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging

# Add parent directory to path for shared_modules import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared_modules.backtesting import WalkForwardBacktester
from stock_trend_analyzer import StockTrendAnalyzer
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockTrendBacktester:
    """
    Walk-forward backtesting integration for Stock Trend Analyzer.

    Tests different TRENDING_THRESHOLD values (3.5, 4.0, 4.5, 5.0) to find
    optimal scoring threshold over historical data.
    """

    def __init__(self, api_key, initial_capital=100000):
        """
        Initialize backtester.

        Args:
            api_key: Polygon.io API key
            initial_capital: Starting capital for backtest
        """
        self.api_key = api_key
        self.initial_capital = initial_capital
        self.analyzer = StockTrendAnalyzer(
            api_key=api_key,
            interval='1d',
            period='full'
        )

    def fetch_historical_data(self, tickers, years=5):
        """
        Fetch historical daily data for backtesting.

        Args:
            tickers: List of ticker symbols
            years: Number of years of historical data

        Returns:
            dict: {ticker: DataFrame} with OHLCV data
        """
        logger.info(f"Fetching {years} years of data for {len(tickers)} tickers...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)

        historical_data = {}

        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}...")
                # Use analyzer's existing data fetching (already adjusted=True)
                df = self.analyzer.get_daily_data(ticker, outputsize='full')

                if df is not None and len(df) > 0:
                    # Filter to requested date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    historical_data[ticker] = df
                    logger.info(f"  ✓ {ticker}: {len(df)} days of data")
                else:
                    logger.warning(f"  ✗ {ticker}: No data available")

            except Exception as e:
                logger.error(f"  ✗ {ticker}: Error fetching data - {e}")

        logger.info(f"Successfully fetched data for {len(historical_data)} tickers")
        return historical_data

    def optimize_threshold(self, train_data, thresholds=[3.5, 4.0, 4.5, 5.0]):
        """
        Optimize TRENDING_THRESHOLD on training data.

        Args:
            train_data: dict of {ticker: DataFrame} for training period
            thresholds: List of threshold values to test

        Returns:
            float: Best threshold value
        """
        best_threshold = 4.0
        best_score = -float('inf')

        logger.info(f"Testing thresholds: {thresholds}")

        for threshold in thresholds:
            # Simulate strategy with this threshold
            total_return = 0
            num_signals = 0

            for ticker, df in train_data.items():
                try:
                    # Calculate technical indicators
                    signals = self._calculate_signals(df, threshold)

                    # Calculate returns for this ticker
                    if len(signals) > 0:
                        returns = self._calculate_returns(df, signals)
                        total_return += returns
                        num_signals += len(signals)

                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    continue

            # Score = average return per signal
            avg_return = total_return / max(num_signals, 1)

            logger.info(f"  Threshold {threshold}: Avg Return {avg_return:.2%}, Signals: {num_signals}")

            if avg_return > best_score:
                best_score = avg_return
                best_threshold = threshold

        logger.info(f"✓ Best threshold: {best_threshold} (Avg Return: {best_score:.2%})")
        return best_threshold

    def _calculate_signals(self, df, threshold):
        """
        Calculate BUY signals based on trending score.

        Args:
            df: DataFrame with OHLCV data
            threshold: TRENDING_THRESHOLD value

        Returns:
            list: List of (date, price) tuples for BUY signals
        """
        signals = []

        # Calculate moving averages
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Simple scoring logic (simplified from full analyzer)
        for i in range(200, len(df)):  # Start after 200 days for MA200
            score = 0

            # MA Bullish (1.5 points)
            if df.iloc[i]['close'] > df.iloc[i]['SMA50'] and \
               df.iloc[i]['SMA50'] > df.iloc[i]['SMA200']:
                score += 1.5

            # RSI Favorable (1.0 points)
            if 50 <= df.iloc[i]['RSI'] <= 70:
                score += 1.0

            # Simple momentum check (1.5 points)
            if df.iloc[i]['close'] > df.iloc[i-5]['close']:
                score += 1.5

            # Volume check (0.5 points)
            vol_ma = df.iloc[i-50:i]['volume'].mean()
            if df.iloc[i]['volume'] > vol_ma:
                score += 0.5

            # Generate signal if score exceeds threshold
            if score >= threshold:
                signals.append((df.index[i], df.iloc[i]['close']))

        return signals

    def _calculate_returns(self, df, signals, holding_period=21):
        """
        Calculate returns for given signals.

        Args:
            df: DataFrame with OHLCV data
            signals: List of (date, price) tuples
            holding_period: Days to hold position (default: 21 = 1 month)

        Returns:
            float: Total return percentage
        """
        total_return = 0

        for signal_date, entry_price in signals:
            try:
                # Find exit date (holding_period days later)
                signal_idx = df.index.get_loc(signal_date)
                exit_idx = min(signal_idx + holding_period, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']

                # Calculate return
                trade_return = (exit_price - entry_price) / entry_price
                total_return += trade_return

            except Exception as e:
                continue

        return total_return

    def test_strategy(self, test_data, threshold):
        """
        Test strategy on out-of-sample test data.

        Args:
            test_data: dict of {ticker: DataFrame} for test period
            threshold: TRENDING_THRESHOLD to use

        Returns:
            dict: Test results with metrics
        """
        trades = []

        for ticker, df in test_data.items():
            try:
                # Generate signals
                signals = self._calculate_signals(df, threshold)

                # Execute trades
                for signal_date, entry_price in signals:
                    signal_idx = df.index.get_loc(signal_date)
                    exit_idx = min(signal_idx + 21, len(df) - 1)
                    exit_date = df.index[exit_idx]
                    exit_price = df.iloc[exit_idx]['close']

                    trade_return = (exit_price - entry_price) / entry_price

                    trades.append({
                        'ticker': ticker,
                        'entry_date': signal_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'outcome': 'win' if trade_return > 0 else 'loss'
                    })

            except Exception as e:
                logger.warning(f"Error testing {ticker}: {e}")
                continue

        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        trades_df = pd.DataFrame(trades)

        results = {
            'total_trades': len(trades),
            'win_rate': len(trades_df[trades_df['outcome'] == 'win']) / len(trades),
            'avg_return': trades_df['return'].mean(),
            'total_return': trades_df['return'].sum(),
            'best_trade': trades_df['return'].max(),
            'worst_trade': trades_df['return'].min(),
            'trades': trades_df
        }

        return results

    def run_walkforward_backtest(self, historical_data, train_window=252, test_window=63):
        """
        Run walk-forward backtesting.

        Args:
            historical_data: dict of {ticker: DataFrame}
            train_window: Training period in days (default: 252 = 1 year)
            test_window: Test period in days (default: 63 = 3 months)

        Returns:
            dict: Backtest results
        """
        logger.info("Starting walk-forward backtesting...")
        logger.info(f"Train window: {train_window} days, Test window: {test_window} days")

        # Find common date range across all tickers
        all_dates = []
        for df in historical_data.values():
            all_dates.extend(df.index.tolist())

        min_date = min(all_dates)
        max_date = max(all_dates)

        logger.info(f"Data range: {min_date.date()} to {max_date.date()}")

        # Create walk-forward windows
        windows = []
        current_start = min_date

        while current_start + timedelta(days=train_window + test_window) <= max_date:
            train_end = current_start + timedelta(days=train_window)
            test_end = train_end + timedelta(days=test_window)

            windows.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })

            # Move forward by test_window
            current_start = train_end

        logger.info(f"Created {len(windows)} walk-forward windows")

        # Run backtest for each window
        all_results = []

        for i, window in enumerate(windows, 1):
            logger.info(f"\n=== Window {i}/{len(windows)} ===")
            logger.info(f"Train: {window['train_start'].date()} to {window['train_end'].date()}")
            logger.info(f"Test:  {window['test_start'].date()} to {window['test_end'].date()}")

            # Split data into train/test
            train_data = {}
            test_data = {}

            for ticker, df in historical_data.items():
                train_df = df[(df.index >= window['train_start']) & (df.index < window['train_end'])]
                test_df = df[(df.index >= window['test_start']) & (df.index < window['test_end'])]

                if len(train_df) > 0:
                    train_data[ticker] = train_df
                if len(test_df) > 0:
                    test_data[ticker] = test_df

            # Optimize on training data
            best_threshold = self.optimize_threshold(train_data)

            # Test on out-of-sample data
            test_results = self.test_strategy(test_data, best_threshold)

            logger.info(f"Test Results: {test_results['total_trades']} trades, "
                       f"Win Rate: {test_results['win_rate']:.1%}, "
                       f"Avg Return: {test_results['avg_return']:.2%}")

            all_results.append({
                'window': i,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'best_threshold': best_threshold,
                **test_results
            })

        # Aggregate results
        summary = self._aggregate_results(all_results)

        return {
            'windows': all_results,
            'summary': summary
        }

    def _aggregate_results(self, all_results):
        """Aggregate results across all windows."""

        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['total_trades'] * r['win_rate'] for r in all_results)
        total_return = sum(r['total_return'] for r in all_results)

        summary = {
            'total_windows': len(all_results),
            'total_trades': total_trades,
            'overall_win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'total_return': total_return,
            'avg_return_per_trade': total_return / total_trades if total_trades > 0 else 0,
            'best_window': max(all_results, key=lambda x: x['total_return']) if all_results else None,
            'worst_window': min(all_results, key=lambda x: x['total_return']) if all_results else None
        }

        logger.info("\n" + "="*70)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Windows:        {summary['total_windows']}")
        logger.info(f"Total Trades:         {summary['total_trades']}")
        logger.info(f"Overall Win Rate:     {summary['overall_win_rate']:.1%}")
        logger.info(f"Total Return:         {summary['total_return']:.2%}")
        logger.info(f"Avg Return/Trade:     {summary['avg_return_per_trade']:.2%}")
        logger.info("="*70)

        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stock Trend Analyzer Walk-Forward Backtest')
    parser.add_argument('--tickers', help='Comma-separated list of tickers')
    parser.add_argument('--file', help='File with tickers (one per line)')
    parser.add_argument('--years', type=int, default=3, help='Years of historical data (default: 3)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--api-key', help='Polygon.io API key (or set MASSIVE_API_KEY env var)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('MASSIVE_API_KEY')
    if not api_key:
        logger.error("API key required. Use --api-key or set MASSIVE_API_KEY environment variable")
        sys.exit(1)

    # Get tickers
    tickers = []
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    elif args.file:
        with open(args.file, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    else:
        logger.error("Must provide --tickers or --file")
        sys.exit(1)

    logger.info(f"Starting backtest for {len(tickers)} tickers: {', '.join(tickers)}")

    # Run backtest
    backtester = StockTrendBacktester(api_key, initial_capital=args.capital)

    # Fetch data
    historical_data = backtester.fetch_historical_data(tickers, years=args.years)

    if len(historical_data) == 0:
        logger.error("No historical data available. Exiting.")
        sys.exit(1)

    # Run walk-forward backtest
    results = backtester.run_walkforward_backtest(historical_data)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'backtest_results'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'stock_trend_backtest_{timestamp}.csv')

    # Save detailed results
    all_trades = []
    for window_result in results['windows']:
        if 'trades' in window_result and window_result['trades'] is not None:
            trades_df = window_result['trades'].copy()
            trades_df['window'] = window_result['window']
            all_trades.append(trades_df)

    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        all_trades_df.to_csv(output_file, index=False)
        logger.info(f"\nDetailed results saved to: {output_file}")

    logger.info("\nBacktest complete!")


if __name__ == '__main__':
    main()
