#!/usr/bin/env python3
"""
Test script to fetch and display earnings data for a ticker
This helps debug the earnings marker feature
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add parent directory to path to import stock_trend_analyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_trend_analyzer import AlphaVantageClient, RateLimiter
import config


def test_earnings_fetch(ticker: str):
    """
    Test fetching earnings data for a specific ticker

    Args:
        ticker: Stock ticker symbol to test
    """
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')

    if not api_key:
        print("❌ Error: ALPHA_VANTAGE_API_KEY environment variable not set")
        print("Please set it with: export ALPHA_VANTAGE_API_KEY='your_key_here'")
        return

    print(f"\n{'='*70}")
    print(f"Testing Earnings Data Fetch for: {ticker}")
    print(f"{'='*70}\n")

    # Create rate limiter and client (uses config.DEFAULT_RATE_LIMIT by default)
    rate_limiter = RateLimiter()
    client = AlphaVantageClient(api_key, rate_limiter)

    # Fetch earnings data
    print(f"Fetching earnings calendar data...\n")
    earnings_df = client.get_earnings_calendar(ticker)

    # Display results
    if earnings_df is None:
        print("❌ No earnings data returned")
        print("\nPossible reasons:")
        print("  1. API endpoint requires premium access (you mentioned you have premium)")
        print("  2. Ticker symbol is invalid or has no earnings data")
        print("  3. API error occurred")
        print("\nCheck the logs above for more details.")
        return

    if earnings_df.empty:
        print("⚠️  Earnings data returned but DataFrame is empty")
        return

    # Success - display earnings data
    print(f"✓ Successfully fetched {len(earnings_df)} earnings record(s)\n")
    print("Earnings Data:")
    print("="*70)

    # Display with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    print(earnings_df.to_string(index=False))
    print("="*70)

    # Show date range
    if 'reportDate' in earnings_df.columns:
        earliest = earnings_df['reportDate'].min()
        latest = earnings_df['reportDate'].max()
        print(f"\nDate Range: {earliest} to {latest}")

    # Additional info
    print(f"\nColumns available: {list(earnings_df.columns)}")
    print(f"Data types:\n{earnings_df.dtypes}")

    return earnings_df


def test_with_price_data(ticker: str, interval='1d'):
    """
    Test earnings data matching with actual price data range
    This simulates what happens in plot_individual_tickers.py

    Args:
        ticker: Stock ticker symbol
        interval: Data interval ('1d' for daily)
    """
    # Get API key from environment
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')

    if not api_key:
        print("❌ Error: ALPHA_VANTAGE_API_KEY environment variable not set")
        return

    print(f"\n{'='*70}")
    print(f"Testing Earnings Matching with Price Data for: {ticker}")
    print(f"{'='*70}\n")

    # Create rate limiter and client (uses config.DEFAULT_RATE_LIMIT by default)
    rate_limiter = RateLimiter()
    client = AlphaVantageClient(api_key, rate_limiter)

    # Fetch price data
    print(f"Fetching price data ({interval})...\n")
    if interval == '1d':
        data = client.get_daily_data(ticker, 'full')
    else:
        data = client.get_intraday_data(ticker, interval, 'full')

    if data is None or data.empty:
        print("❌ No price data returned")
        return

    print(f"✓ Price data: {len(data)} points from {data.index.min()} to {data.index.max()}\n")

    # Fetch earnings data
    print(f"Fetching earnings calendar data...\n")
    earnings_df = client.get_earnings_calendar(ticker)

    if earnings_df is None or earnings_df.empty:
        print("❌ No earnings data returned")
        return

    print(f"✓ Earnings data: {len(earnings_df)} total records\n")

    # Filter earnings to only those within price data range
    data_start = data.index.min()
    data_end = data.index.max()

    earnings_in_range = earnings_df[
        (earnings_df['reportDate'] >= data_start) &
        (earnings_df['reportDate'] <= data_end)
    ]

    print(f"Earnings within price data range ({data_start.date()} to {data_end.date()}):")
    print("="*70)

    if earnings_in_range.empty:
        print("⚠️  No earnings reports found within the price data date range")
        print(f"\nAll earnings dates:")
        for idx, row in earnings_df.iterrows():
            print(f"  - {row['reportDate'].date()}")
    else:
        print(f"Found {len(earnings_in_range)} earnings report(s) in range:\n")

        # Find positions in price data for each earnings date
        for idx, earnings_row in earnings_in_range.iterrows():
            earnings_date = earnings_row['reportDate']

            # Find closest data point to earnings date
            time_diffs = abs(data.index - earnings_date)
            closest_idx = time_diffs.argmin()
            closest_date = data.index[closest_idx]
            time_diff = time_diffs.iloc[closest_idx]

            print(f"  Earnings Date: {earnings_date}")
            print(f"  Closest Price Data: {closest_date}")
            print(f"  Time Difference: {time_diff}")
            print(f"  Position in chart: {closest_idx} (out of {len(data)-1})")

            # Check if within tolerance
            if interval == '1d':
                within_tolerance = time_diff <= pd.Timedelta(days=1)
            else:
                within_tolerance = time_diff <= pd.Timedelta(hours=4)

            if within_tolerance:
                print(f"  ✓ WILL BE MARKED ON CHART (within tolerance)")
            else:
                print(f"  ✗ Will NOT be marked (outside tolerance)")
            print()

    print("="*70)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} TICKER [--with-price]")
        print("\nExamples:")
        print(f"  {sys.argv[0]} AAPL")
        print(f"  {sys.argv[0]} MSFT --with-price")
        print(f"  {sys.argv[0]} EVER")
        print("\nOptions:")
        print("  --with-price  Also fetch price data and show matching")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    with_price = '--with-price' in sys.argv

    if with_price:
        test_with_price_data(ticker)
    else:
        test_earnings_fetch(ticker)


if __name__ == "__main__":
    main()
