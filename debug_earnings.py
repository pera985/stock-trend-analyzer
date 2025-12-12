#!/usr/bin/env python3
"""
Debug script to show earnings fetching in detail
Run this DURING your normal analysis to see what's happening
"""

import os
import sys
import pandas as pd
from io import StringIO
import requests

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_raw_api_call(ticker: str, api_key: str):
    """
    Test the raw API call to see exactly what Alpha Vantage returns

    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key
    """
    print(f"\n{'='*70}")
    print(f"RAW API TEST FOR: {ticker}")
    print(f"{'='*70}\n")

    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'EARNINGS_CALENDAR',
        'symbol': ticker,
        'apikey': api_key
    }

    print(f"Request URL: {url}")
    print(f"Parameters: {params}\n")

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Response Status: {response.status_code}")
        print(f"Response Length: {len(response.text)} characters\n")

        # Show first 500 characters of response
        print("Response Preview (first 500 chars):")
        print("-" * 70)
        print(response.text[:500])
        print("-" * 70)
        print()

        # Try to parse as CSV
        if response.status_code == 200 and response.text:
            # Check for the error pattern
            if response.text.startswith('symbol,name,reportDate,fiscalDateEnding,estimate,currency\nI,n,f,o,r,m'):
                print("❌ DETECTED: Error message disguised as CSV")
                print("   This indicates the API endpoint requires premium access or is unavailable")
                print()
                # Show the full error message
                lines = response.text.split('\n')
                if len(lines) > 1:
                    error_chars = lines[1].split(',')
                    error_message = ''.join(error_chars)
                    print(f"   Error Message: {error_message}")
                return None

            # Try to parse as normal CSV
            try:
                df = pd.read_csv(StringIO(response.text))
                print(f"✓ Successfully parsed as CSV")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Rows: {len(df)}")
                print()

                if not df.empty:
                    print("First few rows:")
                    print(df.head(10).to_string(index=False))
                    print()

                    # Check for valid data
                    if 'symbol' in df.columns:
                        if df['symbol'].iloc[0] == 'I':
                            print("⚠️  First row has symbol='I' - this is likely an error message")
                        else:
                            print(f"✓ Valid data - First symbol: {df['symbol'].iloc[0]}")

                    if 'reportDate' in df.columns:
                        print(f"\nReport Dates:")
                        for idx, row in df.iterrows():
                            print(f"  {row['reportDate']}")

                return df

            except Exception as e:
                print(f"❌ Could not parse as CSV: {e}")
                print()
                return None

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} TICKER")
        print("\nExamples:")
        print(f"  {sys.argv[0]} AAPL")
        print(f"  {sys.argv[0]} EVER")
        print(f"  {sys.argv[0]} MSFT")
        print()
        print("This script will show you EXACTLY what the API returns.")
        print("Make sure ALPHA_VANTAGE_API_KEY is set in your environment.")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # Get API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')

    if not api_key:
        print("\n❌ Error: ALPHA_VANTAGE_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  export ALPHA_VANTAGE_API_KEY='your_actual_key_here'")
        print()
        print("Or run your stock_trend_analyzer.py script which should have the key set")
        sys.exit(1)

    print(f"Using API Key: {api_key[:8]}..." + "*" * 24)

    # Test the API call
    result = test_raw_api_call(ticker, api_key)

    if result is None:
        print("\n" + "="*70)
        print("CONCLUSION:")
        print("="*70)
        print("The EARNINGS_CALENDAR endpoint is NOT returning valid data.")
        print()
        print("Possible reasons:")
        print("  1. Your API plan doesn't include earnings data")
        print("  2. The endpoint is temporarily unavailable")
        print("  3. The ticker doesn't have earnings data")
        print()
        print("Contact Alpha Vantage support to verify your plan includes")
        print("the EARNINGS_CALENDAR endpoint.")
    else:
        print("\n" + "="*70)
        print("CONCLUSION:")
        print("="*70)
        print(f"✓ Successfully retrieved {len(result)} earnings records for {ticker}")
        print()
        print("The earnings feature should work in your charts!")


if __name__ == "__main__":
    main()
