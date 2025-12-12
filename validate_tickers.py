#!/usr/bin/env python3
"""
Ticker Validation Utility
Validates ticker symbols in a file and identifies potentially problematic ones
"""

import sys
import argparse


def validate_ticker(ticker: str, check_intraday: bool = False) -> tuple[bool, list]:
    """
    Validate a ticker symbol and return warnings

    Args:
        ticker: Stock ticker symbol
        check_intraday: Whether to check for intraday compatibility

    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []

    # Check for preferred stock suffixes that might not work with intraday
    if ticker.endswith(('-', 'P', 'PR')) or any(c in ticker for c in ['.', '-']):
        if check_intraday:
            warnings.append("May not have intraday data (preferred stock/warrant)")
        else:
            warnings.append("Preferred stock/warrant")

    # Check for unusual length
    if len(ticker) > 5:
        warnings.append(f"Unusually long ({len(ticker)} chars)")

    # Check for lowercase
    if ticker != ticker.upper():
        warnings.append("Contains lowercase")

    # Check for special characters
    if any(char in ticker for char in ['/', '=']):
        warnings.append("Special characters (may be bond/option)")

    # Check for numbers at start (usually invalid)
    if ticker and ticker[0].isdigit():
        warnings.append("Starts with number")

    # Common invalid patterns
    if ticker.endswith('Z') and len(ticker) > 4:
        warnings.append("Ends with Z (may be warrant/rights)")

    return (len(warnings) == 0, warnings)


def main():
    parser = argparse.ArgumentParser(
        description='Validate stock ticker symbols in a file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate tickers in file
  python validate_tickers.py sample_ticker.txt

  # Check for intraday compatibility
  python validate_tickers.py sample_ticker.txt --intraday

  # Save only valid tickers to new file
  python validate_tickers.py sample_ticker.txt --output valid_tickers.txt

  # Show only problematic tickers
  python validate_tickers.py sample_ticker.txt --show-invalid-only
        """
    )

    parser.add_argument('input_file',
                        help='Input file containing tickers (one per line)')
    parser.add_argument('--intraday',
                        action='store_true',
                        help='Check compatibility with intraday data (5min, etc.)')
    parser.add_argument('--output',
                        help='Output file for valid tickers only')
    parser.add_argument('--show-invalid-only',
                        action='store_true',
                        help='Show only problematic tickers')

    args = parser.parse_args()

    # Read tickers
    try:
        with open(args.input_file, 'r') as f:
            tickers = [line.strip().upper() for line in f
                      if line.strip() and not line.strip().startswith('#')]
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return 1
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    if not tickers:
        print("No tickers found in file")
        return 1

    print(f"Validating {len(tickers)} ticker(s) from {args.input_file}")
    if args.intraday:
        print("Checking for intraday data compatibility\n")
    print("=" * 70)

    valid_tickers = []
    invalid_tickers = []

    for ticker in tickers:
        is_valid, warnings = validate_ticker(ticker, args.intraday)

        if is_valid:
            valid_tickers.append(ticker)
            if not args.show_invalid_only:
                print(f"✓ {ticker:10} - Valid")
        else:
            invalid_tickers.append((ticker, warnings))
            print(f"⚠ {ticker:10} - {', '.join(warnings)}")

    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Valid tickers:   {len(valid_tickers)}")
    print(f"  Invalid tickers: {len(invalid_tickers)}")

    if invalid_tickers:
        print(f"\n⚠️  Problematic tickers that may fail:")
        for ticker, warnings in invalid_tickers:
            print(f"  - {ticker}: {', '.join(warnings)}")

    # Save valid tickers if output file specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                for ticker in valid_tickers:
                    f.write(f"{ticker}\n")
            print(f"\n✓ Saved {len(valid_tickers)} valid tickers to {args.output}")
        except Exception as e:
            print(f"\nError saving output file: {e}")
            return 1

    return 0 if not invalid_tickers else 1


if __name__ == "__main__":
    sys.exit(main())
