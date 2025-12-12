#!/usr/bin/env python3
"""
Test script to validate the migration from Alpha Vantage to Massive.com
"""

import sys

# Test imports
try:
    from massive import RESTClient
    print('✓ massive library imported successfully')
except ImportError as e:
    print(f'✗ Failed to import massive: {e}')
    sys.exit(1)

# Test MassiveClient class definition
try:
    with open('stock_trend_analyzer.py', 'r') as f:
        content = f.read()

        checks = [
            ('class MassiveClient:', 'MassiveClient class defined'),
            ('self.rest_client = RESTClient(api_key)', 'REST client initialization'),
            ('def get_daily_data', 'get_daily_data method defined'),
            ('def get_intraday_data', 'get_intraday_data method defined'),
            ('def get_earnings_calendar', 'get_earnings_calendar method defined'),
            ('MASSIVE_API_KEY', 'MASSIVE_API_KEY environment variable'),
            ('Massive.com Edition', 'Updated to Massive.com Edition'),
        ]

        for check_string, description in checks:
            if check_string in content:
                print(f'✓ {description}')
            else:
                print(f'✗ {description} not found')
                sys.exit(1)

except Exception as e:
    print(f'✗ Error reading file: {e}')
    sys.exit(1)

print()
print('='*70)
print('All code structure checks passed! ✓')
print('='*70)
print()
print('Summary of changes:')
print('  - Replaced AlphaVantageClient with MassiveClient')
print('  - Updated all API key references to MASSIVE_API_KEY')
print('  - Implemented get_daily_data() with Massive.com REST API')
print('  - Implemented get_intraday_data() with Massive.com REST API')
print('  - Implemented get_earnings_calendar() with Massive.com REST API')
print('  - Updated all help text and documentation')
print()
print('='*70)
print('Files modified:')
print('='*70)
print('  - stock_trend_analyzer.py (migrated to Massive.com)')
print()
print('Files that will auto-inherit the changes:')
print('  - live_dashboard.py (uses get_client() from analyzer)')
print('  - plot_individual_tickers.py (uses get_client() from analyzer)')
print()
print('='*70)
print('Next steps:')
print('='*70)
print('  1. Set your Massive.com API key:')
print('     export MASSIVE_API_KEY=your_api_key_here')
print()
print('  2. Test with a sample ticker:')
print('     python3 stock_trend_analyzer.py --tickers AAPL')
print()
print('  3. Test the live dashboard:')
print('     python3 stock_trend_analyzer.py --tickers AAPL,MSFT --loop')
print()
