#!/usr/bin/env python3
"""
Complete script to finalize the migration from Alpha Vantage to Massive.com
"""

import re

def main():
    with open('stock_trend_analyzer.py', 'r') as f:
        content = f.read()

    # Replace get_intraday_data method
    old_intraday_pattern = r'''    def get_intraday_data\(self, ticker: str, interval: str = '5min',
                          outputsize: str = 'full'\) -> Optional\[pd\.DataFrame\]:
        """
        Get intraday time series data
        interval: '1min', '5min', '15min', '30min', '60min'
        outputsize: 'compact' \(latest 100 data points\) or 'full' \(trailing 30 days\)
        """
        logger\.info\(f"Fetching intraday data for \{ticker\} \(interval: \{interval\}, outputsize: \{outputsize\}\)"\)
        self\.rate_limiter\.wait_if_needed\(\)

        params = \{
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self\.api_key,
            'extended_hours': 'true',  # Include pre-market and post-market data
            'entitlement': 'realtime'  # Access real-time data
        \}

        try:
            logger\.debug\(f"Making API request to Alpha Vantage for \{ticker\}"\)
            start_time = time\.time\(\)
            response = requests\.get\(self\.base_url, params=params, timeout=10\)
            elapsed_time = time\.time\(\) - start_time
            logger\.debug\(f"API response received in \{elapsed_time:.2f\}s \(status: \{response\.status_code\}\)"\)

            data = response\.json\(\)

            # Check for API errors
            if 'Error Message' in data:
                logger\.error\(f"API error for \{ticker\}: \{data\['Error Message'\]\}"\)
                logger\.debug\(f"Full API response: \{data\}"\)
                return None

            if 'Note' in data:
                logger\.warning\(f"API note for \{ticker\}: \{data\['Note'\]\}"\)
                return None

            # Check for Information message \(usually indicates invalid ticker or API issue\)
            if 'Information' in data:
                logger\.error\(f"API Information for \{ticker\}: \{data\['Information'\]\}"\)
                logger\.debug\(f"Possible issue: Invalid ticker or API limit reached"\)
                return None

            time_series_key = f'Time Series \(\{interval\}\)'
            if time_series_key not in data:
                logger\.warning\(f"No intraday time series data found for \{ticker\}"\)
                logger\.debug\(f"API response keys: \{list\(data\.keys\(\)\)\}"\)
                # Log full response for debugging if it's small enough
                if len\(str\(data\)\) < 500:
                    logger\.debug\(f"Full API response: \{data\}"\)
                return None

            # Convert to DataFrame
            df = pd\.DataFrame\.from_dict\(data\[time_series_key\], orient='index'\)
            df\.index = pd\.to_datetime\(df\.index\)
            df = df\.sort_index\(\)

            # Rename columns
            df\.columns = \['Open', 'High', 'Low', 'Close', 'Volume'\]

            # Convert to numeric
            for col in df\.columns:
                df\[col\] = pd\.to_numeric\(df\[col\]\)

            logger\.info\(f"Successfully fetched \{len\(df\)\} \{interval\} data points for \{ticker\}"\)
            logger\.debug\(f"Date range: \{df\.index\[0\]\} to \{df\.index\[-1\]\}"\)

            return df

        except requests\.exceptions\.Timeout:
            logger\.error\(f"Timeout fetching intraday data for \{ticker\}"\)
            return None
        except requests\.exceptions\.RequestException as e:
            logger\.error\(f"Request error fetching intraday data for \{ticker\}: \{e\}"\)
            return None
        except Exception as e:
            logger\.error\(f"Unexpected error fetching intraday data for \{ticker\}: \{e\}", exc_info=True\)
            return None'''

    new_intraday = '''    def get_intraday_data(self, ticker: str, interval: str = '5min',
                          outputsize: str = 'full') -> Optional[pd.DataFrame]:
        """
        Get intraday time series data using Massive.com REST API
        interval: '1min', '5min', '15min', '30min', '60min'
        outputsize: 'compact' (latest 100 data points) or 'full' (trailing 30 days)
        """
        logger.info(f"Fetching intraday data for {ticker} (interval: {interval}, outputsize: {outputsize})")
        self.rate_limiter.wait_if_needed()

        try:
            # Map interval to Massive.com timespan and multiplier
            interval_map = {
                '1min': (1, 'minute'),
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
            if outputsize == 'compact':
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

            logger.info(f"Successfully fetched {len(df)} {interval} data points for {ticker}")
            logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            logger.error(f"Unexpected error fetching intraday data for {ticker}: {e}", exc_info=True)
            return None'''

    # Try regex replacement
    content = re.sub(old_intraday_pattern, new_intraday, content, flags=re.DOTALL)

    # Simple string replacements for other references
    replacements = [
        ('AlphaVantageClient(api_key, self.rate_limiter)', 'MassiveClient(api_key, self.rate_limiter)'),
        ('api_key: Alpha Vantage API key', 'api_key: Massive.com API key'),
        ('period: Historical period for daily (\'compact\'=100 days, \'full\'=20+ years)',
         'period: Historical period for daily (\'compact\'=100 days, \'full\'=5 years)'),
        ('# Map period to outputsize for Alpha Vantage', '# Map period to outputsize for Massive.com'),
        ('"""Get the AlphaVantageClient instance for external use"""', '"""Get the MassiveClient instance for external use"""'),
        ('Using interval: {self.interval}, Alpha Vantage API', 'Using interval: {self.interval}, Massive.com API'),
        ('Stock Trend Analyzer - Alpha Vantage Edition', 'Stock Trend Analyzer - Massive.com Edition'),
        ('Alpha Vantage API key (or set ALPHAVANTAGE_API_KEY env variable)', 'Massive.com API key (or set MASSIVE_API_KEY env variable)'),
        ('Historical period: compact (100 days/points) or full (20+ years/30 days)',
         'Historical period: compact (100 days/points) or full (5 years/30 days)'),
        ('ALPHAVANTAGE_API_KEY', 'MASSIVE_API_KEY'),
        ('Error: Alpha Vantage API key is required.', 'Error: Massive.com API key is required.'),
        ('export ALPHAVANTAGE_API_KEY=your_key_here', 'export MASSIVE_API_KEY=your_key_here'),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # Write back
    with open('stock_trend_analyzer.py', 'w') as f:
        f.write(content)

    print("✓ Migration completed successfully!")
    print("  - Replaced all AlphaVantageClient references with MassiveClient")
    print("  - Updated all API key environment variables")
    print("  - Updated all documentation strings")

if __name__ == '__main__':
    main()
