#!/usr/bin/env python3
"""
Example: Using StockTrendAnalyzer Programmatically - Alpha Vantage Edition

This shows how to use the StockTrendAnalyzer class in your own Python scripts
instead of using the command-line interface.
"""

from stock_trend_analyzer import StockTrendAnalyzer
import config
import os

# Get API key from environment variable
API_KEY = os.environ.get('ALPHAVANTAGE_API_KEY')

if not API_KEY:
    print("Error: Please set ALPHAVANTAGE_API_KEY environment variable")
    print("Example: export ALPHAVANTAGE_API_KEY=your_key_here")
    exit(1)

# Example 1: Basic Usage - Analyze a few stocks with daily data
print("="*60)
print("Example 1: Analyzing tech stocks (Daily data)")
print("="*60)

analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full'
    # max_requests_per_minute uses config.DEFAULT_RATE_LIMIT by default
)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META']
results = analyzer.scan_stocks(tickers)
analyzer.print_results(results, detailed=True)

# Example 2: Intraday 5-minute analysis for day trading
print("\n" + "="*60)
print("Example 2: Intraday 5-minute analysis (Day trading)")
print("="*60)

intraday_analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='5min',
    period='full'  # Gets last 30 days of 5-min data
    # max_requests_per_minute uses config.DEFAULT_RATE_LIMIT by default
)

day_trade_tickers = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA']
intraday_results = intraday_analyzer.scan_stocks(day_trade_tickers)
intraday_analyzer.print_results(intraday_results)

# Example 3: Get detailed analysis for a single stock
print("\n" + "="*60)
print("Example 3: Detailed single stock analysis")
print("="*60)

single_analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full'
)

result = single_analyzer.is_trending_up('AAPL')

if result:
    print(f"\nDetailed Analysis for {result['ticker']}:")
    print(f"  Is Trending: {result['is_trending']}")
    print(f"  Score: {result['score']:.1f}/{result['max_score']}")
    print(f"  Current Price: ${result['current_price']}")
    print(f"\n  Moving Averages:")
    print(f"    - 50-day SMA: ${result['sma_50']}")
    print(f"    - 200-day SMA: ${result['sma_200']}")
    print(f"    - Bullish: {result['ma_bullish']}")
    print(f"\n  Momentum (% change):")
    for period, value in result['momentum'].items():
        print(f"    - {period}: {value}%")
    print(f"    - All Positive: {result['momentum_positive']}")
    print(f"\n  Technical Indicators:")
    print(f"    - RSI: {result['rsi']} (Favorable: {result['rsi_favorable']})")
    print(f"    - MACD Bullish: {result['macd_bullish']}")
    print(f"    - ADX: {result['adx']} (Strong: {result['adx_strong']})")
    print(f"\n  Volume:")
    print(f"    - Trend: {result['volume_trend']}")
    print(f"    - Change: {result['volume_change']}%")

# Example 4: Filter results and work with the DataFrame
print("\n" + "="*60)
print("Example 4: Working with results DataFrame")
print("="*60)

large_cap_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 
                     'META', 'BRK-B', 'UNH', 'JNJ', 'XOM', 'V']

df_analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full'
)

results_df = df_analyzer.scan_stocks(large_cap_tickers, show_progress=False)

if not results_df.empty:
    # Get top 5 by score
    top_5 = results_df.head(5)
    print("\nTop 5 Trending Stocks by Score:")
    print(top_5[['ticker', 'score', 'current_price', 'rsi']].to_string(index=False))
    
    # Filter by specific criteria
    strong_momentum = results_df[results_df['momentum_positive'] == True]
    print(f"\nStocks with positive momentum across all periods: {len(strong_momentum)}")
    
    # Save to file
    results_df.to_csv('my_custom_analysis.csv', index=False)
    print("\nResults saved to my_custom_analysis.csv")

# Example 5: Custom scoring - Find stocks with specific characteristics
print("\n" + "="*60)
print("Example 5: Custom filtering")
print("="*60)

custom_analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full'
)

all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'INTC', 
               'QCOM', 'TXN', 'AVGO', 'ASML', 'TSM']

all_results = custom_analyzer.scan_stocks(all_tickers, show_progress=False)

if not all_results.empty:
    # Filter: Strong trend (ADX > 25) AND bullish MACD AND increasing volume
    custom_filter = all_results[
        (all_results['adx_strong'] == True) & 
        (all_results['macd_bullish'] == True) & 
        (all_results['volume_increasing'] == True)
    ]
    
    print(f"\nStocks meeting custom criteria: {len(custom_filter)}")
    if not custom_filter.empty:
        print(custom_filter[['ticker', 'score', 'adx', 'volume_trend']].to_string(index=False))

# Example 6: 15-minute interval analysis
print("\n" + "="*60)
print("Example 6: 15-minute interval analysis")
print("="*60)

interval_15min = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='15min',
    period='full'
)

swing_tickers = ['SPY', 'QQQ', 'DIA', 'IWM']
interval_results = interval_15min.scan_stocks(swing_tickers, show_progress=False)
interval_15min.print_results(interval_results)

print("\n" + "="*60)
print("Examples complete!")
print("="*60)
print("\nNote: With 150 API requests per minute, you can analyze large")
print("portfolios efficiently without manual rate limiting!")

