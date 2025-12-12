# Stock Trend Analyzer

A comprehensive technical analysis tool that identifies stocks trending upward using the Massive.com (Polygon.io) API. The analyzer evaluates multiple technical indicators and generates detailed visualizations with support for both daily and high-frequency intraday data (down to 1-second granularity).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Scoring System](#scoring-system)
- [Technical Indicators](#technical-indicators)
  - [Smoothed Price Curve, Velocity & Acceleration](#smoothed-price-curve-velocity--acceleration)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Command-Line Options](#command-line-options)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Features

### Analysis Capabilities
- ✅ **Multi-timeframe Analysis** - Daily and intraday (1sec, 1min, 5min, 15min, 30min, 60min)
- ⚠️ **Regular Hours Only** - Intraday data currently fetches regular trading hours (9:30 AM - 4:00 PM ET) without pre-market/post-market
- ✅ **High-Frequency Data** - 1-second interval support with automatic aggregation
- ✅ **Comprehensive Scoring** - 6-point scoring system across 6 technical dimensions
- ✅ **Multiple Indicators** - SMA, RSI, MACD, ADX, Bollinger Bands, Volume analysis
- ✅ **Earnings Calendar** - Automatic earnings event markers on charts
- ✅ **Timezone Handling** - Automatic ET to CT conversion for intraday data

### Visualization Features
- 📊 **Dashboard View** - Score comparison, distribution, top performers, statistics
- 📈 **Individual Technical Charts** - 5-panel detailed analysis per stock
  - Price with Bollinger Bands, moving averages, and smoothed price line
  - Volume with dual-axis EMA overlay
  - Color-coded RSI with threshold zones
  - Velocity & Acceleration with quadrant shading for trend analysis
  - Analysis summary panel
- 🎨 **Professional Styling** - High-resolution charts optimized for analysis
- ⏰ **Market Hour Markers** - Pre-market, open, and close indicators on intraday charts
- 🔄 **Live Dashboard Mode** - Continuous real-time monitoring (--loop flag)

### Data Management
- 💾 **Organized Output** - Automatic directory structure for CSV, plots, and charts
- 📁 **Dual CSV Output** - Top 20 trending stocks + complete detailed analysis
- 🏷️ **Smart Naming** - Timestamped files with input source prefix
- 📊 **Tiered Organization** - Separate folders for trending vs non-trending stocks

## Installation

### Prerequisites
- Python 3.9 or higher
- Massive.com (Polygon.io) API key ([Get one here](https://polygon.io/))

### Install Dependencies

```bash
# Clone or download the repository
cd stock_trend_analyzer

# Install required packages
pip3 install pandas numpy matplotlib polygon-api-client
```

### API Key Setup

Choose one of the following methods:

**Option 1: Environment Variable (Recommended)**
```bash
export MASSIVE_API_KEY=your_api_key_here
```

**Option 2: Shell Profile (Permanent)**
```bash
echo 'export MASSIVE_API_KEY=your_api_key_here' >> ~/.zshrc
source ~/.zshrc
```

**Option 3: Command Line**
```bash
python3 stock_trend_analyzer.py --api-key YOUR_API_KEY --tickers AAPL,MSFT
```

## Quick Start

### Basic Analysis

```bash
# Analyze stocks from default file (input_files/sample_ticker.txt)
python3 stock_trend_analyzer.py

# Analyze specific tickers
python3 stock_trend_analyzer.py --tickers AAPL,MSFT,GOOGL,NVDA

# Analyze from custom file
python3 stock_trend_analyzer.py --file my_watchlist.txt
```

### Intraday Analysis

```bash
# 5-minute interval analysis
python3 stock_trend_analyzer.py --interval 5min --tickers AAPL,MSFT

# 1-second interval (aggregated to 30-second bars)
python3 stock_trend_analyzer.py --interval 1sec --aggregate 30

# Use different aggregation (15 seconds)
python3 stock_trend_analyzer.py --interval 1sec --aggregate 15
```

## Usage Examples

### Daily Analysis with Custom Output

```bash
python3 stock_trend_analyzer.py \
    --file tech_stocks.txt \
    --output my_analysis.csv \
    --top-n 20
```

### Intraday with Date Filtering

```bash
python3 stock_trend_analyzer.py \
    --interval 5min \
    --start-date 2025-01-01 \
    --tickers AAPL,MSFT,GOOGL
```

### Scheduled Daily Run

```bash
# Run at 6:00 AM Central Time
python3 stock_trend_analyzer.py \
    --start-at 06:00 \
    --file morning_watchlist.txt
```

### Live Monitoring Dashboard

```bash
# Continuous loop with live dashboard updates
python3 stock_trend_analyzer.py \
    --interval 5min \
    --loop \
    --tickers AAPL,MSFT,GOOGL,NVDA,TSLA,META
```

### Debug Mode

```bash
python3 stock_trend_analyzer.py \
    --log-level DEBUG \
    --tickers AAPL
```

## Scoring System

Stocks are evaluated on a **6-point scale** across six technical dimensions. A score of **4.0 or higher** indicates a trending stock.

### Score Components

| Component | Points | Criteria |
|-----------|--------|----------|
| **MA Bullish** | 1.5 | Price > SMA50 AND SMA50 > SMA200 |
| **Momentum Positive** | 1.5 | All periods (1d, 5d, 10d, 30d) positive |
| **RSI Favorable** | 1.0 | RSI between 50-70 |
| **MACD Bullish** | 1.0 | MACD histogram positive |
| **ADX Strong** | 0.5 | ADX > 25 |
| **Volume Increasing** | 0.5 | Recent volume > previous volume |

### Tier Classification

- **Tier A**: Score ≥ 5.0 (Strong uptrend)
- **Tier B**: Score ≥ 4.0 (Moderate uptrend)
- **Tier C**: Score < 4.0 (Not trending)

## Technical Indicators

### Moving Averages
- **SMA20**: 20-period simple moving average
- **SMA50**: 50-period simple moving average
- **SMA200**: 200-period simple moving average

### Momentum Indicators
- **RSI** (14-period): Relative Strength Index
  - < 30: Oversold (Yellow) - green background shading
  - 30-50: Bearish (Red)
  - 50-70: Bullish (Blue)
  - > 70: Overbought (Yellow) - red background shading

- **MACD** (12, 26, 9): Moving Average Convergence Divergence
  - MACD line
  - Signal line
  - Histogram (bullish when positive)

- **ADX** (14-period): Average Directional Index
  - < 25: Weak trend
  - 25-40: Strong trend
  - > 40: Very strong trend

### Volatility
- **Bollinger Bands** (20-period, 2σ): Price envelope indicator

### Volume Analysis
- **Volume MA50**: 50-period moving average of volume
- **Volume Trend**: Recent vs previous volume comparison

### Smoothed Price Curve, Velocity & Acceleration

The charts include a **smoothed price curve** (solid blue line on the price chart) with corresponding **velocity** and **acceleration** panels, providing early signals for trend changes and inflection points.

#### How the Smooth Curve is Derived

**Step 1: Raw Close Prices**

The smooth curve starts from the raw closing prices - the actual price at which the stock closed for each time interval:

```python
prices = data['Close'].values  # Raw price series: [100.5, 101.2, 100.8, 102.1, ...]
```

**Step 2: Gaussian Smoothing**

The raw price data contains noise - random fluctuations that can obscure the underlying trend. A **Gaussian filter** (`scipy.ndimage.gaussian_filter1d`) smooths this noise by applying a weighted average to each point:

- Nearby points get higher weight
- Distant points get lower weight
- Weights follow a bell-curve (Gaussian) distribution

```python
smoothed = gaussian_filter1d(prices, sigma=3)
```

The **sigma parameter (σ=3)** controls smoothing intensity:
- Lower sigma → less smoothing, closer to raw data
- Higher sigma → more smoothing, smoother curve

**Mathematically**, for each point at position `i`:

```
smoothed[i] = Σ (weight[j] × price[i+j]) / Σ weight[j]
```

Where weights follow: `weight[j] = exp(-j² / (2σ²))`

With σ=3, roughly 6-7 neighboring points on each side have significant influence.

The smoothed price is displayed as a **solid blue line** overlaid on the candlestick price chart.

**Step 3: Velocity (First Derivative)**

**Velocity** represents the **rate of change** of the smoothed price - "how fast is the price moving?"

```python
velocity = np.gradient(smoothed)  # First derivative
```

- **Positive velocity** = price is rising
- **Negative velocity** = price is falling
- **Zero velocity** = price is flat (potential turning point)

**Step 4: Acceleration (Second Derivative)**

**Acceleration** represents the **rate of change of velocity** - "is the movement speeding up or slowing down?"

```python
acceleration = np.gradient(velocity)  # Second derivative
```

- **Positive acceleration** = velocity is increasing (trend strengthening)
- **Negative acceleration** = velocity is decreasing (trend weakening)

#### Quadrant Color Interpretation

The velocity/acceleration panel uses four colors to show market momentum state:

| Velocity | Acceleration | Color | Meaning |
|----------|--------------|-------|---------|
| + | + | Bright Green | Rising & steepening (strong uptrend) |
| + | - | Medium Green | Rising but flattening (uptrend weakening) |
| - | + | Medium Red | Falling but flattening (downtrend weakening) |
| - | - | Bright Red | Falling & steepening (strong downtrend) |

#### Inflection Point Detection

The quadrant colors provide **early signals for trend reversals**:

1. **Bright Green → Medium Green**: Uptrend losing steam, potential top forming
2. **Medium Green → Medium Red**: Trend reversal from up to down
3. **Bright Red → Medium Red**: Downtrend losing steam, potential bottom forming
4. **Medium Red → Medium Green**: Trend reversal from down to up

When medium colors appear, a trend change may be imminent. This gives traders advance warning before the actual price reversal occurs.

#### Vertical Line Indicators

The charts display vertical lines spanning from the velocity panel up to the price chart to highlight key momentum events:

**Solid Vertical Lines** (velocity zero crossings):
- **Blue solid line**: Bullish crossing - velocity crosses from negative to positive (potential buy signal)
- **Orange solid line**: Bearish crossing - velocity crosses from positive to negative (potential sell signal)

**Dashed Vertical Lines** (quadrant changes):
- Drawn when the velocity/acceleration sign combination changes (but not at velocity crossings)
- Color matches the NEW quadrant color (bright green, medium green, medium red, or bright red)
- Indicates acceleration-driven momentum shifts within the same trend direction

These vertical lines make it easy to identify inflection points at a glance across all chart panels.

## Output Files

The analyzer creates an organized directory structure:

```
stock_trend_analyzer/
├── csv/
│   ├── trending/
│   │   └── trending_stocks_TIMESTAMP_INTERVAL_PREFIX.csv  # Top 20 trending
│   └── all/
│       └── all_stocks_detailed_TIMESTAMP_INTERVAL_PREFIX.csv  # All analyzed
├── plots/
│   └── stock_analysis_plot_TIMESTAMP_INTERVAL_PREFIX.png  # Dashboard
├── trending_charts/
│   └── PREFIX_INTERVAL_TIMESTAMP/
│       ├── uptrending/     # Score ≥ 4.0
│       │   ├── 01. AAPL_technical_analysis.png
│       │   ├── 02. MSFT_technical_analysis.png
│       │   └── ...
│       └── non_trending/   # Score < 4.0
│           ├── 01. XYZ_technical_analysis.png
│           └── ...
└── logs/
    └── stock_analyzer_TIMESTAMP_PREFIX.log
```

### CSV File Differences

**trending/**: Top 20 trending stocks with standard columns
**all/**: ALL analyzed stocks with detailed scoring breakdown including:
- Individual score components (`ma_bullish_score`, `momentum_score`, `rsi_score`, `macd_score`, `adx_score`, `volume_score`)
- Helps understand scoring decisions
- Useful for comparing near-miss stocks (score 3.5) vs trending stocks (≥ 4.0)

## Configuration

All parameters are centralized in `config.py` - a comprehensive configuration module with 130+ parameters organized into 11 sections. This provides a single source of truth for all application behavior.

### Configuration Sections

1. **Scoring System** - Trending threshold, max score, component weights
2. **Technical Indicators** - Periods and thresholds for SMA, RSI, MACD, ADX, Bollinger Bands, Volume
3. **Data Validation & Limits** - Minimum data points, API limits, rate limiting
4. **Time & Market Hours** - Timezone settings, market hours, weekend filtering
5. **Chart Styling (Dimensions)** - Figure sizes, DPI, panel ratios, margins
6. **Chart Styling (Fonts)** - Font sizes for titles, labels, legends, annotations
7. **Chart Styling (Lines)** - Line widths, opacity values, bar widths
8. **Chart Styling (Colors)** - Color schemes for all chart elements
9. **Chart Styling (Axis)** - Axis limits, tick configuration
10. **Display & Output** - Default values, top N charts, progress reporting
11. **File Structure** - Directory names, filename handling

### Key Configuration Examples

**Scoring Configuration:**
```python
TRENDING_THRESHOLD = 4.0
SCORE_WEIGHTS = {
    'ma_bullish': 1.5,
    'momentum': 1.5,
    'rsi': 1.0,
    'macd': 1.0,
    'adx': 0.5,
    'volume': 0.5
}
```

**Technical Indicators:**
```python
SMA_20_PERIOD = 20
SMA_50_PERIOD = 50
SMA_200_PERIOD = 200
RSI_PERIOD = 14
RSI_FAVORABLE_MIN = 50
RSI_FAVORABLE_MAX = 70
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
ADX_PERIOD = 14
ADX_STRONG_THRESHOLD = 25
```

**API & Rate Limiting:**
```python
DEFAULT_RATE_LIMIT = 999999  # requests per minute (unlimited plan)
API_LIMIT_PER_REQUEST = 50000
RATE_LIMITER_WINDOW_SECONDS = 60
```

**Chart Styling:**
```python
CHART_FIGSIZE = (22, 16)   # Larger charts for better readability
INDIVIDUAL_CHART_DPI = 150
DASHBOARD_CHART_DPI = 300
PRICE_LINEWIDTH = 1
PRICE_ALPHA = 0.5
```

**Colors:**
```python
COLOR_PASS = '#4CAF50'     # Green
COLOR_FAIL = '#F44336'     # Red
COLOR_THRESHOLD = '#FF9800' # Orange
RSI_COLOR_BULLISH = '#2196F3'  # Blue (50-70)
RSI_COLOR_OVERSOLD = '#FFD700' # Yellow (<30)
RSI_COLOR_OVERBOUGHT = '#FFD700' # Yellow (>70)
```

### Using Configuration

All scripts automatically import and use values from `config.py`:

```python
import config

# Values are used automatically
analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full'
    # max_requests_per_minute uses config.DEFAULT_RATE_LIMIT by default
)
```

### Configuration Validation

The configuration module includes automatic validation:
- Score weights must sum to MAX_SCORE (6.0)
- RSI thresholds must be in proper order
- ADX thresholds must be logically ordered
- Validation runs on import to catch configuration errors early

See [config.py](config.py) for the complete list of all 130+ configurable parameters and their default values.

## Command-Line Options

### Required Options

| Option | Description |
|--------|-------------|
| `--api-key KEY` | Massive.com API key (or set MASSIVE_API_KEY env var) |

### Input Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tickers AAPL,MSFT` | Comma-separated list of tickers | - |
| `--file FILENAME` | Filename in input_files/ directory with tickers (one per line) | `sample_ticker.txt` |

### Analysis Options

| Option | Description | Default |
|--------|-------------|---------|
| `--interval INT` | Time interval: 1d, 1sec, 1min, 5min, 15min, 30min, 60min | `1d` |
| `--period STR` | Data period: compact (100 days) or full (5 years) | `full` |
| `--aggregate N` | Aggregation seconds for 1sec interval (15 or 30) | `30` |

### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output FILE` | Custom CSV output filename | Auto-generated |
| `--top-n N` | Number of trending charts to generate (0=all) | `20` |
| `--start-date YYYY-MM-DD` | Filter individual charts from this date | - |

### Execution Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start-at HH:MM` | Schedule execution for specific time (24hr format, US/Central) | - |
| `--loop` | Enable continuous live dashboard mode | `False` |
| `--rate-limit N` | API rate limit (requests per minute) | `150` |
| `--log-level LEVEL` | Logging level: DEBUG, INFO, WARNING, ERROR | `INFO` |

## Advanced Usage

### Programmatic Usage

```python
from stock_trend_analyzer import StockTrendAnalyzer
import config  # Import config for access to defaults

# Initialize analyzer - uses config.DEFAULT_RATE_LIMIT automatically
analyzer = StockTrendAnalyzer(
    api_key='YOUR_API_KEY',
    interval='1d',
    period='full'
    # max_requests_per_minute uses config.DEFAULT_RATE_LIMIT by default
)

# Or override the rate limit if needed
analyzer_custom = StockTrendAnalyzer(
    api_key='YOUR_API_KEY',
    interval='1d',
    period='full',
    max_requests_per_minute=150  # Override default
)

# Analyze stocks
tickers = ['AAPL', 'MSFT', 'GOOGL']
trending_df, all_results_df = analyzer.scan_stocks(tickers)

# trending_df: Only stocks with score ≥ 4.0
# all_results_df: ALL analyzed stocks (for complete comparison)

print(f"Found {len(trending_df)} trending stocks")
print(trending_df[['ticker', 'score', 'price']])
```

See `example_usage.py` for more integration examples.

### Custom Workflows

**1. Filter by Sector**
```python
# Analyze only tech stocks
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD']
trending_df, all_df = analyzer.scan_stocks(tech_stocks)
```

**2. Score Comparison**
```python
# Compare scores across all analyzed stocks
all_df.sort_values('score', ascending=False)
near_miss = all_df[(all_df['score'] >= 3.5) & (all_df['score'] < 4.0)]
print(f"Near-miss stocks: {len(near_miss)}")
```

**3. Custom Filtering**
```python
# Find high-scoring stocks with specific criteria
high_momentum = trending_df[trending_df['momentum_score'] >= 1.4]
strong_rsi = trending_df[trending_df['rsi_score'] == 1.0]
```

### Batch Processing

**Process multiple watchlists:**
```bash
#!/bin/bash
for watchlist in watchlists/*.txt; do
    python3 stock_trend_analyzer.py --file "$watchlist"
done
```

**Daily automated run:**
```bash
# Add to crontab (runs at 6 AM daily)
0 6 * * 1-5 cd /path/to/stock_trend_analyzer && python3 stock_trend_analyzer.py --file daily_watchlist.txt
```

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
Error: No API key provided
```
**Solution:** Set the MASSIVE_API_KEY environment variable or use --api-key

**2. Rate Limit Exceeded**
```
Warning: Approaching rate limit, waiting...
```
**Solution:** The analyzer automatically manages rate limiting. The default is set to unlimited (`config.DEFAULT_RATE_LIMIT = 999999`). If you need to limit requests, use `--rate-limit N` or modify `config.py`

**3. No Data Returned**
```
Warning: No data available for TICKER
```
**Solution:**
- Check ticker symbol is valid
- For intraday: Use tickers with high volume (avoid preferred stocks, warrants)
- Verify API key has access to requested data tier

**4. Insufficient Data Points**
```
Skipped TICKER: Insufficient data (got 45, need 50)
```
**Solution:** Use `--period full` or check if ticker has trading history

**5. Weekend Data Issues**
```
Warning: No market days available for TICKER
```
**Solution:** Weekend data is automatically filtered. Ensure date range includes trading days.

### Debug Mode

Enable detailed logging:
```bash
python3 stock_trend_analyzer.py --log-level DEBUG --tickers AAPL
```

Check log files in `logs/` directory for detailed execution traces.

### Performance Tips

1. **Rate Limiting**: Unlimited plan configured by default (`config.DEFAULT_RATE_LIMIT = 999999`). Adjust in `config.py` if needed.
2. **Compact Mode**: Use `--period compact` for faster analysis (100 days vs 5 years)
3. **Limit Charts**: Use `--top-n 10` to generate fewer individual charts
4. **Batch Size**: With unlimited API, analyze large portfolios efficiently without manual throttling

## Data Sources & API

### Massive.com (Polygon.io)
- **Provider**: Polygon.io via Massive.com
- **Data Quality**: Adjusted prices (accounts for splits/dividends)
- **Coverage**: US stocks, ETFs
- **Intervals**: 1sec to daily
- **Historical**: Up to 5 years (daily), 30 days (intraday)

### Extended Hours Trading

**Current Status:** ⚠️ Extended hours data is **NOT currently being fetched** for intraday intervals.

The API calls do not include the `include_otc='true'` parameter, which means:
- **Pre-market** (4:00-9:30 AM ET): Not included
- **Regular Hours** (9:30 AM-4:00 PM ET): ✅ Included
- **Post-market** (4:00-8:00 PM ET): Not included

**Note:** While the visualization code and configuration are set up to handle extended hours (market hours configured as 3:00 AM - 7:00 PM CT), the data fetching currently only retrieves regular trading hours. To enable extended hours:

1. Modify the `list_aggs()` call in `stock_trend_analyzer.py` (around line 297)
2. Add parameter: `include_otc=True` for intraday intervals

**Alternative:** Use the `fetch_raw_data` project which includes extended hours data for 5-minute and 15-minute timeframes.

### Rate Limits
- **Free Tier**: 5 requests/minute
- **Premium Tier**: 150 requests/minute
- **Unlimited Tier**: No rate limits (configured as default: `config.DEFAULT_RATE_LIMIT = 999999`)

### Timezone Handling
- **API Returns**: Eastern Time (ET)
- **Charts Display**: Central Time (CT)
- **Automatic Conversion**: ET to CT (-1 hour)

## Project Structure

```
stock_trend_analyzer/
├── config.py                     # Centralized configuration (130+ parameters)
├── stock_trend_analyzer.py      # Main analysis engine
├── plot_results.py               # Dashboard visualization
├── plot_individual_tickers.py    # Individual technical charts
├── live_dashboard.py             # Live monitoring dashboard
├── example_usage.py              # Programmatic usage examples
├── test_earnings.py              # Earnings data testing utility
├── CLAUDE.md                     # AI assistant instructions
├── README.md                     # This file
├── input_files/                  # Input ticker files
│   └── sample_ticker.txt
├── csv/                          # CSV outputs
│   ├── trending/                 # Top 20 trending stocks
│   └── all/                      # All analyzed stocks with details
├── plots/                        # Dashboard plots
├── trending_charts/              # Individual charts
│   └── PREFIX_INTERVAL_TIMESTAMP/
│       ├── uptrending/           # Charts for score ≥ 4.0
│       └── non_trending/         # Charts for score < 4.0
└── logs/                         # Log files
```

## Technical Notes

- **Adjusted Prices**: All data is adjusted for splits and dividends
- **Minimum Data**: Requires 50 data points for analysis
- **Chart Resolution**: Individual charts at 150 DPI, dashboard at 300 DPI
- **Memory Efficient**: Processes stocks one at a time
- **Thread Safe**: Single-threaded execution with rate limiting

## License

This project is for educational and personal use. Ensure compliance with Polygon.io's terms of service.

## Support & Documentation

- **Configuration Reference**: See `config.py` for all parameters
- **AI Assistant Guide**: See `CLAUDE.md` for development guidelines
- **API Documentation**: [Polygon.io Docs](https://polygon.io/docs)

## Version History

- **v2.0** - Migration to Massive.com/Polygon.io API
- **v1.5** - Added 1-second interval support
- **v1.0** - Initial release with Alpha Vantage API

---

**Happy Trading! 📈**

Remember: This tool is for analysis purposes only. Always do your own research and never invest money you can't afford to lose.
