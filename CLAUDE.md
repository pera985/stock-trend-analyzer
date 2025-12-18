# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock Trend Analyzer is a technical analysis tool that identifies stocks trending upward using Massive.com (Polygon.io) API. It analyzes multiple technical indicators (moving averages, momentum, RSI, MACD, ADX, volume) and generates comprehensive visualizations with support for 1-second granularity intraday data.

## Core Architecture

### Main Components

1. **config.py** - Centralized configuration module
   - Contains 130+ configurable parameters organized into 11 sections
   - Scoring system, technical indicators, chart styling, colors, API limits
   - `DEFAULT_RATE_LIMIT = 999999` (unlimited API plan)
   - Configuration validation on import
   - All scripts import from this module for consistent behavior

2. **stock_trend_analyzer.py** - Main analysis engine with three key classes:
   - `RateLimiter`: Manages API rate limiting (defaults to `config.DEFAULT_RATE_LIMIT`)
   - `MassiveClient`: Handles Massive.com/Polygon.io API calls for daily and intraday data
     - Supports 1-second interval data with automatic aggregation to 15s or 30s bars
     - Includes earnings calendar integration
     - Timezone handling (ET to CT conversion for intraday data)
     - `get_latest_price()`: Fetches real-time price including pre-market/after-hours
   - `StockTrendAnalyzer`: Core analysis engine that:
     - Calculates technical indicators (SMA, RSI, MACD, ADX)
     - Scores stocks on 6-point scale (threshold: 4.0 for trending)
     - Returns both trending stocks (≥4.0) and all analyzed stocks
     - Uses daily data for momentum calculations even during intraday analysis
     - **Uses real-time pricing (including extended hours) for current price and momentum**
     - Defaults to `config.DEFAULT_RATE_LIMIT` for rate limiting

3. **plot_results.py** - Dashboard visualization module
   - Creates comprehensive dashboard with score comparison, distribution, top performers, statistics
   - Limits main chart to top 20 stocks for clarity
   - Uses all analyzed stocks for statistics and distribution

4. **plot_individual_tickers.py** - Individual technical charts
   - Generates detailed charts with Bollinger Bands, volume analysis, color-coded RSI
   - Supports date filtering via `start_date` parameter
   - Intraday charts limited to last 2 days with market hour markers (pre-market, open, close)
   - Earnings events marked with purple 'E' markers
   - Timezone converted to Central Time (CT) for display
   - Weekend data automatically filtered out
   - Uses sequential x-axis positions for smooth continuous lines (eliminates weekend gaps)

5. **example_usage.py** - Programmatic usage examples
   - Imports `config` module for default settings
   - Demonstrates usage without hardcoded rate limits

6. **live_dashboard.py** - Continuous loop mode with real-time dashboard (when using `--loop` flag)

7. **streamlit_dashboard.py** - Web-based dashboard for Streamlit Cloud deployment
   - Interactive Plotly charts (pan, zoom, hover tooltips)
   - Sidebar with settings (input file, tickers, interval, auto-refresh)
   - Heatmap showing all stocks ranked by score with daily gain percentage
   - Tabbed interface for top N stock charts
   - Summary metrics (trending count, avg score, avg gain)
   - Full data table with all analyzed stocks
   - Auto-refresh capability with configurable interval

### Scoring System

Stocks are scored out of 6.0 points based on:
- MA Bullish (1.5): Price > SMA50 and SMA50 > SMA200
- Momentum Positive (1.5): All momentum periods (1d, 5d, 10d, 30d) positive
- RSI Favorable (1.0): RSI between 50-70
- MACD Bullish (1.0): MACD histogram positive
- ADX Strong (0.5): ADX > 25
- Volume Increasing (0.5): Recent volume > previous volume

**Threshold: 4.0 or higher = trending stock**

### Data Flow

1. Input: Tickers from file in `input_files/` directory (default: `sample_ticker.txt`) or command-line `--tickers`
2. Analysis: `scan_stocks()` returns tuple of `(trending_df, all_results_df)`
   - `trending_df`: Only stocks with score ≥ 4.0
   - `all_results_df`: ALL analyzed stocks (for complete comparison)
3. Output: CSV files (top 20 only), dashboard plot, individual technical charts (all automatic)

### Output Structure

The tool automatically creates organized directories under `output/`:
```
output/                                 # Main output directory
  csv/                                  # CSV output files
    trending/                           # Top 20 trending stocks only (score ≥ 4.0)
      trending_stocks_TIMESTAMP_INTERVAL_PREFIX.csv
    not_trending/                       # Non-trending stocks only (score < 4.0)
      not_trending_stocks_TIMESTAMP_INTERVAL_PREFIX.csv
  plots/                                # Dashboard visualizations
    stock_analysis_plot_TIMESTAMP_INTERVAL_PREFIX.png
  trending_charts/                      # Individual technical charts and live dashboard
    Live_Dashboard_TIMESTAMP_INTERVAL_PREFIX.png  # Live dashboard (--loop mode)
    PREFIX_INTERVAL_TIMESTAMP/          # Individual charts (non-loop mode)
      uptrending/                       # Stocks with score ≥ 4.0
        01. AAPL_technical_analysis.png
        02. MSFT_technical_analysis.png
        ...
      not_trending/                     # Stocks with score < 4.0
        01. XYZ_technical_analysis.png
        02. ABC_technical_analysis.png
        ...
  logs/                                 # Log files
    stock_analyzer_TIMESTAMP_PREFIX.log
```

**CSV File Differences:**
- **csv/trending/**: Contains only top 20 trending stocks (score ≥ 4.0), standard columns
- **csv/not_trending/**: Contains only non-trending stocks (score < 4.0) with detailed scoring breakdown columns:
  - Individual score components: `ma_bullish_score`, `momentum_score`, `rsi_score`, `macd_score`, `adx_score`, `volume_score`
  - Helps understand why stocks didn't meet the trending threshold
  - Useful for identifying near-miss stocks (e.g., score 3.5)

## Common Commands

### Basic Usage

```bash
# Analyze stocks from default file (sample_ticker.txt in input_files/)
python3 stock_trend_analyzer.py

# Analyze specific tickers
python3 stock_trend_analyzer.py --tickers AAPL,MSFT,GOOGL

# Load from custom file (in input_files/ directory)
python3 stock_trend_analyzer.py --file my_watchlist.txt

# Intraday 5-minute analysis
python3 stock_trend_analyzer.py --interval 5min

# 1-second analysis (aggregated to 30-second bars)
python3 stock_trend_analyzer.py --interval 1sec --aggregate 30

# Custom output filename
python3 stock_trend_analyzer.py --output my_analysis.csv

# Control chart generation (default: top 20)
python3 stock_trend_analyzer.py --top-n 20  # Top 20 charts
python3 stock_trend_analyzer.py --top-n 0   # All trending stocks

# Date filtering for individual charts
python3 stock_trend_analyzer.py --start-date 2025-01-01

# Debug logging
python3 stock_trend_analyzer.py --log-level DEBUG

# Scheduled execution (US/Central timezone)
python3 stock_trend_analyzer.py --start-at 06:00 --file my_watchlist.txt

# Loop mode with live dashboard
python3 stock_trend_analyzer.py --interval 5min --loop

# Streamlit web dashboard (local)
streamlit run streamlit_dashboard.py
```ER_20251215_AMC

### API Configuration

```bash
# Set API key via environment variable (recommended)
export MASSIVE_API_KEY=your_key_here

# Or pass directly
python3 stock_trend_analyzer.py --api-key YOUR_KEY

# Adjust rate limiting (optional - defaults to config.DEFAULT_RATE_LIMIT)
python3 stock_trend_analyzer.py --rate-limit 150  # Override default if needed
```

**Note:** The default rate limit is configured in `config.py` as `DEFAULT_RATE_LIMIT = 999999` (unlimited plan). All scripts automatically use this value unless explicitly overridden.

### Intervals and Periods

- **Intervals**: `1d` (daily), `1sec`, `1min`, `5min`, `15min`, `30min`, `60min`
  - `1sec` interval is automatically aggregated to 15s or 30s bars for visualization
  - `1sec` interval automatically limited to current trading day only
- **Periods**: `compact` (100 days/points) or `full` (5 years for daily, 30 days for intraday)
- **Aggregation**: For `1sec` interval, use `--aggregate 15` or `--aggregate 30` (default: 30)

### Programmatic Usage

See `example_usage.py` for integration examples. Key pattern:

```python
from stock_trend_analyzer import StockTrendAnalyzer
import config  # Import config for access to defaults

# Basic usage - uses config.DEFAULT_RATE_LIMIT automatically
analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full'
    # max_requests_per_minute uses config.DEFAULT_RATE_LIMIT by default
)

# Or override the rate limit if needed
analyzer = StockTrendAnalyzer(
    api_key=API_KEY,
    interval='1d',
    period='full',
    max_requests_per_minute=150,  # Override default
    aggregate_seconds=30  # For 1sec interval
)

# Returns tuple: (trending_df, all_results_df)
trending_df, all_results_df = analyzer.scan_stocks(tickers)

# Use all_results_df for complete comparisons
# Use trending_df for only passing stocks
```

## Important Implementation Details

### Configuration System
- All configurable parameters centralized in `config.py` (130+ parameters)
- Organized into 11 sections: Scoring, Technical Indicators, Data Limits, Time/Market Hours, Chart Styling (Dimensions, Fonts, Lines, Colors, Axis), Display, File Structure
- Configuration validation runs on import to ensure parameter integrity
- All scripts import `config` module for consistent behavior
- Modify `config.py` to customize behavior across the entire application

### Rate Limiting
- `RateLimiter` tracks requests in 60-second rolling window
- Automatically waits when approaching limit to prevent API errors
- Default rate limit: `config.DEFAULT_RATE_LIMIT = 999999` (unlimited plan)
- Can be overridden per-instance if needed via `max_requests_per_minute` parameter
- Both `RateLimiter` and `StockTrendAnalyzer` default to `config.DEFAULT_RATE_LIMIT` when `max_requests_per_minute=None`

### Data Handling
- `scan_stocks()` returns **two DataFrames** (trending and all results)
- Dashboard plots limited to top 20 stocks on main chart, but use all stocks for statistics
- Individual charts generated for both trending (score ≥ 4.0) and non-trending stocks (< 4.0)
- Default limit: top 20 trending stocks for uptrending charts (use `--top-n` to control)
- CSV outputs:
  - `csv/trending/`: Top 20 trending stocks only
  - `csv/not_trending/`: Non-trending stocks only (score < 4.0) with detailed scoring breakdown (ma_bullish_score, momentum_score, rsi_score, macd_score, adx_score, volume_score)
- For intraday analysis, daily data is fetched separately for momentum calculations

### Extended Hours Pricing
- **Real-time price includes pre-market and after-hours trading** (4 AM - 8 PM ET)
- `get_latest_price()` fetches 1-minute bars from current day to get most recent price
- Current price used for:
  - MA comparisons (price vs SMA50/SMA200)
  - 1-day momentum calculation (current price vs previous day's close)
  - All momentum periods (5d, 10d, 30d) use real-time price
- Falls back to last daily close if no intraday data available
- Heatmap percentage in loop mode reflects extended hours price movement

### Intraday Data Features
- **1-second interval**: Automatically aggregated to 15s or 30s bars for better visualization
- **Timezone handling**: Data converted from Eastern Time (ET) to Central Time (CT) for display
- **Weekend filtering**: Automatically removes weekend data points
- **Market hours**: Intraday charts show only extended hours (3:00 AM - 7:00 PM CT)
- **Market markers**:
  - Pre-market start (3:00 AM CT) - green line
  - Market open (8:30 AM CT) - green line
  - Market close (3:00 PM CT) - red line
- **Earnings markers**: Purple 'E' markers show earnings report dates on charts

### Date Filtering
- Individual charts support `start_date` parameter (YYYY-MM-DD format)
- For intraday intervals, charts automatically limited to last 2 days
- Dashboard always shows full historical scores

### Logging
- Comprehensive logging system with configurable levels
- Automatic log file generation in `logs/` directory
- Filename includes timestamp and input file prefix
- Use DEBUG level for troubleshooting API/rate limiting issues

### File Naming
- Input file prefix automatically extracted and used in output filenames
- Special handling for `sample_ticker.txt` → `Sample_Ticker`
- Display interval in filenames: `1sec` becomes `15sec` or `30sec` based on aggregation
- Organized subdirectories for individual charts with timestamp

### Scheduled Execution
- `--start-at HH:MM` parameter schedules execution for specific time (24-hour format)
- Timezone: US/Central
- Can be combined with `--loop` for automated daily runs

### Loop Mode
- `--loop` flag enables continuous live dashboard
- No file outputs generated in loop mode
- Shows top 6 tickers with last 2 days of data
- Updates immediately after each scan cycle
- **Latest price displayed** at top right of each chart with timestamp (CT timezone)
- All times displayed in Central Time (CT)

### Streamlit Web Dashboard

The `streamlit_dashboard.py` provides a shareable web interface as an alternative to the matplotlib-based `live_dashboard.py`.

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_dashboard.py

# Opens at http://localhost:8501
```

**Streamlit Cloud Deployment:**
1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `streamlit_dashboard.py`
5. Add secret in Streamlit Cloud settings:
   - Key: `MASSIVE_API_KEY`
   - Value: Your Polygon.io API key

**Features:**
- Interactive Plotly charts with pan, zoom, and hover tooltips
- Sidebar settings: input file selection, manual ticker input, interval, auto-refresh
- Heatmap showing all stocks ranked by score and daily gain
- Tabbed interface for detailed stock charts (configurable number)
- Summary metrics: trending count, average score, average daily gain
- Full data table with export capability
- Auto-refresh with configurable interval (30-300 seconds)

**Streamlit Cloud Limits (Free Tier):**
- Unlimited public apps
- 1 GB RAM per app
- Apps sleep after 7 days of inactivity (wake on visit)
- No GPU access

## Configuration System Details

### Overview
The `config.py` module centralizes all 130+ configurable parameters into a single source of truth. All scripts import from this module to ensure consistent behavior across the application.

### Configuration Sections
1. **SCORING SYSTEM CONFIGURATION** - Trending threshold (4.0), max score (6.0), component weights
2. **TECHNICAL INDICATORS - PARAMETERS** - Periods for SMA, RSI, MACD, ADX, Bollinger Bands, Volume
3. **DATA VALIDATION & LIMITS** - Minimum data points, API limits, compact/full period definitions
4. **TIME & MARKET HOURS CONFIGURATION** - Timezone offsets, market hours (CT), weekend filtering
5. **CHART STYLING - DIMENSIONS & LAYOUT** - Figure sizes, DPI, panel height ratios, margins
6. **CHART STYLING - FONTS & TEXT** - Font sizes for all chart elements
7. **CHART STYLING - LINES & SHAPES** - Line widths, alpha values, bar widths
8. **CHART STYLING - COLORS** - Comprehensive color scheme for all chart components
9. **CHART STYLING - AXIS & TICKS** - Axis limits, tick configuration
10. **DISPLAY & OUTPUT CONFIGURATION** - Default values, top N charts, progress intervals
11. **FILE STRUCTURE CONFIGURATION** - Directory names, special filename handling

### Key Configuration Values

**Rate Limiting (Unlimited Plan):**
```python
DEFAULT_RATE_LIMIT = 999999  # requests per minute (unlimited plan)
```

**Usage Pattern:**
```python
# In class definitions
def __init__(self, max_requests_per_minute: int = None):
    if max_requests_per_minute is None:
        max_requests_per_minute = config.DEFAULT_RATE_LIMIT
```

**Validation:**
The configuration module includes `validate_config()` which runs on import:
- Validates score weights sum to MAX_SCORE (6.0)
- Checks RSI thresholds are in proper order
- Ensures ADX thresholds are logically ordered
- Raises `ValueError` with detailed error messages if validation fails

### Modifying Configuration

To customize behavior:
1. Edit values in `config.py`
2. Configuration validation runs automatically on import
3. All scripts automatically use new values on next execution
4. No need to modify individual scripts

### Helper Functions

```python
get_score_weights()      # Returns copy of score weights dictionary
get_rsi_colors()         # Returns RSI color mapping
get_dashboard_colors()   # Returns dashboard color mapping
validate_config()        # Validates all configuration values
```

## Technical Notes

- Massive.com/Polygon.io API returns adjusted prices (accounts for splits/dividends)
- **Real-time pricing**: Fetches latest price including pre-market/after-hours for accurate momentum
- Minimum 50 data points required for analysis
- All visualizations automatically generated at 300 DPI
- RSI color coding: Yellow (<30, >80), Red (30-50), Blue (50-80)
- Volume bars colored by price action: green (up day), red (down day)
- Individual charts include earnings calendar integration
- Ticker validation for intraday intervals (warns about preferred stocks, warrants, etc.)
- Sequential x-axis positioning eliminates weekend gaps for smooth continuous chart lines
- All configuration parameters centralized in `config.py` for easy customization

## Final Directory Structure

When --loop is requested (live dashboard mode):
```
output/
  └── trending_charts/
      └── Live_Dashboard_{timestamp}_{interval}_{file_prefix}.png
  └── plots/
      └── stock_analysis_plot_{timestamp}_{interval}_{file_prefix}.png
```

When --loop is not requested (batch analysis mode):
```
output/
  └── trending_charts/
      └── {file_prefix}_{interval}_{timestamp}/
          ├── uptrending/
          │   ├── 01. AAPL_technical_analysis.png
          │   ├── 02. MSFT_technical_analysis.png
          │   └── ... (top N stocks with score ≥ 4.0)
          └── not_trending/
              ├── 01. XYZ_technical_analysis.png
              ├── 02. ABC_technical_analysis.png
              └── ... (all stocks with score < 4.0)
```

### Key Features
✅ Uptrending folder - Contains top N trending stocks (respects --top-n parameter, default 20)
✅ Not trending folder - Contains ALL stocks that didn't meet the 4.0 threshold
✅ Ranked by score - Both folders have numbered charts sorted by score
✅ Complete visibility - You can now review all analyzed stocks, not just the winners
✅ Clear organization - Easy to compare uptrending vs not_trending patterns
✅ Live dashboard - Saves both Live_Dashboard and stock_analysis_plot on exit (Ctrl+C)