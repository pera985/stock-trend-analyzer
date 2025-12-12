"""
Stock Trend Analyzer - Configuration File

This file centralizes all configuration parameters for the stock trend analyzer.
Modify these values to customize scoring, technical indicators, chart styling, and more.
"""

# =============================================================================
# SCORING SYSTEM CONFIGURATION
# =============================================================================

# Trending threshold - stocks with score >= this value are considered "trending"
TRENDING_THRESHOLD = 4.0

# Maximum possible score
MAX_SCORE = 6.0

# Score component weights (must sum to MAX_SCORE)
SCORE_WEIGHTS = {
    'ma_bullish': 1.5,      # Moving average alignment
    'momentum': 1.5,        # Price momentum positive
    'rsi': 1.0,             # RSI in favorable range
    'macd': 1.0,            # MACD bullish
    'adx': 0.5,             # ADX shows strong trend
    'volume': 0.5           # Volume increasing
}

# =============================================================================
# TECHNICAL INDICATORS - PARAMETERS
# =============================================================================

# Moving Averages
SMA_20_PERIOD = 20
SMA_50_PERIOD = 50
SMA_200_PERIOD = 200

# RSI (Relative Strength Index)
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_NEUTRAL_THRESHOLD = 50
RSI_FAVORABLE_MIN = 50
RSI_FAVORABLE_MAX = 70
RSI_OVERBOUGHT_THRESHOLD = 80

# MACD (Moving Average Convergence Divergence)
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# ADX (Average Directional Index)
ADX_PERIOD = 14
ADX_STRONG_THRESHOLD = 25
ADX_VERY_STRONG_THRESHOLD = 40

# Bollinger Bands
BB_PERIOD = 20
BB_NUM_STD = 2

# Volume Analysis
VOLUME_MA_PERIOD = 50
VOLUME_ANALYSIS_MIN = 20      # Minimum data points for volume analysis
VOLUME_RECENT_WINDOW = 10     # Recent volume window size
VOLUME_PREVIOUS_WINDOW = 10   # Previous volume window size
VOLUME_RECENT_PERIODS = -10   # Last N periods for recent volume
VOLUME_PREVIOUS_PERIODS_START = -20
VOLUME_PREVIOUS_PERIODS_END = -10

# Momentum Periods (in days)
MOMENTUM_PERIODS = [1, 5, 10, 30]
MOMENTUM_1D_LOOKBACK_HOURS = 24

# EMA (Exponential Moving Averages) for volume panel
EMA_5_SPAN = 5
EMA_20_SPAN = 20

# =============================================================================
# DATA VALIDATION & LIMITS
# =============================================================================

# Minimum data points required for analysis
MIN_DATA_POINTS = 50
MIN_VOLUME_DATA_POINTS = 20

# Data fetch limits
COMPACT_DAYS_LIMIT = 100
FULL_YEARS_LIMIT = 5
FULL_DAYS_LIMIT = FULL_YEARS_LIMIT * 365

# Intraday data limits
INTRADAY_COMPACT_DAYS = 3
INTRADAY_FULL_DAYS = 30

# API limits
API_LIMIT_PER_REQUEST = 50000
DEFAULT_RATE_LIMIT = 999999  # requests per minute (unlimited plan)
RATE_LIMITER_WINDOW_SECONDS = 60

# Aggregation options for 1-second interval
DEFAULT_AGGREGATE_SECONDS = 30
ALLOWED_AGGREGATIONS = [15, 30]

# Force 1-second interval to current day only
FORCE_1SEC_CURRENT_DAY = True

# =============================================================================
# TIME & MARKET HOURS CONFIGURATION
# =============================================================================

# Timezone offset (ET to CT)
ET_TO_CT_OFFSET_HOURS = 1

# Market hours (in Central Time)
MARKET_HOURS_START = 3   # 3:00 AM CT (pre-market start)
MARKET_HOURS_END = 19    # 7:00 PM CT (post-market end)

PREMARKET_START_HOUR = 3        # 3:00 AM CT
MARKET_OPEN_HOUR = 8            # 8:30 AM CT
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 15          # 3:00 PM CT

# Weekend filtering
WEEKEND_START_DAYOFWEEK = 5  # 5=Saturday, 6=Sunday

# Intraday chart lookback
INTRADAY_CHART_DAYS = 2  # Show last 2 days on intraday charts

# Earnings data timeframe thresholds
EARNINGS_DAILY_THRESHOLD_DAYS = 1
EARNINGS_INTRADAY_THRESHOLD_HOURS = 4

# =============================================================================
# CHART STYLING - DIMENSIONS & LAYOUT
# =============================================================================

# Individual chart dimensions
CHART_FIGSIZE = (16, 14)  # Width, Height in inches
INDIVIDUAL_CHART_DPI = 150

# Panel height ratios (Price, Volume, RSI, MACD, ADX, Summary)
PANEL_HEIGHT_RATIOS = [2.5, 0.7, 0.7, 0.7, 0.7, 1.0]
PANEL_HSPACE = 0.0  # Vertical space between panels

# Dashboard dimensions
DASHBOARD_FIGSIZE = (16, 12)
DASHBOARD_CHART_DPI = 300

# Live dashboard dimensions
LIVE_DASHBOARD_FIGSIZE = (20, 12)

# Plot margins
PLOT_LEFT_MARGIN = 0.08
PLOT_RIGHT_MARGIN = 0.92
PLOT_TOP_MARGIN = 0.97
PLOT_BOTTOM_MARGIN = 0.05

# =============================================================================
# CHART STYLING - FONTS & TEXT
# =============================================================================

# Font sizes
TITLE_FONTSIZE = 16
YLABEL_FONTSIZE = 12
RSI_YLABEL_FONTSIZE = 8
MACD_YLABEL_FONTSIZE = 8
AXIS_LABEL_FONTSIZE = 12
DASHBOARD_TITLE_FONTSIZE = 14
STATS_FONTSIZE = 11

# Legend
RSI_LEGEND_FONTSIZE = 8
RSI_LEGEND_COLS = 2

# Annotations
RSI_ANNOTATION_FONTSIZE = 10

# =============================================================================
# CHART STYLING - LINES & SHAPES
# =============================================================================

# Line widths
PRICE_LINEWIDTH = 1
MA_LINEWIDTH = 1
BB_LINEWIDTH = 1
VOLUME_MA_LINEWIDTH = 1
EMA_LINEWIDTH = 1
RSI_LINEWIDTH = 1
MACD_LINEWIDTH = 1
ADX_LINEWIDTH = 1
BAR_EDGE_LINEWIDTH = 1.5

# Opacity/Alpha values
PRICE_ALPHA = 0.5
MA_ALPHA = 0.7
BB_ALPHA = 0.7
BB_FILL_ALPHA = 0.1
VOLUME_BAR_ALPHA = 0.5
VOLUME_MA_ALPHA = 0.8
EMA_ALPHA = 0.8
GRID_ALPHA = 0.6

# Volume bars
VOLUME_BAR_WIDTH = 0.8

# =============================================================================
# CHART STYLING - COLORS
# =============================================================================

# Dashboard colors (hex codes)
COLOR_PASS = '#4CAF50'       # Green for passing/trending stocks
COLOR_FAIL = '#F44336'       # Red for failing/non-trending stocks
COLOR_THRESHOLD = '#FF9800'  # Orange for threshold line
COLOR_GRID = '#E0E0E0'       # Light gray for grid lines

# RSI colors (hex codes)
RSI_COLOR_OVERSOLD = '#FFD700'   # Yellow (<30)
RSI_COLOR_BEARISH = '#F44336'    # Red (30-50)
RSI_COLOR_BULLISH = '#2196F3'    # Blue (50-80)
RSI_COLOR_OVERBOUGHT = '#FFD700' # Yellow (>80)

# Price chart colors
PRICE_LINE_COLOR = 'black'
MA20_COLOR = 'blue'
MA50_COLOR = 'orange'
MA200_COLOR = 'red'

# Bollinger Bands colors
BB_UPPER_COLOR = 'g'      # Green
BB_MIDDLE_COLOR = 'b'     # Blue
BB_LOWER_COLOR = 'r'      # Red
BB_FILL_COLOR = 'gray'

# Volume colors
VOLUME_UP_COLOR = 'green'
VOLUME_DOWN_COLOR = 'red'
VOLUME_MA_COLOR = 'purple'

# EMA colors (for volume panel)
EMA5_COLOR = 'orange'
EMA20_COLOR = 'blue'

# MACD colors
MACD_LINE_COLOR = 'blue'
MACD_SIGNAL_COLOR = 'red'
MACD_HISTOGRAM_COLOR = 'gray'

# ADX colors
ADX_LINE_COLOR = 'orange'

# Market hour marker colors
PREMARKET_COLOR = 'green'
MARKET_OPEN_COLOR = 'green'
MARKET_CLOSE_COLOR = 'red'

# RSI reference lines
RSI_OVERSOLD_LINE_COLOR = 'green'
RSI_NEUTRAL_LINE_COLOR = 'green'
RSI_OVERBOUGHT_LINE_COLOR = 'gray'

# ADX reference lines
ADX_STRONG_LINE_COLOR = 'green'
ADX_VERY_STRONG_LINE_COLOR = 'red'

# =============================================================================
# CHART STYLING - AXIS & TICKS
# =============================================================================

# Axis limits
AXIS_LIMIT_MAX = 6.5  # Y-axis max for score charts (6.0 max score + 0.5 buffer)

# X-axis ticks
INTRADAY_MAX_TICKS = 12
DAILY_MAX_TICKS = 10

# =============================================================================
# DISPLAY & OUTPUT CONFIGURATION
# =============================================================================

# Default values for command-line arguments
DEFAULT_START_DATE = '2025-01-01'
DEFAULT_TOP_N_CHARTS = 20

# Dashboard display
DASHBOARD_TOP_N_MAIN = 20  # Show top 20 stocks on main dashboard chart

# Live dashboard
LIVE_DASHBOARD_TOP_N = 6   # Show top 6 tickers in live mode

# CSV output
TOP_N_CSV_OUTPUT = 20  # Save top 20 trending stocks to CSV

# Top performers
TOP_PERFORMERS_N = 10  # Show top 10 performers in dashboard

# Progress reporting
PROGRESS_REPORT_INTERVAL_1 = 10
PROGRESS_REPORT_INTERVAL_2 = 50

# =============================================================================
# FILE STRUCTURE CONFIGURATION
# =============================================================================

# Main output directory (parent folder for all outputs)
OUTPUT_DIR_MAIN = 'output'

# Output subdirectory names (under OUTPUT_DIR_MAIN)
OUTPUT_DIR_LOGS = 'logs'
OUTPUT_DIR_CSV = 'csv'
OUTPUT_DIR_PLOTS = 'plots'
OUTPUT_DIR_CHARTS = 'trending_charts'

# Special filename handling
SAMPLE_TICKER_PREFIX = 'sample_ticker'
SAMPLE_TICKER_DISPLAY_NAME = 'Sample_Ticker'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_score_weights():
    """Return the score weights dictionary"""
    return SCORE_WEIGHTS.copy()

def get_rsi_colors():
    """Return RSI color mapping"""
    return {
        'oversold': RSI_COLOR_OVERSOLD,
        'bearish': RSI_COLOR_BEARISH,
        'bullish': RSI_COLOR_BULLISH,
        'overbought': RSI_COLOR_OVERBOUGHT
    }

def get_dashboard_colors():
    """Return dashboard color mapping"""
    return {
        'pass': COLOR_PASS,
        'fail': COLOR_FAIL,
        'threshold': COLOR_THRESHOLD,
        'grid': COLOR_GRID
    }

def validate_config():
    """Validate configuration values"""
    errors = []

    # Check score weights sum to MAX_SCORE
    total_weight = sum(SCORE_WEIGHTS.values())
    if abs(total_weight - MAX_SCORE) > 0.01:
        errors.append(f"Score weights sum to {total_weight}, expected {MAX_SCORE}")

    # Check RSI thresholds are in order
    if not (0 < RSI_OVERSOLD_THRESHOLD < RSI_NEUTRAL_THRESHOLD < RSI_OVERBOUGHT_THRESHOLD <= 100):
        errors.append("RSI thresholds must be in order: 0 < oversold < neutral < overbought <= 100")

    # Check ADX thresholds
    if ADX_STRONG_THRESHOLD >= ADX_VERY_STRONG_THRESHOLD:
        errors.append("ADX strong threshold must be less than very strong threshold")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    return True

# Validate configuration on import
validate_config()
