#!/usr/bin/env python3
"""
Individual Stock Technical Analysis Charts
Creates detailed technical charts for each analyzed ticker
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy.ndimage import gaussian_filter1d

# Color scheme for RSI (same as inflection_detector)
RSI_COLOR_OVERSOLD = '#FFD700'  # Yellow (<30)
RSI_COLOR_BEARISH = '#F44336'   # Red (30-50)
RSI_COLOR_BULLISH = '#2196F3'   # Blue (50-70)
RSI_COLOR_OVERBOUGHT = '#FFD700' # Yellow (>70)

# Velocity/Acceleration colors (same as inflection_detector)
SMOOTHED_COLOR = 'blue'           # Blue for smoothed price line (more visible)
VELOCITY_COLOR = 'blue'           # Blue for velocity line
ACCEL_COLOR = '#C49821'           # Gold/amber color for acceleration

# Quadrant colors for velocity/acceleration shading
COLOR_VEL_POS_ACC_POS = '#00C853'  # Bright green - rising & steepening
COLOR_VEL_POS_ACC_NEG = '#69F0AE'  # Medium green - rising but flattening
COLOR_VEL_NEG_ACC_POS = '#FF8A80'  # Medium red - falling but flattening
COLOR_VEL_NEG_ACC_NEG = '#D50000'  # Bright red - falling & steepening


def calculate_bollinger_bands(data, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data: DataFrame with 'Close' column
        period: Period for moving average (default 20)
        num_std: Number of standard deviations (default 2)
    
    Returns:
        DataFrame with middle, upper, and lower bands
    """
    middle_band = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return pd.DataFrame({
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band
    })


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index
    
    Args:
        data: DataFrame with 'Close' column
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def get_rsi_color(rsi_value):
    """Get color based on RSI value (same thresholds as inflection_detector)"""
    if rsi_value < 30:
        return RSI_COLOR_OVERSOLD
    elif 30 <= rsi_value < 50:
        return RSI_COLOR_BEARISH
    elif 50 <= rsi_value <= 70:
        return RSI_COLOR_BULLISH
    else:  # > 70
        return RSI_COLOR_OVERBOUGHT


def calculate_smoothed_velocity_acceleration(data, sigma=3):
    """
    Calculate smoothed price, velocity (1st derivative), and acceleration (2nd derivative)
    using Gaussian smoothing - same approach as inflection_detector.

    Args:
        data: DataFrame with 'Close' column
        sigma: Gaussian smoothing parameter (default: 3)

    Returns:
        Dict with 'smoothed', 'velocity', 'acceleration' Series
    """
    prices = data['Close'].values.astype(float)

    # Smooth prices using Gaussian filter
    smoothed = gaussian_filter1d(prices, sigma=sigma)

    # Calculate derivatives
    velocity = np.gradient(smoothed)       # First derivative (rate of change)
    acceleration = np.gradient(velocity)   # Second derivative

    return {
        'smoothed': pd.Series(smoothed, index=data.index),
        'velocity': pd.Series(velocity, index=data.index),
        'acceleration': pd.Series(acceleration, index=data.index)
    }


def detect_swing_points(data, window=5):
    """
    Detect swing highs and lows and classify them as HH, HL, LH, LL.

    HH (Higher High): A swing high that is higher than the previous swing high
    LH (Lower High): A swing high that is lower than the previous swing high
    HL (Higher Low): A swing low that is higher than the previous swing low
    LL (Lower Low): A swing low that is lower than the previous swing low

    Args:
        data: DataFrame with 'High' and 'Low' columns
        window: Lookback window for swing detection (bars on each side)

    Returns:
        DataFrame with swing point classifications added
    """
    df = data.copy()
    highs = df['High'].values
    lows = df['Low'].values

    # Initialize columns
    df['swing_high'] = False
    df['swing_low'] = False
    df['swing_label'] = ''
    df['is_major_swing'] = False  # For distinguishing major vs minor

    swing_highs = []  # List of (index, price)
    swing_lows = []   # List of (index, price)

    # Detect swing highs and lows
    for i in range(window, len(df) - window):
        # Swing high: highest point in the window
        if highs[i] == max(highs[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('swing_high')] = True
            swing_highs.append((i, highs[i]))

        # Swing low: lowest point in the window
        if lows[i] == min(lows[i-window:i+window+1]):
            df.iloc[i, df.columns.get_loc('swing_low')] = True
            swing_lows.append((i, lows[i]))

    # Classify swing highs as HH or LH
    prev_swing_high = None
    for idx, price in swing_highs:
        if prev_swing_high is not None:
            if price > prev_swing_high:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'HH'
            else:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'LH'
        prev_swing_high = price

    # Classify swing lows as HL or LL
    prev_swing_low = None
    for idx, price in swing_lows:
        if prev_swing_low is not None:
            if price > prev_swing_low:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'HL'
            else:
                df.iloc[idx, df.columns.get_loc('swing_label')] = 'LL'
        prev_swing_low = price

    # Mark major swings (larger window = more significant)
    major_window = window * 2
    for i in range(major_window, len(df) - major_window):
        if df.iloc[i]['swing_high']:
            if highs[i] == max(highs[i-major_window:i+major_window+1]):
                df.iloc[i, df.columns.get_loc('is_major_swing')] = True
        if df.iloc[i]['swing_low']:
            if lows[i] == min(lows[i-major_window:i+major_window+1]):
                df.iloc[i, df.columns.get_loc('is_major_swing')] = True

    return df


# Swing label colors (same as inflection_detector)
SWING_LABEL_COLORS = {
    'HH': 'darkgreen',   # Higher High - bullish
    'HL': 'green',       # Higher Low - bullish
    'LH': 'darkred',     # Lower High - bearish
    'LL': 'red'          # Lower Low - bearish
}


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        data: DataFrame with 'Close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        DataFrame with MACD line, signal line, and histogram
    """
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    return pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX)

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns
        period: ADX period (default 14)

    Returns:
        Series with ADX values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate smoothed +DI and -DI
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def plot_ticker_technical_analysis(ticker, data, score=None, output_dir='individual_plots', start_date=None, rank=None, interval='1d', client=None):
    """
    Create individual technical analysis chart for a ticker

    Args:
        ticker: Stock ticker symbol
        data: DataFrame with OHLCV data
        score: Optional trend score to display
        output_dir: Directory to save plots
        start_date: Optional start date (YYYY-MM-DD) to filter data from
        rank: Optional rank number for filename prefix
        interval: Data interval ('1d' for daily, or intraday like '5min')
        client: Optional MassiveClient for fetching earnings data

    Returns:
        Path to saved plot file
    """

    # For intraday data, limit to last 2 days to avoid overcrowding
    # Do this BEFORE indicator calculations since intraday doesn't need warmup
    if interval != '1d' and len(data) > 0:
        # Calculate 2 days ago from the most recent data point
        latest_date = data.index[-1]
        two_days_ago = latest_date - pd.Timedelta(days=2)
        data = data[data.index >= two_days_ago]

        if data.empty:
            print(f"  Warning: No data available for {ticker} in the last 2 days")
            return None

    # Filter to market days only (exclude weekends: Saturday=5, Sunday=6)
    # Always apply this filter for both daily and intraday data
    if len(data) > 0:
        # Check for weekend data before filtering
        weekend_count = len(data[data.index.dayofweek >= 5])
        if weekend_count > 0:
            print(f"  Filtering out {weekend_count} weekend data points")

        # Filter out weekends: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
        # Exclude: 5=Saturday, 6=Sunday
        data = data[data.index.dayofweek < 5]

        if data.empty:
            print(f"  Warning: No market days available for {ticker} in filtered date range")
            return None

    # For intraday data, filter to market hours (including extended hours)
    # Massive.com returns data in Eastern Time (ET)
    # US market hours in ET: Pre-market (4:00 AM - 9:30 AM), Regular (9:30 AM - 4:00 PM), Post-market (4:00 PM - 8:00 PM)
    # Converting to Central Time (CT) for display: Pre-market (3:00 AM - 8:30 AM), Regular (8:30 AM - 3:00 PM), Post-market (3:00 PM - 7:00 PM)
    if interval != '1d' and len(data) > 0:
        # Convert from ET to CT (subtract 1 hour)
        data.index = data.index - pd.Timedelta(hours=1)

        # Keep only times between 3:00 AM and 7:00 PM CT (hour < 19 means up to 18:59:59)
        data = data[(data.index.hour >= 3) & (data.index.hour < 19)]

        if data.empty:
            print(f"  Warning: No market hours data available for {ticker} in filtered date range")
            return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate technical indicators on FULL data first (before trimming to display days)
    # This ensures SMA 200 and Bollinger Bands have valid values from the start of display
    bb = calculate_bollinger_bands(data, period=20, num_std=2)
    rsi = calculate_rsi(data, period=14)
    macd_data = calculate_macd(data)
    adx = calculate_adx(data)

    # Calculate moving averages on full data (MA20 removed - redundant with Bollinger Bands SMA20)
    sma_50 = data['Close'].rolling(window=50).mean()
    sma_200 = data['Close'].rolling(window=200).mean()
    volume_ma_50 = data['Volume'].rolling(window=50).mean()

    # Calculate EMAs for volume panel
    ema_5 = data['Close'].ewm(span=5, adjust=False).mean()
    ema_20 = data['Close'].ewm(span=20, adjust=False).mean()

    # Calculate smoothed price, velocity, and acceleration on full data
    derivatives = calculate_smoothed_velocity_acceleration(data, sigma=3)

    # For daily data, trim to display period AFTER calculating indicators
    # This ensures SMA 200 has valid values from the start of display
    if interval == '1d' and len(data) > 0:
        # Determine display start: use start_date if provided, otherwise last 252 days
        if start_date:
            # Find the index position of start_date
            display_mask = data.index >= start_date
            if not display_mask.any():
                print(f"  Warning: No data available for {ticker} from {start_date} onwards")
                return None
        else:
            # Default to last 252 trading days (~1 year)
            display_days = 252
            if len(data) > display_days:
                display_mask = pd.Series([False] * (len(data) - display_days) + [True] * display_days, index=data.index)
            else:
                display_mask = pd.Series([True] * len(data), index=data.index)

        # Apply display mask to data and all indicators
        data = data[display_mask]
        bb = {k: v[display_mask] for k, v in bb.items()}
        rsi = rsi[display_mask]
        macd_data = {k: v[display_mask] for k, v in macd_data.items()}
        adx = adx[display_mask]
        sma_50 = sma_50[display_mask]
        sma_200 = sma_200[display_mask]
        volume_ma_50 = volume_ma_50[display_mask]
        ema_5 = ema_5[display_mask]
        ema_20 = ema_20[display_mask]
        derivatives = {k: v[display_mask] for k, v in derivatives.items()}
    smoothed = derivatives['smoothed']
    velocity = derivatives['velocity']
    acceleration = derivatives['acceleration']

    # Detect swing points for HH/HL/LH/LL labels
    # Use window=5 for intraday, window=5 for daily (same as inflection_detector)
    swing_window = 5
    data = detect_swing_points(data, window=swing_window)

    # Create figure with GridSpec - 5 panels vertical stacked
    # Height ratios: Price (2.5), Volume (0.7), RSI (0.7), Velocity/Acceleration (0.7), Summary (1.0)
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[2.5, 0.7, 0.7, 0.7, 1.0], hspace=0.0)
    
    # Get date range for display
    date_range = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"

    # Fetch earnings dates if client is provided (same approach as inflection_detector)
    earnings_dates = []
    if client:
        try:
            # Use get_earnings_dates method (same as inflection_detector)
            earnings_dates = client.get_earnings_dates(ticker)
        except Exception as e:
            # Silently fail - earnings dates are optional
            pass

    # Find pre-market start times (3:00 AM CT) for intraday charts
    premarket_start_times = []
    market_open_times = []
    market_close_times = []
    if interval != '1d':
        # For intraday data, find all instances where pre-market starts (3:00 AM CT)
        # Mark the first data point at or after 3:00 AM for each trading day (store positions)
        current_date = None
        for idx, timestamp in enumerate(data.index):
            # Check if this is a new trading day
            date_only = timestamp.date()
            if date_only != current_date:
                current_date = date_only
                # Find first data point at or after 3:00 AM (hour >= 3)
                if timestamp.hour >= 3:
                    premarket_start_times.append(idx)
                else:
                    # Look ahead for the 3:00 AM mark on this day
                    for future_idx in range(idx, len(data)):
                        future_timestamp = data.index[future_idx]
                        if future_timestamp.date() != date_only:
                            # Moved to next day without finding 3:00 AM
                            break
                        if future_timestamp.hour >= 3:
                            premarket_start_times.append(future_idx)
                            break

        # Find market open times (8:30 AM CT) - store positions
        current_date = None
        for idx, timestamp in enumerate(data.index):
            date_only = timestamp.date()
            if date_only != current_date:
                current_date = date_only
                # Look for 8:30 AM CT (hour=8, minute>=30) or later
                if timestamp.hour > 8 or (timestamp.hour == 8 and timestamp.minute >= 30):
                    market_open_times.append(idx)
                else:
                    # Look ahead for the 8:30 AM mark on this day
                    for future_idx in range(idx, len(data)):
                        future_timestamp = data.index[future_idx]
                        if future_timestamp.date() != date_only:
                            break
                        if future_timestamp.hour > 8 or (future_timestamp.hour == 8 and future_timestamp.minute >= 30):
                            market_open_times.append(future_idx)
                            break

        # Find market close times (3:00 PM CT = 15:00) - store positions
        current_date = None
        for idx, timestamp in enumerate(data.index):
            date_only = timestamp.date()
            if date_only != current_date:
                current_date = date_only
                # Look for 3:00 PM CT (hour=15, minute=0) or later
                if timestamp.hour >= 15:
                    market_close_times.append(idx)
                else:
                    # Look ahead for the 3:00 PM mark on this day
                    for future_idx in range(idx, len(data)):
                        future_timestamp = data.index[future_idx]
                        if future_timestamp.date() != date_only:
                            break
                        if future_timestamp.hour >= 15:
                            market_close_times.append(future_idx)
                            break

    # Create sequential x-axis positions to eliminate gaps (smooth lines)
    x_positions = np.arange(len(data))

    # ============================================
    # PANEL 1: PRICE WITH BOLLINGER BANDS AND MAs
    # ============================================
    ax1 = fig.add_subplot(gs[0])

    # Draw candlesticks (same setup as inflection_detector)
    candle_width = 0.8
    wick_width = 1.0  # Thicker wick for solid black appearance

    # FIRST PASS: Draw ALL wicks (behind bodies)
    for i in range(len(data)):
        high_price = data['High'].iloc[i]
        low_price = data['Low'].iloc[i]
        ax1.plot([x_positions[i], x_positions[i]], [low_price, high_price],
                 color='black', linewidth=wick_width, alpha=1.0, zorder=2)

    # SECOND PASS: Draw ALL bodies (on top of wicks)
    for i in range(len(data)):
        open_price = data['Open'].iloc[i]
        close_price = data['Close'].iloc[i]

        # Determine color: green for up, red for down
        if close_price >= open_price:
            body_color = '#56B05C'  # Green
        else:
            body_color = '#F77272'  # Light red

        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)

        # If open == close, draw a small horizontal line (doji)
        if body_height == 0:
            ax1.plot([x_positions[i] - candle_width/2, x_positions[i] + candle_width/2],
                     [close_price, close_price], color=body_color, linewidth=1, zorder=3)
        else:
            rect = plt.Rectangle((x_positions[i] - candle_width/2, body_bottom),
                                  candle_width, body_height,
                                  facecolor=body_color, edgecolor=body_color,
                                  linewidth=0.5, alpha=1.0, zorder=3)
            ax1.add_patch(rect)

    # Plot moving averages (using sequential positions)
    # MA20 removed - redundant with Bollinger Bands SMA20
    ax1.plot(x_positions, sma_50, label='MA50', color='orange', linewidth=1, alpha=0.7)
    # SMA 200 in magenta (dashed) - same as inflection_detector
    ax1.plot(x_positions, sma_200, label='SMA 200', color='magenta', linewidth=1.5, linestyle='--', alpha=0.8)

    # Bollinger Bands - draw boundary lines and fill (original colors)
    ax1.plot(x_positions, bb['upper'], label='Upper BB (2σ)', color='g', linewidth=1, alpha=0.7, linestyle='--')
    ax1.plot(x_positions, bb['middle'], label='SMA (20)', color='b', linewidth=1)
    ax1.plot(x_positions, bb['lower'], label='Lower BB (2σ)', color='r', linewidth=1, alpha=0.7, linestyle='--')
    ax1.fill_between(x_positions, bb['lower'], bb['upper'], alpha=0.1, color='gray')

    # Smoothed price line (solid green)
    ax1.plot(x_positions, smoothed.values, label='Smoothed', color=SMOOTHED_COLOR,
             linewidth=1.2, alpha=0.8)

    # Add swing point labels (HH, HL, LH, LL) - same as inflection_detector
    if 'swing_label' in data.columns:
        for idx in range(len(data)):
            label = data['swing_label'].iloc[idx]
            if label and label in SWING_LABEL_COLORS:
                is_major = data['is_major_swing'].iloc[idx]

                # Position label above highs, below lows
                if label in ['HH', 'LH']:
                    y_pos = data['High'].iloc[idx]
                    va = 'bottom'
                    offset = data['High'].iloc[idx] * 0.01  # 1% offset
                else:  # HL, LL
                    y_pos = data['Low'].iloc[idx]
                    va = 'top'
                    offset = -data['Low'].iloc[idx] * 0.01

                # Same font size for all, but bold for major swings
                fontweight = 'bold' if is_major else 'normal'
                alpha = 1.0 if is_major else 0.7

                ax1.annotate(label, xy=(x_positions[idx], y_pos + offset),
                            fontsize=8, fontweight=fontweight,
                            color=SWING_LABEL_COLORS[label],
                            ha='center', va=va, alpha=alpha,
                            zorder=6)

    # Title with score
    title = f'{ticker} - Technical Analysis'
    if score is not None:
        title += f' | Trend Score: {score:.1f}/6.0'
    if interval != '1d':
        title += f' ({interval} - Last 2 Days)'
    ax1.set_title(title, fontsize=16, fontweight='bold')

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.6)  # Increased grid opacity

    # Format x-axis with ticks but no date labels on price chart
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))  # 3 minor ticks between major ticks (n=4)
    ax1.tick_params(axis='x', which='major', direction='in', length=8)  # Major ticks inside
    ax1.tick_params(axis='x', which='minor', direction='in', length=5)  # Minor ticks inside
    ax1.tick_params(axis='x', which='major', labeltop=False, labelbottom=False)  # Hide date labels
    ax1.tick_params(axis='x', labelsize=0)  # Ensure labels are hidden

    # Add secondary y-axis for price on right side
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_ylabel('Price ($)', fontsize=12)

    # Add earnings date annotations ("E") on price chart - same as inflection_detector
    if earnings_dates:
        # Get y-axis limits to position "E" at bottom
        y_min, y_max = ax1.get_ylim()
        y_pos = y_min

        for earn_date in earnings_dates:
            # Convert datetime to date for comparison
            earn_date_compare = earn_date.date() if hasattr(earn_date, 'date') else earn_date

            # Find matching date in data index
            for idx, (i, row) in enumerate(data.iterrows()):
                row_date = i.date() if hasattr(i, 'date') else i
                if row_date == earn_date_compare:
                    ax1.annotate('E', xy=(x_positions[idx], y_pos),
                                fontsize=10, fontweight='bold',
                                color='purple', ha='center', va='bottom',
                                zorder=7)
                    break

    # Add timestamp at lower right of price chart
    chart_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax1.text(0.98, 0.02, f'Generated: {chart_timestamp}',
            transform=ax1.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add "Times are CDT" label for intraday charts
    if interval != '1d':
        ax1.text(0.02, 0.02, 'Times are CDT', transform=ax1.transAxes,
                fontsize=8, style='italic', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # ============================================
    # PANEL 2: VOLUME WITH EMA5/EMA20 ON SECONDARY AXIS
    # ============================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot volume bars - color based on price movement (close >= open = green, else red)
    # Same setup as inflection_detector
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
              for i in range(len(data))]
    ax2.bar(x_positions, data['Volume'], color=colors, alpha=0.65, width=0.8)

    # Volume MA50 on left axis (original purple color)
    vol_line = ax2.plot(x_positions, volume_ma_50, label='Volume MA (50)', color='purple', linewidth=1, alpha=0.8)
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.grid(True, alpha=0.6)  # Increased grid opacity
    ax2.tick_params(axis='x', which='major', direction='in', length=8)
    ax2.tick_params(axis='x', which='minor', direction='in', length=5)
    plt.setp(ax2.get_xticklabels(), visible=False)  # Hide date labels

    # Add secondary y-axis for SMAs (right side) - original colors
    ax2_price = ax2.twinx()
    ema5_line = ax2_price.plot(x_positions, ema_5, label='SMA (5)', color='orange', linewidth=1, alpha=0.8)
    ema20_line = ax2_price.plot(x_positions, ema_20, label='SMA (20)', color='blue', linewidth=1, alpha=0.8)
    ax2_price.set_ylabel('Price ($)', fontsize=10)
    ax2_price.tick_params(axis='y', labelsize=9)

    # Combine legends from both axes
    lines = vol_line + ema5_line + ema20_line
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=8)
    
    # ============================================
    # PANEL 3: RSI WITH COLOR CODING (ORIGINAL)
    # ============================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Plot RSI with color segments (original color-coding, using sequential positions)
    for i in range(len(rsi) - 1):
        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i + 1]):
            continue
        x = [x_positions[i], x_positions[i + 1]]
        y = [rsi.iloc[i], rsi.iloc[i + 1]]
        color = get_rsi_color(rsi.iloc[i])
        ax3.plot(x, y, color=color, linewidth=1)

    # Add horizontal reference lines (same as inflection_detector)
    ax3.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax3.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    ax3.axhline(y=50, color='magenta', linestyle='-', linewidth=1, alpha=0.8)

    # Shade overbought/oversold regions (same as inflection_detector)
    ax3.fill_between(x_positions, 70, 100, alpha=0.1, color='red')
    ax3.fill_between(x_positions, 0, 30, alpha=0.1, color='green')

    ax3.set_ylabel('RSI (14)', fontsize=8)  # Smaller y-axis label
    ax3.set_ylim(0, 100)

    # Create custom legend for RSI colors (same thresholds as inflection_detector)
    rsi_legend_elements = [
        mpatches.Patch(color=RSI_COLOR_OVERSOLD, label='RSI < 30 (Oversold)'),
        mpatches.Patch(color=RSI_COLOR_BEARISH, label='RSI 30-50 (Bearish)'),
        mpatches.Patch(color=RSI_COLOR_BULLISH, label='RSI 50-70 (Bullish)'),
        mpatches.Patch(color=RSI_COLOR_OVERBOUGHT, label='RSI > 70 (Overbought)')
    ]
    ax3.legend(handles=rsi_legend_elements, loc='upper left', fontsize=8, ncol=2)

    # Add current RSI value annotation
    if not pd.isna(rsi.iloc[-1]):
        current_rsi = rsi.iloc[-1]
        rsi_color = get_rsi_color(current_rsi)
        ax3.text(0.98, 0.95, f'Current RSI: {current_rsi:.1f}',
                transform=ax3.transAxes, fontsize=10, fontweight='bold',
                ha='right', va='top', color=rsi_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.grid(True, alpha=0.6)  # Increased grid opacity
    ax3.tick_params(axis='y', labelsize=8)  # Smaller y-axis tick labels
    ax3.tick_params(axis='x', which='major', direction='in', length=8)
    ax3.tick_params(axis='x', which='minor', direction='in', length=5)
    plt.setp(ax3.get_xticklabels(), visible=False)  # Hide date labels

    # ============================================
    # PANEL 4: VELOCITY & ACCELERATION (SHOWS DATES)
    # ============================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # Velocity line (left Y axis) - blue, same as smoothed price
    ax4.plot(x_positions, velocity.values, color=VELOCITY_COLOR,
             linewidth=1.2, label='Velocity', alpha=1.0)
    ax4.axhline(y=0, color=VELOCITY_COLOR, linestyle='-', linewidth=0.8, alpha=1.0)
    ax4.set_ylabel('Velocity', color=VELOCITY_COLOR, fontsize=10)
    ax4.tick_params(axis='y', labelcolor=VELOCITY_COLOR, labelsize=8)
    ax4.grid(True, alpha=0.3)

    # Acceleration line (right Y axis) - gold/amber
    ax4_right = ax4.twinx()
    ax4_right.plot(x_positions, acceleration.values, color=ACCEL_COLOR,
                   linewidth=1.2, label='Acceleration', alpha=0.8)
    ax4_right.axhline(y=0, color=ACCEL_COLOR, linestyle='-', linewidth=0.8, alpha=1.0)
    ax4_right.set_ylabel('Acceleration', color=ACCEL_COLOR, fontsize=10)
    ax4_right.tick_params(axis='y', labelcolor=ACCEL_COLOR, labelsize=8)

    # Shaded regions based on velocity/acceleration sign combinations
    # Same colors as inflection_detector
    vel_values = velocity.values
    acc_values = acceleration.values

    for i in range(len(vel_values)):
        vel_pos = vel_values[i] > 0
        acc_pos = acc_values[i] > 0

        if vel_pos and acc_pos:
            color = COLOR_VEL_POS_ACC_POS  # Bright green - rising & steepening
        elif vel_pos and not acc_pos:
            color = COLOR_VEL_POS_ACC_NEG  # Medium green - rising but flattening
        elif not vel_pos and acc_pos:
            color = COLOR_VEL_NEG_ACC_POS  # Medium red - falling but flattening
        else:
            color = COLOR_VEL_NEG_ACC_NEG  # Bright red - falling & steepening

        # Shade a vertical bar for this data point
        if i < len(x_positions):
            width = 1 if i == len(x_positions) - 1 else x_positions[min(i+1, len(x_positions)-1)] - x_positions[i]
            ax4.axvspan(x_positions[i], x_positions[i] + width,
                       facecolor=color, alpha=0.35, zorder=0)

    # =================================================================
    # Draw vertical lines (same as inflection_detector)
    # SOLID lines at velocity zero crossings
    # DASHED lines at quadrant changes (excluding velocity crossings)
    # =================================================================
    axes_for_vlines = [ax1, ax2, ax3, ax4]  # Price, Volume, RSI, Velocity panels

    # Detect velocity zero crossings
    velocity_crossings = []  # List of (index, is_bullish) tuples
    for i in range(1, len(vel_values)):
        if vel_values[i-1] * vel_values[i] < 0:  # Sign change = zero crossing
            # Determine direction: negative to positive = bullish (up), positive to negative = bearish (down)
            is_bullish = vel_values[i-1] < 0 and vel_values[i] > 0
            velocity_crossings.append((i, is_bullish))

    velocity_crossing_indices = set(idx for idx, _ in velocity_crossings)

    # Draw SOLID vertical lines at velocity zero crossings
    for idx, is_bullish in velocity_crossings:
        # Use RSI blue (#2196F3) for bullish, darkorange for bearish (same as inflection_detector)
        line_color = RSI_COLOR_BULLISH if is_bullish else 'darkorange'
        for ax in axes_for_vlines:
            ax.axvline(x=x_positions[idx], color=line_color, linestyle='-', alpha=1.0, linewidth=1.5)

    # Draw DASHED vertical lines at quadrant changes (excluding velocity crossings)
    def get_quadrant_color(vel_pos, acc_pos):
        if vel_pos and acc_pos:
            return COLOR_VEL_POS_ACC_POS
        elif vel_pos and not acc_pos:
            return COLOR_VEL_POS_ACC_NEG
        elif not vel_pos and acc_pos:
            return COLOR_VEL_NEG_ACC_POS
        else:
            return COLOR_VEL_NEG_ACC_NEG

    for i in range(1, len(vel_values)):
        # Skip if this index already has a velocity crossing solid line
        if i in velocity_crossing_indices:
            continue

        prev_vel_pos = vel_values[i-1] > 0
        prev_acc_pos = acc_values[i-1] > 0
        curr_vel_pos = vel_values[i] > 0
        curr_acc_pos = acc_values[i] > 0

        # Check if quadrant changed (either velocity or acceleration sign changed)
        if (prev_vel_pos != curr_vel_pos) or (prev_acc_pos != curr_acc_pos):
            # Use the color of the NEW quadrant
            line_color = get_quadrant_color(curr_vel_pos, curr_acc_pos)

            # Draw dashed vertical line on all panels
            if i < len(x_positions):
                for ax in axes_for_vlines:
                    ax.axvline(x=x_positions[i], color=line_color, linestyle='--', alpha=0.8, linewidth=1.0)

    # Combined legend for velocity and acceleration
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_right.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # Calculate appropriate tick positions and labels (using sequential positions)
    if interval != '1d':
        # For intraday, show every Nth point to avoid overcrowding
        num_ticks = min(12, len(data))  # Max 12 ticks
    else:
        # For daily data, show dates at regular intervals
        num_ticks = min(10, len(data))  # Max 10 ticks

    # Get evenly spaced tick positions
    tick_positions = np.linspace(0, len(data) - 1, num_ticks, dtype=int)

    # Get corresponding date labels
    if interval != '1d':
        tick_labels = [data.index[i].strftime('%m/%d\n%H:%M') for i in tick_positions]
    else:
        tick_labels = [data.index[i].strftime('%m/%d/%y') for i in tick_positions]

    # Set x-axis limits with 3-day gap on the right
    x_min = 0
    # Calculate bars per day based on interval for the right gap
    if interval == '1d':
        bars_for_gap = 3  # 3 days
    elif interval in ['5min', '3min']:
        bars_for_gap = 3 * 78  # ~78 bars per day for 5min (6.5 hours * 12)
    elif interval == '15min':
        bars_for_gap = 3 * 26  # ~26 bars per day for 15min
    elif interval == '30min':
        bars_for_gap = 3 * 13  # ~13 bars per day for 30min
    elif interval == '60min' or interval == '1hour':
        bars_for_gap = 3 * 7  # ~7 bars per day for 60min
    else:
        bars_for_gap = 3  # Default to 3 bars
    x_max = len(data) - 1 + bars_for_gap

    # Apply tick settings to velocity/acceleration panel (only panel showing dates)
    ax4.set_xlim(x_min, x_max)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax4.xaxis.set_minor_locator(AutoMinorLocator(4))  # 3 minor ticks between major ticks (n=4)
    ax4.tick_params(axis='x', which='major', direction='inout', length=10)  # Larger major ticks crossing
    ax4.tick_params(axis='x', which='minor', direction='inout', length=7)  # Larger minor ticks crossing
    ax4.tick_params(axis='x', labelsize=9)

    # ============================================
    # PANEL 5: ANALYSIS SUMMARY
    # ============================================
    ax5 = fig.add_subplot(gs[4])
    ax5.axis('off')

    # Build summary text
    latest_price = data['Close'].iloc[-1]
    latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 0
    latest_velocity = velocity.iloc[-1] if not pd.isna(velocity.iloc[-1]) else 0
    latest_accel = acceleration.iloc[-1] if not pd.isna(acceleration.iloc[-1]) else 0
    latest_volume = data['Volume'].iloc[-1]

    info_text = f"ANALYSIS SUMMARY\n{'='*170}\n\n"
    info_text += f"Price: ${latest_price:.2f}  |  Score: {score:.1f}/6.0  |  RSI: {latest_rsi:.1f}  |  Velocity: {latest_velocity:.4f}  |  Accel: {latest_accel:.4f}  |  Volume: {latest_volume:,.0f}\n\n"
    info_text += f"Date Range: {date_range}\n"

    # Add interval info for intraday
    if interval != '1d':
        info_text += f"Interval: {interval} (last 2 days shown)\n"

    # Position text to span wider
    ax5.text(0.0, 0.95, info_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ============================================
    # MARKET HOURS MARKERS (APPLY TO ALL PANELS)
    # ============================================
    all_axes = [ax1, ax2, ax3, ax4]

    for ax in all_axes:
        # Add pre-market start markers (3:00 AM CT) for intraday charts (using positions)
        if premarket_start_times:
            for premarket_pos in premarket_start_times:
                ax.axvline(x=premarket_pos, color='green', linestyle='-', linewidth=1, alpha=0.75, zorder=1)

        # Add market open markers (8:30 AM CT) (using positions)
        if market_open_times:
            for market_open_pos in market_open_times:
                ax.axvline(x=market_open_pos, color='green', linestyle='-', linewidth=1, alpha=1.0, zorder=2)

        # Add market close markers (3:00 PM CT) (using positions)
        if market_close_times:
            for market_close_pos in market_close_times:
                ax.axvline(x=market_close_pos, color='red', linestyle='-', linewidth=1, alpha=1.0, zorder=2)

    # ============================================
    # FINAL SAVE
    # ============================================

    # Adjust layout to center plots with balanced margins
    plt.subplots_adjust(left=0.08, right=0.92, top=0.97, bottom=0.05)

    # Save figure with optional rank prefix
    if rank is not None:
        filename = f'{rank:02d}. {ticker}_technical_analysis.png'
    else:
        filename = f'{ticker}_technical_analysis.png'

    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_file


def create_individual_plots_for_all(results_df, client, interval, outputsize, output_dir='individual_plots', start_date=None, display_interval=None):
    """
    Create individual technical analysis plots for all analyzed stocks

    Args:
        results_df: DataFrame with analysis results
        client: MassiveClient instance to fetch data
        interval: Data interval ('1d', '5min', etc.) for fetching data
        outputsize: 'compact' or 'full'
        output_dir: Directory to save plots
        start_date: Optional start date (YYYY-MM-DD) to filter data from
        display_interval: Display interval for plot titles (defaults to interval if not provided)

    Returns:
        List of created plot files
    """

    # Use display_interval for titles if provided, otherwise use actual interval
    if display_interval is None:
        display_interval = interval
    
    if results_df.empty:
        print("No stocks to plot")
        return []
    
    date_info = f" (from {start_date})" if start_date else ""
    print(f"\nGenerating individual technical analysis charts for {len(results_df)} stocks{date_info}...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_files = []

    for rank, (i, row) in enumerate(results_df.iterrows(), start=1):
        ticker = row['ticker']
        score = row.get('score', None)

        print(f"  [{rank:02d}] Creating chart for {ticker}... ", end='', flush=True)

        try:
            # Fetch data
            if interval == '1d':
                data = client.get_daily_data(ticker, outputsize)
            else:
                # Interval is already in correct format (e.g., '5min')
                data = client.get_intraday_data(ticker, interval, outputsize)

            if data is None or data.empty:
                print("❌ No data")
                continue

            # Create plot with start_date filter, rank, interval, and client for earnings
            # Pass display_interval for plot titles
            plot_file = plot_ticker_technical_analysis(ticker, data, score, output_dir, start_date, rank, display_interval, client)
            if plot_file:
                plot_files.append(plot_file)
                print(f"✓ Saved")
            else:
                print(f"⚠️ Skipped (no data in date range)")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print(f"\n✓ Generated {len(plot_files)} individual charts in '{output_dir}/' directory")
    
    return plot_files


def main():
    """Standalone usage for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_individual_tickers.py <ticker>")
        print("Example: python plot_individual_tickers.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    # This is just for demonstration
    print(f"To use this, integrate with stock_trend_analyzer.py")
    print(f"Example: python stock_trend_analyzer.py --tickers {ticker} --plot --plot-individual")


if __name__ == "__main__":
    main()
