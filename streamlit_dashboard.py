"""
Streamlit Live Dashboard for Stock Trend Analyzer
Web-based version replicating the matplotlib live_dashboard.py
Displays heatmap + top 6 stocks with 4-panel technical analysis

Run locally: streamlit run streamlit_dashboard.py
Deploy: Push to GitHub and connect to Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime
import time
import os
from scipy.ndimage import gaussian_filter1d

# Import from the main analyzer
from stock_trend_analyzer import StockTrendAnalyzer, MassiveClient, RateLimiter
import config

# Page configuration
st.set_page_config(
    page_title="Stock Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color constants (same as live_dashboard.py)
RSI_COLOR_OVERSOLD = '#FF4444'    # Red - oversold (< 30)
RSI_COLOR_BEARISH = '#FFA500'     # Orange - bearish (30-50)
RSI_COLOR_BULLISH = '#4169E1'     # Royal blue - bullish (50-80)
RSI_COLOR_OVERBOUGHT = '#FFD700'  # Gold - overbought (> 80)

SMOOTHED_COLOR = 'blue'
VELOCITY_COLOR = 'blue'
ACCEL_COLOR = '#C49821'  # Gold/amber

# Quadrant colors for velocity/acceleration shading
COLOR_VEL_POS_ACC_POS = 'rgba(0, 200, 83, 0.35)'   # Bright green
COLOR_VEL_POS_ACC_NEG = 'rgba(105, 240, 174, 0.35)'  # Medium green
COLOR_VEL_NEG_ACC_POS = 'rgba(255, 138, 128, 0.35)'  # Medium red
COLOR_VEL_NEG_ACC_NEG = 'rgba(213, 0, 0, 0.35)'    # Bright red


def get_api_key():
    """Get API key from Streamlit secrets or environment"""
    if hasattr(st, 'secrets') and 'MASSIVE_API_KEY' in st.secrets:
        return st.secrets['MASSIVE_API_KEY']
    return os.getenv('MASSIVE_API_KEY')


def load_tickers_from_file(filename):
    """Load tickers from input file"""
    # Try multiple possible paths for Streamlit Cloud compatibility
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_files', filename),
        os.path.join('input_files', filename),
        filename
    ]

    for filepath in possible_paths:
        if os.path.exists(filepath):
            tickers = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip().upper()
                    if line and not line.startswith('#'):
                        tickers.append(line)
            return tickers
    return []


def get_available_input_files():
    """Get list of available input files"""
    possible_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_files'),
        'input_files'
    ]

    for input_files_dir in possible_dirs:
        if os.path.exists(input_files_dir):
            return [f for f in os.listdir(input_files_dir) if f.endswith('.txt')]
    return []


def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    return {
        'upper': sma + (std * num_std),
        'middle': sma,
        'lower': sma - (std * num_std)
    }


def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_smoothed_velocity_acceleration(data, sigma=3):
    """Calculate smoothed price, velocity, and acceleration using Gaussian smoothing"""
    prices = data['Close'].values.astype(float)
    smoothed = gaussian_filter1d(prices, sigma=sigma)
    velocity = np.gradient(smoothed)
    acceleration = np.gradient(velocity)
    return {
        'smoothed': pd.Series(smoothed, index=data.index),
        'velocity': pd.Series(velocity, index=data.index),
        'acceleration': pd.Series(acceleration, index=data.index)
    }


def get_rsi_color(rsi_value):
    """Get color based on RSI value"""
    if pd.isna(rsi_value):
        return 'gray'
    elif rsi_value < 30:
        return RSI_COLOR_OVERSOLD
    elif 30 <= rsi_value < 50:
        return RSI_COLOR_BEARISH
    elif 50 <= rsi_value <= 80:
        return RSI_COLOR_BULLISH
    else:
        return RSI_COLOR_OVERBOUGHT


def create_stock_chart(ticker, data, result, interval):
    """
    Create a 4-panel Plotly chart replicating live_dashboard.py layout:
    Panel 1: Price with candlesticks, Bollinger Bands, and smoothed line
    Panel 2: Volume with SMA5/SMA20 on secondary axis
    Panel 3: RSI with color coding
    Panel 4: Velocity & Acceleration with quadrant shading
    """
    # Calculate indicators
    bb = calculate_bollinger_bands(data, period=20, num_std=2)
    rsi = calculate_rsi(data, period=14)
    sma_5 = data['Close'].rolling(window=5).mean()
    sma_20 = data['Close'].rolling(window=20).mean()
    volume_ma_50 = data['Volume'].rolling(window=50).mean()
    derivatives = calculate_smoothed_velocity_acceleration(data, sigma=3)
    smoothed = derivatives['smoothed']
    velocity = derivatives['velocity']
    acceleration = derivatives['acceleration']

    # Create sequential x-axis positions (eliminates gaps)
    x_positions = list(range(len(data)))

    # Create date labels for tick labels
    # For intraday: show MM/DD<br>HH:MM (CT time format matching live_dashboard)
    # For daily: show MM/DD only
    if interval != '1d':
        date_labels = [d.strftime('%m/%d<br>%H:%M') for d in data.index]
    else:
        date_labels = [d.strftime('%m/%d') for d in data.index]

    # Create subplots with 4 rows
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    # ============================================
    # PANEL 1: PRICE WITH BOLLINGER BANDS
    # ============================================

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=x_positions,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#56B05C',
            decreasing_line_color='#F77272',
            increasing_fillcolor='#56B05C',
            decreasing_fillcolor='#F77272'
        ),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=x_positions, y=bb['upper'], mode='lines',
                   name='BB Upper', line=dict(color='green', width=1, dash='dash'),
                   opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_positions, y=bb['middle'], mode='lines',
                   name='BB Middle', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_positions, y=bb['lower'], mode='lines',
                   name='BB Lower', line=dict(color='red', width=1, dash='dash'),
                   fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                   opacity=0.7),
        row=1, col=1
    )

    # Smoothed price line
    fig.add_trace(
        go.Scatter(x=x_positions, y=smoothed.values, mode='lines',
                   name='Smoothed', line=dict(color=SMOOTHED_COLOR, width=1.5),
                   opacity=0.8),
        row=1, col=1
    )

    # Add market hour vertical lines for intraday intervals
    if interval != '1d':
        market_open_positions = []
        market_close_positions = []

        # Find market open (8:30 AM CT) and close (3:00 PM CT) positions
        for i, dt in enumerate(data.index):
            hour = dt.hour
            minute = dt.minute

            # Market open at 8:30 AM CT
            if hour == 8 and minute == 30:
                market_open_positions.append(x_positions[i])

            # Market close at 3:00 PM CT (15:00)
            if hour == 15 and minute == 0:
                market_close_positions.append(x_positions[i])

        # Add vertical lines for market open
        for pos in market_open_positions:
            fig.add_vline(x=pos, line_color='green', line_width=1,
                          line_dash='solid', opacity=1.0)

        # Add vertical lines for market close
        for pos in market_close_positions:
            fig.add_vline(x=pos, line_color='red', line_width=1,
                          line_dash='solid', opacity=1.0)

    # ============================================
    # PANEL 2: VOLUME WITH SMA5/SMA20
    # ============================================

    # Volume bars
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=x_positions, y=data['Volume'], name='Volume',
               marker_color=colors, opacity=0.65),
        row=2, col=1, secondary_y=False
    )

    # Volume MA50
    fig.add_trace(
        go.Scatter(x=x_positions, y=volume_ma_50, mode='lines',
                   name='Vol MA50', line=dict(color='purple', width=1)),
        row=2, col=1, secondary_y=False
    )

    # SMA5 and SMA20 on secondary y-axis
    fig.add_trace(
        go.Scatter(x=x_positions, y=sma_5, mode='lines',
                   name='SMA5', line=dict(color='orange', width=1)),
        row=2, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=x_positions, y=sma_20, mode='lines',
                   name='SMA20', line=dict(color='blue', width=1)),
        row=2, col=1, secondary_y=True
    )

    # ============================================
    # PANEL 3: RSI WITH COLOR CODING
    # ============================================

    # Create RSI segments with different colors
    for i in range(len(rsi) - 1):
        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i + 1]):
            continue
        color = get_rsi_color(rsi.iloc[i])
        fig.add_trace(
            go.Scatter(
                x=[x_positions[i], x_positions[i + 1]],
                y=[rsi.iloc[i], rsi.iloc[i + 1]],
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )

    # RSI horizontal lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="magenta", line_width=1, row=3, col=1)

    # RSI shaded regions
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0, row=3, col=1)

    # ============================================
    # PANEL 4: VELOCITY & ACCELERATION
    # ============================================

    # Add quadrant shading based on velocity/acceleration signs
    # Using shapes with explicit axis references for proper subplot positioning
    vel_values = velocity.values
    acc_values = acceleration.values

    # Get y-axis range for panel 4 (velocity)
    vel_min, vel_max = velocity.min(), velocity.max()
    vel_padding = (vel_max - vel_min) * 0.1 if vel_max != vel_min else 1
    y_range = [vel_min - vel_padding, vel_max + vel_padding]

    for i in range(len(vel_values)):
        vel_pos = vel_values[i] > 0
        acc_pos = acc_values[i] > 0

        if vel_pos and acc_pos:
            color = COLOR_VEL_POS_ACC_POS
        elif vel_pos and not acc_pos:
            color = COLOR_VEL_POS_ACC_NEG
        elif not vel_pos and acc_pos:
            color = COLOR_VEL_NEG_ACC_POS
        else:
            color = COLOR_VEL_NEG_ACC_NEG

        if i < len(x_positions):
            # Use add_shape with explicit axis references for row 4
            # Subplot axis naming: row1=y, row2=y2/y3, row3=y4, row4=y5/y6
            # x-axes: x (row1), x2 (row2), x3 (row3), x4 (row4)
            fig.add_shape(
                type="rect",
                x0=x_positions[i] - 0.5, x1=x_positions[i] + 0.5,
                y0=y_range[0], y1=y_range[1],
                fillcolor=color,
                line_width=0,
                layer="below",
                xref="x4", yref="y5"
            )

    # Velocity line (primary y-axis)
    fig.add_trace(
        go.Scatter(x=x_positions, y=velocity.values, mode='lines',
                   name='Velocity', line=dict(color=VELOCITY_COLOR, width=1.5)),
        row=4, col=1, secondary_y=False
    )

    # Acceleration line (secondary y-axis)
    fig.add_trace(
        go.Scatter(x=x_positions, y=acceleration.values, mode='lines',
                   name='Acceleration', line=dict(color=ACCEL_COLOR, width=1.5)),
        row=4, col=1, secondary_y=True
    )

    # Zero lines for velocity and acceleration
    fig.add_hline(y=0, line_color=VELOCITY_COLOR, line_width=1, row=4, col=1)

    # ============================================
    # LAYOUT AND FORMATTING
    # ============================================

    # Get result data
    current_price = result.get('current_price', data['Close'].iloc[-1]) if result else data['Close'].iloc[-1]
    score = result.get('score', 0) if result else 0
    momentum = result.get('momentum', {}) if result else {}
    gain_1d = momentum.get('1d', 0)

    # Create tick labels (show every Nth label)
    num_ticks = min(8, len(data))
    tick_indices = np.linspace(0, len(data) - 1, num_ticks, dtype=int)
    tick_labels = [date_labels[i] for i in tick_indices]

    # Title with score, ticker, and daily gain
    title_text = f"<b>{score:.1f} | {ticker}"
    if interval != '1d':
        title_text += f" ({interval})"
    if gain_1d != 'N/A' and gain_1d is not None:
        title_text += f" | {gain_1d:+.2f}%"
    title_text += f" | ${current_price:.2f}</b>"

    # Calculate width based on data points to maintain proportionality
    # Minimum 800px, scale up for more data points
    min_width = 800
    width_per_point = 8  # pixels per data point
    calculated_width = max(min_width, len(data) * width_per_point)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        height=700,
        width=calculated_width,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=50, b=60),
        hovermode='x unified'
    )

    # Update axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="Velocity", row=4, col=1, secondary_y=False,
                     title_font=dict(color=VELOCITY_COLOR), range=y_range)
    fig.update_yaxes(title_text="Accel", row=4, col=1, secondary_y=True,
                     title_font=dict(color=ACCEL_COLOR))

    # Update x-axes with proper tick labels (CT times for intraday)
    # Apply to all x-axes since they're shared
    for row in range(1, 5):
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(tick_indices),
            ticktext=tick_labels,
            row=row, col=1
        )

    # Hide x-axis labels for upper panels (only bottom shows labels)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(showticklabels=True, row=4, col=1)

    return fig


def create_heatmap(results):
    """Create a heatmap showing all stocks by score and daily gain (replicating live_dashboard)"""
    if not results:
        return None

    # Sort by score (descending)
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    tickers = [r['ticker'] for r in sorted_results]
    scores = [r.get('score', 0) for r in sorted_results]

    # Get daily gains
    daily_gains = []
    for r in sorted_results:
        momentum = r.get('momentum', {})
        gain = momentum.get('1d', 0)
        if isinstance(gain, str) or gain is None:
            gain = 0
        daily_gains.append(gain)

    # Create colors based on daily gain
    max_abs_gain = max(abs(min(daily_gains)) if daily_gains else 1,
                       abs(max(daily_gains)) if daily_gains else 1, 0.1)

    colors = []
    for gain in daily_gains:
        if gain >= 0:
            intensity = min(gain / max_abs_gain, 1.0)
            colors.append(f'rgb({int(51)}, {int(153 + 102 * intensity)}, {int(51)})')
        else:
            intensity = min(abs(gain) / max_abs_gain, 1.0)
            colors.append(f'rgb({int(204 + 51 * intensity)}, {int(51)}, {int(51)})')

    # Create figure
    fig = go.Figure()

    # Add horizontal bars
    fig.add_trace(go.Bar(
        y=list(range(len(tickers))),
        x=scores,
        orientation='h',
        marker_color=colors,
        marker_opacity=1.0,
        text=[f"{s:.1f}" for s in scores],
        textposition='outside',
        textfont=dict(color='rgba(0,0,0,1)', size=11),
        hovertemplate='%{customdata[0]}<br>Score: %{x:.1f}<br>Gain: %{customdata[1]:+.1f}%<extra></extra>',
        customdata=list(zip(tickers, daily_gains))
    ))

    # Add threshold line at 4.0
    fig.add_vline(x=4.0, line_dash="dash", line_color="orange", line_width=2)

    # Add ticker labels and gain annotations
    annotations = []
    for i, (ticker, gain) in enumerate(zip(tickers, daily_gains)):
        # Ticker label on left
        annotations.append(dict(
            x=-0.3, y=i,
            text=f"<b>{ticker}</b>",
            showarrow=False,
            font=dict(size=10, color='rgba(0,0,0,1)'),
            xanchor='right',
            opacity=1.0
        ))
        # Gain label on right - use rgba for full opacity
        gain_color = 'rgba(0,128,0,1)' if gain >= 0 else 'rgba(255,0,0,1)'
        annotations.append(dict(
            x=6.5, y=i,
            text=f"<b>{gain:+.1f}%</b>",
            showarrow=False,
            font=dict(size=12, color=gain_color),
            xanchor='left',
            opacity=1.0
        ))

    fig.update_layout(
        title=dict(text="<b>All Stocks by Score</b><br><sub>(Color = Daily Gain)</sub>",
                   font=dict(size=12)),
        xaxis=dict(title="Score (out of 6)", range=[-0.5, 7.5], dtick=1),
        yaxis=dict(showticklabels=False, autorange='reversed'),
        height=max(400, len(tickers) * 28),
        margin=dict(l=80, r=80, t=60, b=40),
        annotations=annotations,
        showlegend=False
    )

    return fig


def main():
    """Main Streamlit app - replicating live_dashboard.py"""

    # Sidebar
    st.sidebar.title("📈 Stock Trend Analyzer")
    st.sidebar.markdown("---")

    # API Key check
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ MASSIVE_API_KEY not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()

    # Settings
    st.sidebar.subheader("Settings")

    # Input file selection
    available_files = get_available_input_files()
    if available_files:
        selected_file = st.sidebar.selectbox(
            "Input File",
            available_files,
            index=available_files.index('sample_ticker.txt') if 'sample_ticker.txt' in available_files else 0
        )
    else:
        selected_file = None
        st.sidebar.warning("No input files found in input_files/")

    # Manual ticker input with default value
    default_tickers = "AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN"
    manual_tickers = st.sidebar.text_input(
        "Or enter tickers (comma-separated)",
        value=default_tickers,
        help="Enter ticker symbols separated by commas"
    )

    # Interval selection
    interval = st.sidebar.selectbox(
        "Interval",
        ['5min', '15min', '30min', '60min', '1d'],
        index=0
    )

    # Refresh settings
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        disabled=not auto_refresh
    )

    # Number of charts to show (max 6 for grid layout)
    num_charts = st.sidebar.slider(
        "Number of detailed charts",
        min_value=1,
        max_value=6,
        value=6
    )

    st.sidebar.markdown("---")

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Get current time
    ct_tz = pytz.timezone('US/Central')
    current_time = datetime.now(ct_tz).strftime('%Y-%m-%d %H:%M:%S CT')
    st.sidebar.markdown(f"**Last update:** {current_time}")

    # Load tickers - prefer manual input, fall back to file
    if manual_tickers and manual_tickers.strip():
        tickers = [t.strip().upper() for t in manual_tickers.split(',') if t.strip()]
    elif selected_file:
        tickers = load_tickers_from_file(selected_file)
    else:
        tickers = []

    if not tickers:
        st.warning("No tickers to analyze. Enter tickers in the sidebar.")
        st.stop()

    st.sidebar.markdown(f"**Analyzing:** {len(tickers)} tickers")

    # Main content
    st.title("📊 Live Stock Dashboard")
    st.markdown(f"*Updated: {current_time}* | *Replicating live_dashboard.py layout*")

    # Initialize analyzer and analyze stocks
    with st.spinner("Analyzing stocks..."):
        analyzer = StockTrendAnalyzer(
            api_key=api_key,
            interval=interval,
            period='full'
        )

        results = []
        progress_bar = st.progress(0)

        for i, ticker in enumerate(tickers):
            result = analyzer.is_trending_up(ticker)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(tickers))

        progress_bar.empty()

    if not results:
        st.error("No data available for the selected tickers.")
        st.stop()

    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Summary metrics (same as live_dashboard)
    col1, col2, col3, col4 = st.columns(4)

    trending_count = sum(1 for r in results if r.get('score', 0) >= 4.0)
    avg_score = sum(r.get('score', 0) for r in results) / len(results)
    avg_gain = sum(r.get('momentum', {}).get('1d', 0) or 0 for r in results) / len(results)
    top_ticker = results[0]['ticker'] if results else "N/A"

    with col1:
        st.metric("Trending Stocks", f"{trending_count}/{len(results)}",
                  help="Stocks with score >= 4.0")
    with col2:
        st.metric("Avg Score", f"{avg_score:.2f}")
    with col3:
        delta_color = "normal" if avg_gain >= 0 else "inverse"
        st.metric("Avg Daily Gain", f"{avg_gain:+.2f}%")
    with col4:
        top_score = results[0].get('score', 0) if results else 0
        st.metric("Top Stock", top_ticker, f"Score: {top_score:.1f}")

    st.markdown("---")

    # Two-column layout: Heatmap on left, Charts on right
    col_heatmap, col_charts = st.columns([1, 3])

    with col_heatmap:
        st.subheader("📊 Stock Rankings")
        heatmap = create_heatmap(results)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

    with col_charts:
        st.subheader(f"📈 Top {num_charts} Stocks (4-Panel Technical Analysis)")

        # Get top N results
        top_results = results[:num_charts]

        # Create client for fetching data
        client = MassiveClient(api_key, RateLimiter())

        # Display charts in a 2x3 grid (or fewer if num_charts < 6)
        rows = (num_charts + 2) // 3  # Calculate number of rows needed

        for row_idx in range(rows):
            cols = st.columns(min(3, num_charts - row_idx * 3))

            for col_idx, col in enumerate(cols):
                chart_idx = row_idx * 3 + col_idx
                if chart_idx >= len(top_results):
                    break

                result = top_results[chart_idx]
                ticker = result['ticker']

                with col:
                    # Fetch data for chart
                    if interval == '1d':
                        data = client.get_daily_data(ticker, 'compact')
                    else:
                        data = client.get_intraday_data(ticker, interval, 'compact')

                    if data is not None and not data.empty:
                        # For intraday, convert to CT
                        if interval != '1d':
                            data.index = data.index - pd.Timedelta(hours=1)
                            # Filter to market hours starting at 6:30 AM CT until 7 PM CT
                            data = data[((data.index.hour > 6) | ((data.index.hour == 6) & (data.index.minute >= 30))) & (data.index.hour < 19)]

                        # Limit to last 100 bars for display
                        display_data = data.tail(100)

                        if not display_data.empty:
                            fig = create_stock_chart(ticker, display_data, result, interval)
                            # Don't force container width - let chart maintain proportions
                            st.plotly_chart(fig, use_container_width=False)
                        else:
                            st.warning(f"No data for {ticker}")
                    else:
                        st.warning(f"No data available for {ticker}")

    # All stocks table
    st.markdown("---")
    st.subheader("📋 All Analyzed Stocks")

    # Create DataFrame for display
    table_data = []
    for r in results:
        momentum = r.get('momentum', {})
        table_data.append({
            'Ticker': r['ticker'],
            'Score': f"{r.get('score', 0):.2f}",
            'Price': f"${r.get('current_price', 0):.2f}",
            '1D %': f"{momentum.get('1d', 0) or 0:+.2f}%",
            '5D %': f"{momentum.get('5d', 0) or 0:+.2f}%",
            'RSI': f"{r.get('current_rsi', 0):.1f}",
            'ADX': f"{r.get('current_adx', 0):.1f}",
            'Trending': '✅' if r.get('score', 0) >= 4.0 else '❌'
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
