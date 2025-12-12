"""
Streamlit Live Dashboard for Stock Trend Analyzer
Web-based version for sharing via Streamlit Cloud

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

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_api_key():
    """Get API key from Streamlit secrets or environment"""
    # Try Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and 'MASSIVE_API_KEY' in st.secrets:
        return st.secrets['MASSIVE_API_KEY']
    # Fall back to environment variable
    return os.getenv('MASSIVE_API_KEY')


def load_tickers_from_file(filename):
    """Load tickers from input file"""
    input_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_files')
    filepath = os.path.join(input_files_dir, filename)

    if not os.path.exists(filepath):
        return []

    tickers = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().upper()
            if line and not line.startswith('#'):
                tickers.append(line)
    return tickers


def get_available_input_files():
    """Get list of available input files"""
    input_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_files')
    if not os.path.exists(input_files_dir):
        return []
    return [f for f in os.listdir(input_files_dir) if f.endswith('.txt')]


def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    return {
        'middle': sma,
        'upper': sma + (std * num_std),
        'lower': sma - (std * num_std)
    }


def get_rsi_color(rsi_value):
    """Get color based on RSI value"""
    if pd.isna(rsi_value):
        return '#808080'
    if rsi_value < 30 or rsi_value > 80:
        return '#FFD700'  # Yellow - extreme
    elif rsi_value < 50:
        return '#FF4444'  # Red - bearish
    else:
        return '#4169E1'  # Blue - bullish


def create_stock_chart(ticker, data, result, interval):
    """Create a 4-panel Plotly chart for a single stock"""

    # Calculate indicators
    bb = calculate_bollinger_bands(data)
    rsi = calculate_rsi(data)
    sma_5 = data['Close'].rolling(window=5).mean()
    sma_20 = data['Close'].rolling(window=20).mean()
    volume_ma_50 = data['Volume'].rolling(window=50).mean()

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=('Price & Bollinger Bands', 'Volume', 'RSI', 'Momentum')
    )

    # Panel 1: Candlestick with Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=bb['upper'], mode='lines',
                   name='BB Upper', line=dict(color='rgba(128,128,128,0.5)', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=bb['middle'], mode='lines',
                   name='BB Middle', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=bb['lower'], mode='lines',
                   name='BB Lower', line=dict(color='rgba(128,128,128,0.5)', width=1),
                   fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )

    # Panel 2: Volume
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=volume_ma_50, mode='lines',
                   name='Vol MA50', line=dict(color='purple', width=1)),
        row=2, col=1
    )

    # SMA on volume panel (secondary y-axis would be complex, skip for simplicity)

    # Panel 3: RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI',
                   line=dict(color='blue', width=1.5)),
        row=3, col=1
    )
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    # Panel 4: Price SMAs
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_5, mode='lines',
                   name='SMA5', line=dict(color='orange', width=1)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_20, mode='lines',
                   name='SMA20', line=dict(color='blue', width=1)),
        row=4, col=1
    )

    # Get current price and score
    current_price = result.get('current_price', data['Close'].iloc[-1]) if result else data['Close'].iloc[-1]
    score = result.get('score', 0) if result else 0
    momentum = result.get('momentum', {}) if result else {}
    gain_1d = momentum.get('1d', 0)

    # Update layout
    fig.update_layout(
        title=f"{ticker} | Score: {score:.1f} | {gain_1d:+.2f}% | ${current_price:.2f}",
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=30)
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="SMA", row=4, col=1)

    return fig


def create_heatmap(results):
    """Create a heatmap showing all stocks by score and daily gain"""
    if not results:
        return None

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    tickers = [r['ticker'] for r in sorted_results]
    scores = [r.get('score', 0) for r in sorted_results]
    gains = [r.get('momentum', {}).get('1d', 0) for r in sorted_results]

    # Create figure
    fig = go.Figure()

    # Add horizontal bars for scores
    colors = ['green' if g >= 0 else 'red' for g in gains]

    fig.add_trace(go.Bar(
        y=tickers,
        x=scores,
        orientation='h',
        marker_color=colors,
        marker_opacity=[0.3 + min(abs(g) / 10, 0.7) for g in gains],
        text=[f"{s:.1f} | {g:+.1f}%" for s, g in zip(scores, gains)],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))

    fig.update_layout(
        title="Stock Rankings (Score | Daily Gain %)",
        xaxis_title="Score",
        yaxis_title="",
        height=max(300, len(tickers) * 25),
        margin=dict(l=80, r=20, t=40, b=40),
        xaxis=dict(range=[0, 6.5])
    )

    # Reverse y-axis so highest score is at top
    fig.update_yaxes(autorange="reversed")

    return fig


def main():
    """Main Streamlit app"""

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

    # Or manual ticker input
    manual_tickers = st.sidebar.text_input(
        "Or enter tickers (comma-separated)",
        placeholder="AAPL,MSFT,GOOGL"
    )

    # Interval selection
    interval = st.sidebar.selectbox(
        "Interval",
        ['1d', '5min', '15min', '30min', '60min'],
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

    # Number of charts to show
    num_charts = st.sidebar.slider(
        "Number of detailed charts",
        min_value=1,
        max_value=12,
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

    # Load tickers
    if manual_tickers:
        tickers = [t.strip().upper() for t in manual_tickers.split(',') if t.strip()]
    elif selected_file:
        tickers = load_tickers_from_file(selected_file)
    else:
        tickers = []

    if not tickers:
        st.warning("No tickers to analyze. Select an input file or enter tickers manually.")
        st.stop()

    st.sidebar.markdown(f"**Analyzing:** {len(tickers)} tickers")

    # Main content
    st.title("📊 Live Stock Dashboard")
    st.markdown(f"*Updated: {current_time}*")

    # Initialize analyzer
    with st.spinner("Analyzing stocks..."):
        analyzer = StockTrendAnalyzer(
            api_key=api_key,
            interval=interval,
            period='full'
        )

        # Analyze all tickers
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

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    trending_count = sum(1 for r in results if r.get('score', 0) >= 4.0)
    avg_score = sum(r.get('score', 0) for r in results) / len(results)
    avg_gain = sum(r.get('momentum', {}).get('1d', 0) for r in results) / len(results)
    top_ticker = results[0]['ticker'] if results else "N/A"

    with col1:
        st.metric("Trending Stocks", f"{trending_count}/{len(results)}",
                  help="Stocks with score >= 4.0")
    with col2:
        st.metric("Avg Score", f"{avg_score:.2f}")
    with col3:
        st.metric("Avg Daily Gain", f"{avg_gain:+.2f}%",
                  delta_color="normal" if avg_gain >= 0 else "inverse")
    with col4:
        st.metric("Top Stock", top_ticker,
                  f"{results[0].get('score', 0):.1f}" if results else "")

    st.markdown("---")

    # Two-column layout: Heatmap and Top Charts
    col_heatmap, col_charts = st.columns([1, 2])

    with col_heatmap:
        st.subheader("📊 All Stocks Ranking")
        heatmap = create_heatmap(results)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

    with col_charts:
        st.subheader(f"📈 Top {num_charts} Stocks")

        # Get top N results
        top_results = results[:num_charts]

        # Create tabs for each stock
        if top_results:
            tabs = st.tabs([f"{r['ticker']} ({r.get('score', 0):.1f})" for r in top_results])

            for tab, result in zip(tabs, top_results):
                with tab:
                    ticker = result['ticker']

                    # Fetch data for chart
                    client = MassiveClient(api_key, RateLimiter())

                    if interval == '1d':
                        data = client.get_daily_data(ticker, 'compact')
                    else:
                        data = client.get_intraday_data(ticker, interval, 'compact')

                    if data is not None and not data.empty:
                        # Limit to last 100 bars for display
                        display_data = data.tail(100)
                        fig = create_stock_chart(ticker, display_data, result, interval)
                        st.plotly_chart(fig, use_container_width=True)

                        # Stock details
                        with st.expander("📋 Details"):
                            dcol1, dcol2, dcol3 = st.columns(3)
                            with dcol1:
                                st.write(f"**Score:** {result.get('score', 0):.2f}")
                                st.write(f"**Current Price:** ${result.get('current_price', 0):.2f}")
                            with dcol2:
                                momentum = result.get('momentum', {})
                                st.write(f"**1D Gain:** {momentum.get('1d', 0):+.2f}%")
                                st.write(f"**5D Gain:** {momentum.get('5d', 0):+.2f}%")
                            with dcol3:
                                st.write(f"**RSI:** {result.get('current_rsi', 0):.1f}")
                                st.write(f"**ADX:** {result.get('current_adx', 0):.1f}")
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
            '1D %': f"{momentum.get('1d', 0):+.2f}%",
            '5D %': f"{momentum.get('5d', 0):+.2f}%",
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
