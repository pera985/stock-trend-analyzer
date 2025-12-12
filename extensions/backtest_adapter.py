"""
Stock Trend Analyzer Backtest Adapter

Connects stock_trend_analyzer to shared_modules functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "stock_trend_analyzer"))

from shared_modules.base_adapter import BaseAnalyzerAdapter


class StockTrendBacktestAdapter(BaseAnalyzerAdapter):
    """
    Adapter for stock_trend_analyzer integration with shared_modules.

    Usage:
        from extensions import StockTrendBacktestAdapter

        adapter = StockTrendBacktestAdapter(
            api_key="your_polygon_api_key",
            storage_path="./output/backtest_data"
        )

        # Run walk-forward backtest
        results = adapter.run_backtest(
            tickers=['AAPL', 'MSFT', 'NVDA'],
            start_date='2022-01-01',
            end_date='2024-12-31'
        )

        print(results['report'])

        # Track a signal
        adapter.track_signal('AAPL', 'BUY', 150.00, score=5.2)

        # Check performance
        perf = adapter.check_performance(30)
        print(f"Current win rate: {perf['current_metrics']['win_rate']}%")
    """

    def __init__(
        self,
        api_key: str,
        storage_path: str = "./output/backtest_data"
    ):
        """
        Initialize the adapter.

        Args:
            api_key: Polygon.io API key
            storage_path: Path for storing backtest data
        """
        super().__init__(
            analyzer_name="stock_trend_analyzer",
            storage_path=storage_path,
            api_key=api_key
        )

        # Import and initialize the analyzer
        try:
            from stock_trend_analyzer import StockTrendAnalyzer
            self.analyzer = StockTrendAnalyzer(api_key=api_key)
        except ImportError:
            # Fallback: try direct import
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from stock_trend_analyzer import StockTrendAnalyzer
                self.analyzer = StockTrendAnalyzer(api_key=api_key)
            except ImportError as e:
                raise ImportError(
                    f"Could not import StockTrendAnalyzer. "
                    f"Ensure stock_trend_analyzer.py is in the correct location. Error: {e}"
                )

        # Store parameters (from config.py)
        self._parameters = self._load_default_parameters()

        # Initialize versioning with current parameters
        if not self.parameter_versioning.versions:
            self.parameter_versioning.initialize(
                self._parameters,
                "Initial stock_trend_analyzer parameters"
            )

    def _load_default_parameters(self) -> Dict[str, Any]:
        """Load default parameters from config."""
        try:
            from config import (
                TRENDING_THRESHOLD, SMA_20_PERIOD, SMA_50_PERIOD, SMA_200_PERIOD,
                RSI_PERIOD, RSI_FAVORABLE_MIN, RSI_FAVORABLE_MAX,
                ADX_PERIOD, ADX_STRONG_THRESHOLD,
                MACD_FAST, MACD_SLOW, MACD_SIGNAL
            )
            return {
                'trending_threshold': TRENDING_THRESHOLD,
                'sma_20_period': SMA_20_PERIOD,
                'sma_50_period': SMA_50_PERIOD,
                'sma_200_period': SMA_200_PERIOD,
                'rsi_period': RSI_PERIOD,
                'rsi_favorable_min': RSI_FAVORABLE_MIN,
                'rsi_favorable_max': RSI_FAVORABLE_MAX,
                'adx_period': ADX_PERIOD,
                'adx_strong_threshold': ADX_STRONG_THRESHOLD,
                'macd_fast': MACD_FAST,
                'macd_slow': MACD_SLOW,
                'macd_signal': MACD_SIGNAL
            }
        except ImportError:
            # Return sensible defaults if config not available
            return {
                'trending_threshold': 4.0,
                'sma_20_period': 20,
                'sma_50_period': 50,
                'sma_200_period': 200,
                'rsi_period': 14,
                'rsi_favorable_min': 50,
                'rsi_favorable_max': 70,
                'adx_period': 14,
                'adx_strong_threshold': 25,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }

    def generate_signal(
        self,
        ticker: str,
        date: datetime,
        price_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal using stock_trend_analyzer.

        Args:
            ticker: Stock ticker symbol
            date: Date for signal generation
            price_data: Historical price data

        Returns:
            Signal dictionary or None
        """
        try:
            # Use the analyzer's is_trending_up method
            result = self.analyzer.is_trending_up(ticker)

            if result is None:
                return None

            # Convert to standard signal format
            is_trending = result.get('is_trending', False)
            score = result.get('score', 0)

            return {
                'signal': 'BUY' if is_trending else 'HOLD',
                'score': score,
                'is_trending': is_trending,
                'ma_bullish': result.get('ma_bullish', False),
                'rsi': result.get('rsi', 0),
                'macd_bullish': result.get('macd_bullish', False),
                'adx': result.get('adx', 0),
                'volume_trend': result.get('volume_trend', 'Unknown')
            }

        except Exception as e:
            # Log error but don't crash
            print(f"Signal generation failed for {ticker}: {e}")
            return None

    def get_parameters(self) -> Dict[str, Any]:
        """Get current analyzer parameters."""
        return self._parameters.copy()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set analyzer parameters."""
        self._parameters.update(params)

        # Note: The actual analyzer uses config.py values
        # To apply parameters, you would need to modify config.py
        # or pass parameters to analyzer methods

    def optimize_parameters(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Optimize parameters on training data.

        This is a simple grid search implementation.
        For production, consider more sophisticated optimization.

        Args:
            tickers: List of ticker symbols
            price_data: Dictionary of price DataFrames
            start_date: Training period start
            end_date: Training period end

        Returns:
            Optimized parameters
        """
        # Simple optimization: test threshold variations
        best_params = self._parameters.copy()
        best_score = 0

        # Test different thresholds
        for threshold in [3.5, 4.0, 4.5, 5.0]:
            test_params = self._parameters.copy()
            test_params['trending_threshold'] = threshold

            # Simulate signals with this threshold
            wins = 0
            total = 0

            for ticker, df in price_data.items():
                if df is None or len(df) < 50:
                    continue

                # Simple backtest simulation
                for i in range(50, len(df) - 20):
                    current_date = df.iloc[i].get('date', start_date)
                    if isinstance(current_date, str):
                        current_date = datetime.fromisoformat(current_date)

                    if current_date < start_date or current_date > end_date:
                        continue

                    # Generate signal with threshold
                    # (Simplified - in production, apply threshold properly)
                    entry_price = df.iloc[i]['close']
                    exit_price = df.iloc[min(i + 20, len(df) - 1)]['close']

                    if exit_price > entry_price:
                        wins += 1
                    total += 1

            if total > 0:
                score = wins / total
                if score > best_score:
                    best_score = score
                    best_params = test_params

        return best_params

    def fetch_price_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch price data using the analyzer's API client.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate days needed
            days = (end_date - start_date).days + 50  # Extra for MAs

            # Use analyzer's client
            df = self.analyzer.client.get_daily_data(ticker, outputsize='full')

            if df is not None and not df.empty:
                # Ensure date column exists and filter
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            return df

        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()


# Convenience function
def create_adapter(api_key: str, storage_path: str = "./output/backtest_data"):
    """Create a StockTrendBacktestAdapter instance."""
    return StockTrendBacktestAdapter(api_key=api_key, storage_path=storage_path)
