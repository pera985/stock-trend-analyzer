"""
Stock Trend Analyzer Extensions

Provides integration with shared_modules for:
- Walk-forward backtesting
- Performance monitoring
- Parameter versioning
- Market regime detection
"""

from .backtest_adapter import StockTrendBacktestAdapter

__all__ = ['StockTrendBacktestAdapter']
