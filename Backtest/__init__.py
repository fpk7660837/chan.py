"""
Backtest Module - 回测模块

提供完整的缠论策略回测功能，包括：
- 回测引擎（BacktestEngine）
- 策略基类（Strategy）
- 持仓管理（Position）
- 绩效分析（Performance）
- 可视化（Visualizer）
"""

from .BacktestConfig import CBacktestConfig
from .Strategy import CStrategy, CSignal
from .Position import CPosition, CPositionManager
from .Trade import CTrade
from .BacktestEngine import CBacktestEngine
from .Performance import CPerformance

__all__ = [
    'CBacktestConfig',
    'CStrategy',
    'CSignal',
    'CPosition',
    'CPositionManager',
    'CTrade',
    'CBacktestEngine',
    'CPerformance',
]
