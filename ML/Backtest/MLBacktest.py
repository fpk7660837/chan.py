"""
机器学习回测引擎

基于ML预测分数进行回测，评估模型在实际交易中的表现
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from Chan import CChan
from ..Prediction.Predictor import Predictor
from ..Evaluation.Metrics import Metrics


class MLBacktest:
    """机器学习回测引擎"""

    def __init__(self, predictor: Predictor, config: Dict[str, Any] = None):
        """
        初始化回测引擎

        Args:
            predictor: 预测器
            config: 回测配置字典，包含：
                - score_threshold: 交易信号阈值
                - holding_period: 持仓周期
                - initial_capital: 初始资金
                - commission_rate: 手续费率
                - slippage: 滑点
                - max_position: 最大仓位
        """
        self.predictor = predictor
        self.config = config or {}

        self.score_threshold = self.config.get('score_threshold', 0.7)
        self.holding_period = self.config.get('holding_period', 20)
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.commission_rate = self.config.get('commission_rate', 0.0003)
        self.slippage = self.config.get('slippage', 0.001)
        self.max_position = self.config.get('max_position', 1.0)

    def run(self, chan_list: List[CChan]) -> Dict[str, Any]:
        """
        运行回测

        Args:
            chan_list: CChan实例列表（测试数据）

        Returns:
            回测结果字典
        """
        print("Starting backtest...")

        # 收集所有交易信号
        all_trades = []
        all_returns = []

        for i, chan in enumerate(chan_list):
            print(f"Processing CChan {i+1}/{len(chan_list)}...")

            # 获取买卖点并预测
            bsp_signals = self.predictor.filter_bsp_by_threshold(
                chan,
                threshold=self.score_threshold,
                direction='all'
            )

            # 模拟交易
            trades = self._simulate_trades(bsp_signals, chan)
            all_trades.extend(trades)

            # 计算收益
            for trade in trades:
                all_returns.append(trade['return'])

        # 计算指标
        returns_array = np.array(all_returns) if len(all_returns) > 0 else np.array([0])

        metrics = Metrics.calculate_trading_metrics(returns_array, all_trades)

        # 添加额外信息
        metrics['total_signals'] = len(all_trades)

        print(f"\nBacktest completed!")
        print(f"Total signals: {metrics['total_signals']}")

        # 打印结果
        Metrics.print_metrics(metrics, title="Backtest Results")

        return metrics

    def _simulate_trades(self, bsp_signals: List[Tuple], chan: CChan) -> List[Dict[str, Any]]:
        """
        模拟交易

        Args:
            bsp_signals: [(买卖点, 分数), ...]
            chan: CChan实例

        Returns:
            交易记录列表
        """
        trades = []

        for bsp, score in bsp_signals:
            trade = self._execute_trade(bsp, score)
            if trade is not None:
                trades.append(trade)

        return trades

    def _execute_trade(self, bsp, score: float) -> Dict[str, Any]:
        """
        执行单笔交易

        Args:
            bsp: 买卖点
            score: 预测分数

        Returns:
            交易记录字典
        """
        if bsp.klu is None:
            return None

        # 入场价格（加上滑点）
        entry_price = bsp.klu.close * (1 + self.slippage if bsp.is_buy else 1 - self.slippage)

        # 获取未来价格（持仓周期）
        future_prices = self._get_future_prices(bsp.klu, self.holding_period)

        if len(future_prices) == 0:
            return None

        # 出场价格
        if bsp.is_buy:
            # 买点：持仓期间的最高价
            exit_price = max(future_prices) * (1 - self.slippage)
        else:
            # 卖点（做空）：持仓期间的最低价
            exit_price = min(future_prices) * (1 + self.slippage)

        # 计算收益率
        if bsp.is_buy:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            # 做空收益
            gross_return = (entry_price - exit_price) / entry_price

        # 扣除手续费（买入和卖出各一次）
        net_return = gross_return - 2 * self.commission_rate

        # 计算盈亏
        position_size = self.initial_capital * self.max_position
        profit = position_size * net_return

        # 记录交易
        trade = {
            'entry_time': bsp.klu.time if hasattr(bsp.klu, 'time') else None,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': 'buy' if bsp.is_buy else 'sell',
            'score': score,
            'return': net_return,
            'profit': profit,
            'holding_period': len(future_prices),
        }

        return trade

    def _get_future_prices(self, klu, periods: int) -> List[float]:
        """
        获取未来N个周期的价格

        Args:
            klu: 当前K线
            periods: 周期数

        Returns:
            价格列表
        """
        prices = []
        current = klu

        for _ in range(periods):
            if hasattr(current, 'next') and current.next is not None:
                current = current.next
                prices.append(current.close)
            else:
                break

        return prices

    def run_with_validation(self, train_chan_list: List[CChan], test_chan_list: List[CChan]) -> Dict[str, Any]:
        """
        运行带验证的回测（分别在训练集和测试集上回测）

        Args:
            train_chan_list: 训练集CChan列表
            test_chan_list: 测试集CChan列表

        Returns:
            包含训练集和测试集指标的字典
        """
        print("Running backtest on training set...")
        train_metrics = self.run(train_chan_list)

        print("\nRunning backtest on test set...")
        test_metrics = self.run(test_chan_list)

        return {
            'train': train_metrics,
            'test': test_metrics,
        }
