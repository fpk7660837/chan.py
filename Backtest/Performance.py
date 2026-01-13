"""
Performance - 绩效分析模块

计算回测的各项绩效指标
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
from Backtest.BacktestConfig import CBacktestConfig


class CPerformance:
    """绩效分析器"""

    def __init__(self, result, config: CBacktestConfig):
        """
        Args:
            result: CBacktestResult对象
            config: 回测配置
        """
        self.result = result
        self.config = config

    def calculate_metrics(self) -> Dict:
        """
        计算所有绩效指标

        Returns:
            包含所有指标的字典
        """
        metrics = {}

        # 基础数据
        equity_curve = self.result.equity_curve
        trades = self.result.trades
        initial_capital = self.result.initial_capital

        if not equity_curve:
            return metrics

        # 最终资产
        final_value = equity_curve[-1][1] if equity_curve else initial_capital
        metrics['final_value'] = final_value

        # 总盈亏
        total_profit = final_value - initial_capital
        metrics['total_profit'] = total_profit

        # 累计收益率
        total_return = total_profit / initial_capital if initial_capital > 0 else 0.0
        metrics['total_return'] = total_return

        # 年化收益率
        annual_return = self._calculate_annual_return(equity_curve, initial_capital)
        metrics['annual_return'] = annual_return

        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_drawdown

        # 夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        metrics['sharpe_ratio'] = sharpe_ratio

        # 波动率
        volatility = self._calculate_volatility(equity_curve)
        metrics['volatility'] = volatility

        # 交易统计
        trade_metrics = self._calculate_trade_metrics(trades)
        metrics.update(trade_metrics)

        return metrics

    def _calculate_annual_return(self, equity_curve: List, initial_capital: float) -> float:
        """计算年化收益率"""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        # 计算总天数
        start_time = equity_curve[0][0]
        end_time = equity_curve[-1][0]

        start_dt = datetime(start_time.year, start_time.month, start_time.day)
        end_dt = datetime(end_time.year, end_time.month, end_time.day)
        total_days = (end_dt - start_dt).days

        if total_days == 0:
            return 0.0

        # 计算总收益率
        final_value = equity_curve[-1][1]
        total_return = (final_value - initial_capital) / initial_capital

        # 年化收益率 = (1 + 总收益率) ^ (365 / 总天数) - 1
        years = total_days / 365.0
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0.0

        return annual_return

    def _calculate_max_drawdown(self, equity_curve: List) -> float:
        """计算最大回撤"""
        if not equity_curve:
            return 0.0

        # 提取资产曲线
        values = [point[1] for point in equity_curve]

        max_dd = 0.0
        peak = values[0]

        for value in values:
            if value > peak:
                peak = value

            dd = (peak - value) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self, equity_curve: List, risk_free_rate: float = 0.03) -> float:
        """
        计算夏普比率

        Args:
            equity_curve: 权益曲线
            risk_free_rate: 无风险利率（年化）

        Returns:
            夏普比率
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        # 计算日收益率
        returns = []
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1][1]
            curr_value = equity_curve[i][1]
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        if not returns:
            return 0.0

        # 日无风险利率
        daily_rf = risk_free_rate / 252

        # 超额收益
        excess_returns = [r - daily_rf for r in returns]

        # 夏普比率 = 年化超额收益 / 年化波动率
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0:
            return 0.0

        # 年化
        sharpe = (mean_excess / std_excess) * np.sqrt(252)

        return sharpe

    def _calculate_volatility(self, equity_curve: List) -> float:
        """计算波动率（年化）"""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        # 计算日收益率
        returns = []
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1][1]
            curr_value = equity_curve[i][1]
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        if not returns:
            return 0.0

        # 年化波动率
        volatility = np.std(returns) * np.sqrt(252)

        return volatility

    def _calculate_trade_metrics(self, trades: List) -> Dict:
        """计算交易统计指标"""
        metrics = {}

        # 筛选出卖出交易（只有卖出才有盈亏）
        sell_trades = [t for t in trades if t.direction == 'sell' and t.profit is not None]

        # 总交易次数（买卖配对）
        metrics['trade_count'] = len(sell_trades)

        if not sell_trades:
            metrics['win_rate'] = 0.0
            metrics['profit_loss_ratio'] = 0.0
            metrics['avg_profit'] = 0.0
            metrics['avg_loss'] = 0.0
            metrics['avg_hold_days'] = 0.0
            return metrics

        # 盈利和亏损交易
        win_trades = [t for t in sell_trades if t.profit > 0]
        loss_trades = [t for t in sell_trades if t.profit <= 0]

        # 胜率
        metrics['win_rate'] = len(win_trades) / len(sell_trades) if sell_trades else 0.0

        # 平均盈利和平均亏损
        avg_profit = np.mean([t.profit for t in win_trades]) if win_trades else 0.0
        avg_loss = abs(np.mean([t.profit for t in loss_trades])) if loss_trades else 0.0

        metrics['avg_profit'] = avg_profit
        metrics['avg_loss'] = avg_loss

        # 盈亏比
        metrics['profit_loss_ratio'] = avg_profit / avg_loss if avg_loss > 0 else 0.0

        # 平均持仓天数（通过买卖配对计算）
        hold_days_list = self._calculate_hold_days(trades)
        metrics['avg_hold_days'] = np.mean(hold_days_list) if hold_days_list else 0.0

        # 最大单笔盈利和亏损
        metrics['max_profit'] = max([t.profit for t in win_trades]) if win_trades else 0.0
        metrics['max_loss'] = min([t.profit for t in loss_trades]) if loss_trades else 0.0

        # 总盈利和总亏损
        metrics['total_profit_trades'] = sum([t.profit for t in win_trades])
        metrics['total_loss_trades'] = sum([t.profit for t in loss_trades])

        return metrics

    def _calculate_hold_days(self, trades: List) -> List[int]:
        """计算每笔交易的持仓天数"""
        hold_days = []

        # 按股票分组
        code_trades = {}
        for trade in trades:
            if trade.code not in code_trades:
                code_trades[trade.code] = []
            code_trades[trade.code].append(trade)

        # 对每个股票的交易进行配对
        for code, code_trade_list in code_trades.items():
            # 按时间排序
            code_trade_list.sort(key=lambda t: t.datetime)

            # 简单的买卖配对（FIFO）
            buy_queue = []
            for trade in code_trade_list:
                if trade.direction == 'buy':
                    buy_queue.append(trade)
                elif trade.direction == 'sell':
                    # 从买入队列中取出对应数量
                    remaining = trade.volume
                    while remaining > 0 and buy_queue:
                        buy_trade = buy_queue[0]
                        matched = min(remaining, buy_trade.volume)

                        # 计算持仓天数
                        buy_dt = datetime(buy_trade.time.year, buy_trade.time.month, buy_trade.time.day)
                        sell_dt = datetime(trade.time.year, trade.time.month, trade.time.day)
                        days = (sell_dt - buy_dt).days
                        hold_days.append(days)

                        # 更新剩余数量
                        remaining -= matched
                        buy_trade.volume -= matched

                        # 如果买入交易已全部卖出，从队列中移除
                        if buy_trade.volume == 0:
                            buy_queue.pop(0)

        return hold_days

    def plot_equity_curve(self, save_path: str = None):
        """绘制权益曲线"""
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # 设置中文字体
        rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        rcParams['axes.unicode_minus'] = False

        equity_curve = self.result.equity_curve
        if not equity_curve:
            print("没有权益曲线数据")
            return

        # 提取数据
        times = [point[0] for point in equity_curve]
        values = [point[1] for point in equity_curve]
        dates = [f"{t.year}-{t.month:02d}-{t.day:02d}" for t in times]

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制权益曲线
        ax.plot(dates, values, linewidth=2, label='总资产')
        ax.axhline(y=self.result.initial_capital, color='r', linestyle='--', linewidth=1, label='初始资金')

        # 设置标题和标签
        ax.set_title(f'权益曲线 - {self.result.strategy_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('资产（元）', fontsize=12)

        # 设置x轴刻度（只显示部分日期）
        step = max(1, len(dates) // 10)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"权益曲线已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_drawdown(self, save_path: str = None):
        """绘制回撤曲线"""
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        rcParams['axes.unicode_minus'] = False

        equity_curve = self.result.equity_curve
        if not equity_curve:
            print("没有权益曲线数据")
            return

        # 计算回撤曲线
        times = [point[0] for point in equity_curve]
        values = [point[1] for point in equity_curve]
        dates = [f"{t.year}-{t.month:02d}-{t.day:02d}" for t in times]

        drawdowns = []
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            drawdowns.append(dd * 100)  # 转换为百分比

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制回撤曲线
        ax.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
        ax.plot(drawdowns, linewidth=2, color='red', label='回撤')

        # 设置标题和标签
        ax.set_title(f'回撤曲线 - {self.result.strategy_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('回撤 (%)', fontsize=12)

        # 设置x轴刻度
        step = max(1, len(dates) // 10)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"回撤曲线已保存到: {save_path}")
        else:
            plt.show()

        plt.close()
