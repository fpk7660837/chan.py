"""
BacktestConfig - 回测配置类

定义回测系统的所有配置参数，包括：
- 资金设置
- 交易成本
- 仓位控制
- 时间设置
- 数据源配置
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from Common.CEnum import DATA_SRC, KL_TYPE


@dataclass
class CBacktestConfig:
    """回测配置类"""

    # ============ 资金设置 ============
    initial_capital: float = 100000.0  # 初始资金

    # ============ 交易成本 ============
    commission_rate: float = 0.0003    # 手续费率（万三）
    slippage_rate: float = 0.001       # 滑点率（0.1%）
    stamp_tax_rate: float = 0.001      # 印花税率（千一，仅卖出）
    min_commission: float = 5.0        # 最小手续费（元）

    # ============ 仓位控制 ============
    max_position_per_stock: float = 0.3   # 单只股票最大仓位比例
    max_total_position: float = 0.95      # 总仓位上限
    min_trade_amount: float = 100.0       # 最小交易金额（元）

    # ============ 时间设置 ============
    begin_time: str = "2020-01-01"        # 回测开始时间
    end_time: Optional[str] = None        # 回测结束时间（None表示至今）

    # ============ 数据源设置 ============
    data_src: DATA_SRC = DATA_SRC.AKSHARE  # 数据源
    lv_list: list = field(default_factory=lambda: [KL_TYPE.K_DAY])  # K线级别列表
    autype: str = "qfq"                    # 复权类型：qfq/hfq/none

    # ============ 缠论配置 ============
    chan_config: Dict = field(default_factory=dict)  # 缠论配置参数

    # ============ 其他设置 ============
    benchmark: Optional[str] = None        # 基准代码（如"sh000001"上证指数）
    match_mode: str = "next_open"          # 成交方式：
                                           # - next_open: 下一根K线开盘价成交
                                           # - next_close: 下一根K线收盘价成交
                                           # - current_close: 当前K线收盘价成交（不推荐，存在未来函数）

    # ============ 回测控制 ============
    print_progress: bool = True            # 是否打印回测进度
    progress_interval: int = 100           # 进度打印间隔（每N根K线）

    def __post_init__(self):
        """初始化后的验证"""
        # 确保trigger_step=True用于回测
        if 'trigger_step' not in self.chan_config:
            self.chan_config['trigger_step'] = True

        # 验证配置合理性
        assert 0 <= self.max_position_per_stock <= 1, "单只股票仓位应在0-1之间"
        assert 0 <= self.max_total_position <= 1, "总仓位应在0-1之间"
        assert self.initial_capital > 0, "初始资金必须大于0"
        assert self.commission_rate >= 0, "手续费率不能为负"
        assert self.slippage_rate >= 0, "滑点率不能为负"

    def calculate_commission(self, price: float, volume: int) -> float:
        """
        计算手续费

        Args:
            price: 成交价格
            volume: 成交数量（股）

        Returns:
            手续费金额
        """
        commission = price * volume * self.commission_rate
        return max(commission, self.min_commission)

    def calculate_slippage(self, price: float, direction: str) -> float:
        """
        计算滑点后的价格

        Args:
            price: 原始价格
            direction: 交易方向 "buy"/"sell"

        Returns:
            滑点后的价格
        """
        if direction == 'buy':
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)

    def calculate_stamp_tax(self, price: float, volume: int) -> float:
        """
        计算印花税（仅卖出时收取）

        Args:
            price: 成交价格
            volume: 成交数量（股）

        Returns:
            印花税金额
        """
        return price * volume * self.stamp_tax_rate

    def calculate_total_cost(self, price: float, volume: int, direction: str) -> tuple:
        """
        计算总交易成本

        Args:
            price: 成交价格
            volume: 成交数量（股）
            direction: 交易方向 "buy"/"sell"

        Returns:
            (实际成交价, 手续费, 印花税, 总成本)
        """
        # 计算滑点后价格
        actual_price = self.calculate_slippage(price, direction)

        # 计算手续费
        commission = self.calculate_commission(actual_price, volume)

        # 计算印花税（仅卖出）
        tax = self.calculate_stamp_tax(actual_price, volume) if direction == 'sell' else 0.0

        # 总成本
        total_cost = commission + tax

        return actual_price, commission, tax, total_cost
