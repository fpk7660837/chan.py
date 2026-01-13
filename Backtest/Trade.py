"""
Trade - 交易记录类

记录每笔交易的详细信息
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from Common.CTime import CTime


@dataclass
class CTrade:
    """单笔交易记录"""

    # 基本信息
    code: str                  # 股票代码
    direction: str             # 交易方向：buy/sell
    volume: int                # 成交数量（股）
    price: float               # 成交价格

    # 时间信息
    time: CTime                # 交易时间（K线时间）
    datetime: datetime         # 交易发生时间戳

    # 成本信息
    commission: float = 0.0    # 手续费
    tax: float = 0.0           # 印花税
    slippage: float = 0.0      # 滑点成本

    # 交易原因
    reason: str = ""           # 交易原因/信号描述

    # 关联信息
    position_change: float = 0.0  # 仓位变化（金额）
    cash_change: float = 0.0      # 资金变化

    # 盈亏信息（仅卖出时有效）
    profit: Optional[float] = None       # 本次交易盈亏
    profit_rate: Optional[float] = None  # 本次交易收益率

    def __post_init__(self):
        """计算总成本"""
        self.total_cost = self.commission + self.tax + self.slippage

        # 计算资金变化
        if self.direction == 'buy':
            # 买入：资金减少
            self.cash_change = -(self.price * self.volume + self.total_cost)
            self.position_change = self.price * self.volume
        else:
            # 卖出：资金增加
            self.cash_change = self.price * self.volume - self.total_cost
            self.position_change = -(self.price * self.volume)

    def get_total_amount(self) -> float:
        """获取交易总金额（不含成本）"""
        return self.price * self.volume

    def get_net_amount(self) -> float:
        """获取交易净金额（含成本）"""
        if self.direction == 'buy':
            return -(self.price * self.volume + self.total_cost)
        else:
            return self.price * self.volume - self.total_cost

    def __str__(self) -> str:
        """字符串表示"""
        direction_str = "买入" if self.direction == "buy" else "卖出"
        profit_str = ""
        if self.profit is not None:
            profit_str = f", 盈亏: {self.profit:.2f} ({self.profit_rate*100:.2f}%)"

        return (f"{self.time} {direction_str} {self.code} "
                f"{self.volume}股 @ {self.price:.2f}, "
                f"成本: {self.total_cost:.2f}{profit_str}, "
                f"原因: {self.reason}")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'code': self.code,
            'direction': self.direction,
            'volume': self.volume,
            'price': self.price,
            'time': str(self.time),
            'datetime': self.datetime.isoformat(),
            'commission': self.commission,
            'tax': self.tax,
            'slippage': self.slippage,
            'total_cost': self.total_cost,
            'reason': self.reason,
            'cash_change': self.cash_change,
            'position_change': self.position_change,
            'profit': self.profit,
            'profit_rate': self.profit_rate,
        }
