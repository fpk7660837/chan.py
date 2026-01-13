"""
Position - 持仓管理类

管理回测过程中的持仓信息和资金
"""

from typing import Dict, Optional
from Common.CTime import CTime


class CPosition:
    """单个股票持仓"""

    def __init__(self, code: str):
        self.code = code                # 股票代码
        self.volume: int = 0            # 持仓数量（股）
        self.available: int = 0         # 可用数量（T+1制度）
        self.cost_price: float = 0.0    # 成本价
        self.current_price: float = 0.0 # 当前价格
        self.market_value: float = 0.0  # 市值
        self.cost_value: float = 0.0    # 成本总额
        self.profit: float = 0.0        # 浮动盈亏
        self.profit_rate: float = 0.0   # 盈亏比例
        self.hold_days: int = 0         # 持仓天数
        self.first_buy_time: Optional[CTime] = None  # 首次买入时间

    def update_price(self, price: float, current_time: Optional[CTime] = None):
        """
        更新当前价格和市值

        Args:
            price: 当前价格
            current_time: 当前时间（用于计算持仓天数）
        """
        self.current_price = price
        self.market_value = self.volume * price

        if self.volume > 0:
            self.profit = self.market_value - self.cost_value
            self.profit_rate = self.profit / self.cost_value if self.cost_value > 0 else 0.0

            # 更新持仓天数
            if current_time and self.first_buy_time:
                self.hold_days = self._calculate_days(self.first_buy_time, current_time)

    def add(self, volume: int, price: float, commission: float, buy_time: CTime):
        """
        增加持仓

        Args:
            volume: 买入数量
            price: 买入价格
            commission: 手续费
            buy_time: 买入时间
        """
        # 计算新的成本价（加权平均）
        total_cost = self.cost_value + (price * volume + commission)
        total_volume = self.volume + volume

        self.cost_price = total_cost / total_volume if total_volume > 0 else 0.0
        self.cost_value = total_cost
        self.volume = total_volume

        # 记录首次买入时间
        if self.first_buy_time is None:
            self.first_buy_time = buy_time

        # 更新市值
        self.market_value = self.volume * price

    def reduce(self, volume: int, price: float, commission: float, tax: float) -> tuple:
        """
        减少持仓

        Args:
            volume: 卖出数量
            price: 卖出价格
            commission: 手续费
            tax: 印花税

        Returns:
            (实现盈亏, 收益率)
        """
        if volume > self.available:
            raise ValueError(f"可用数量不足：需要{volume}，可用{self.available}")

        # 计算实现盈亏
        cost = self.cost_price * volume
        revenue = price * volume - commission - tax
        realized_profit = revenue - cost
        realized_profit_rate = realized_profit / cost if cost > 0 else 0.0

        # 更新持仓
        self.volume -= volume
        self.available -= volume
        self.cost_value = self.cost_price * self.volume

        # 清仓时重置首次买入时间
        if self.volume == 0:
            self.first_buy_time = None
            self.hold_days = 0
            self.cost_price = 0.0
            self.cost_value = 0.0

        # 更新市值
        self.market_value = self.volume * price

        return realized_profit, realized_profit_rate

    def update_available(self):
        """更新可用数量（T+1：当天买入的第二天才能卖出）"""
        self.available = self.volume

    def _calculate_days(self, start: CTime, end: CTime) -> int:
        """计算天数差"""
        from datetime import datetime
        start_dt = datetime(start.year, start.month, start.day)
        end_dt = datetime(end.year, end.month, end.day)
        return (end_dt - start_dt).days

    def __str__(self) -> str:
        """字符串表示"""
        if self.volume == 0:
            return f"{self.code}: 空仓"
        return (f"{self.code}: {self.volume}股 @ {self.current_price:.2f}, "
                f"成本价: {self.cost_price:.2f}, "
                f"市值: {self.market_value:.2f}, "
                f"盈亏: {self.profit:.2f} ({self.profit_rate*100:.2f}%), "
                f"持仓{self.hold_days}天")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'code': self.code,
            'volume': self.volume,
            'available': self.available,
            'cost_price': self.cost_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'cost_value': self.cost_value,
            'profit': self.profit,
            'profit_rate': self.profit_rate,
            'hold_days': self.hold_days,
        }


class CPositionManager:
    """持仓管理器"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital  # 初始资金
        self.cash = initial_capital             # 可用资金
        self.positions: Dict[str, CPosition] = {}  # 持仓字典
        self.frozen_cash = 0.0                  # 冻结资金

    def get_position(self, code: str) -> CPosition:
        """获取持仓，不存在则创建"""
        if code not in self.positions:
            self.positions[code] = CPosition(code)
        return self.positions[code]

    def has_position(self, code: str) -> bool:
        """是否持有某只股票"""
        return code in self.positions and self.positions[code].volume > 0

    def update_all_prices(self, prices: Dict[str, float], current_time: Optional[CTime] = None):
        """
        更新所有持仓的价格

        Args:
            prices: {code: price} 价格字典
            current_time: 当前时间
        """
        for code, position in self.positions.items():
            if position.volume > 0 and code in prices:
                position.update_price(prices[code], current_time)

    def update_available(self):
        """更新所有持仓的可用数量（每日收盘后调用）"""
        for position in self.positions.values():
            position.update_available()

    def get_total_value(self) -> float:
        """获取总资产"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    def get_positions_value(self) -> float:
        """获取持仓总市值"""
        return sum(p.market_value for p in self.positions.values())

    def get_position_ratio(self) -> float:
        """获取仓位比例"""
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
        return self.get_positions_value() / total_value

    def get_total_profit(self) -> float:
        """获取总浮动盈亏"""
        return sum(p.profit for p in self.positions.values() if p.volume > 0)

    def get_total_profit_rate(self) -> float:
        """获取总收益率"""
        if self.initial_capital == 0:
            return 0.0
        return (self.get_total_value() - self.initial_capital) / self.initial_capital

    def can_buy(self, code: str, price: float, volume: int, cost: float) -> bool:
        """
        检查是否可以买入

        Args:
            code: 股票代码
            price: 价格
            volume: 数量
            cost: 总成本（含手续费等）

        Returns:
            是否可以买入
        """
        # 检查资金是否足够
        total_cost = price * volume + cost
        if total_cost > self.cash:
            return False

        return True

    def can_sell(self, code: str, volume: int) -> bool:
        """
        检查是否可以卖出

        Args:
            code: 股票代码
            volume: 数量

        Returns:
            是否可以卖出
        """
        if code not in self.positions:
            return False

        position = self.positions[code]
        return position.available >= volume

    def buy(self, code: str, volume: int, price: float, commission: float, buy_time: CTime):
        """
        买入股票

        Args:
            code: 股票代码
            volume: 数量
            price: 价格
            commission: 手续费
            buy_time: 买入时间
        """
        total_cost = price * volume + commission

        # 扣除资金
        self.cash -= total_cost

        # 增加持仓
        position = self.get_position(code)
        position.add(volume, price, commission, buy_time)

    def sell(self, code: str, volume: int, price: float, commission: float, tax: float) -> tuple:
        """
        卖出股票

        Args:
            code: 股票代码
            volume: 数量
            price: 价格
            commission: 手续费
            tax: 印花税

        Returns:
            (实现盈亏, 收益率)
        """
        if code not in self.positions:
            raise ValueError(f"没有持仓: {code}")

        # 减少持仓
        position = self.positions[code]
        realized_profit, realized_profit_rate = position.reduce(volume, price, commission, tax)

        # 增加资金
        revenue = price * volume - commission - tax
        self.cash += revenue

        return realized_profit, realized_profit_rate

    def __str__(self) -> str:
        """字符串表示"""
        lines = [f"资金: {self.cash:.2f}, 总资产: {self.get_total_value():.2f}, "
                 f"总盈亏: {self.get_total_profit():.2f} ({self.get_total_profit_rate()*100:.2f}%), "
                 f"仓位: {self.get_position_ratio()*100:.1f}%"]

        for code, position in self.positions.items():
            if position.volume > 0:
                lines.append(f"  {position}")

        return "\n".join(lines)
