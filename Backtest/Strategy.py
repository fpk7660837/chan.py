"""
Strategy - 策略基类

提供策略开发的基础框架
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from Chan import CChan
from Common.CTime import CTime
from Backtest.Position import CPosition


@dataclass
class CSignal:
    """交易信号"""

    code: str                           # 股票代码
    direction: str                      # 交易方向：buy/sell/close
    volume: Optional[int] = None        # 指定数量（股），与percent互斥
    percent: Optional[float] = None     # 指定仓位百分比（0-1），与volume互斥
    price: Optional[float] = None       # 限价（None表示市价）
    reason: str = ""                    # 信号原因

    def __post_init__(self):
        """验证信号"""
        assert self.direction in ['buy', 'sell', 'close'], f"无效的交易方向: {self.direction}"

        # volume和percent只能指定一个
        if self.volume is not None and self.percent is not None:
            raise ValueError("volume和percent不能同时指定")

        if self.volume is None and self.percent is None:
            # 默认使用percent
            self.percent = 1.0 if self.direction in ['sell', 'close'] else 0.0

        # 验证percent范围
        if self.percent is not None:
            assert 0 <= self.percent <= 1, f"仓位百分比必须在0-1之间: {self.percent}"

    def __str__(self) -> str:
        direction_str = {"buy": "买入", "sell": "卖出", "close": "平仓"}[self.direction]
        amount_str = ""
        if self.volume is not None:
            amount_str = f"{self.volume}股"
        elif self.percent is not None:
            amount_str = f"{self.percent*100:.1f}%仓位"

        price_str = f"@{self.price:.2f}" if self.price else "市价"
        return f"{direction_str} {self.code} {amount_str} {price_str}, 原因: {self.reason}"


class CStrategy(ABC):
    """
    策略基类

    用户需要继承此类并实现on_bar方法来定义自己的策略逻辑
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.context = {}  # 用户自定义上下文，可以存储策略状态

    @abstractmethod
    def on_bar(self,
               chan_dict: Dict[str, CChan],
               positions: Dict[str, CPosition],
               timestamp: CTime) -> List[CSignal]:
        """
        每个时间步的回调函数

        Args:
            chan_dict: {code: CChan对象}，每个CChan包含截至当前时间的所有数据
            positions: {code: CPosition对象}，当前持仓情况
            timestamp: 当前时间

        Returns:
            List[CSignal]: 交易信号列表
        """
        pass

    def on_trade(self, trade):
        """
        交易执行后的回调（可选重写）

        Args:
            trade: CTrade对象
        """
        pass

    def on_backtest_start(self):
        """回测开始前的回调（可选重写）"""
        pass

    def on_backtest_end(self, result):
        """
        回测结束后的回调（可选重写）

        Args:
            result: 回测结果对象
        """
        pass

    def __str__(self) -> str:
        return f"Strategy: {self.name}"


class CBSPStrategy(CStrategy):
    """
    一类买卖点策略示例

    策略逻辑：
    - 当出现一类买点且形成底分型后买入
    - 当出现一类卖点且形成顶分型后卖出
    """

    def __init__(self, name: str = "一类买卖点策略",
                 buy_percent: float = 0.2,
                 bsp_types: Optional[List] = None):
        """
        Args:
            name: 策略名称
            buy_percent: 每次买入仓位比例
            bsp_types: 关注的买卖点类型列表，默认只关注T1和T1P
        """
        super().__init__(name)
        self.buy_percent = buy_percent

        # 默认只关注一类买卖点
        if bsp_types is None:
            from Common.CEnum import BSP_TYPE
            self.bsp_types = [BSP_TYPE.T1, BSP_TYPE.T1P]
        else:
            self.bsp_types = bsp_types

    def on_bar(self, chan_dict, positions, timestamp):
        from Common.CEnum import FX_TYPE

        signals = []

        for code, chan in chan_dict.items():
            # 获取最近的买卖点
            bsp_list = chan.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            # 检查买卖点类型是否匹配
            if not any(t in last_bsp.type for t in self.bsp_types):
                continue

            cur_lv_chan = chan[0]

            # 确保有足够的K线
            if len(cur_lv_chan) < 2:
                continue

            # 检查买卖点是否在倒数第二根合并K线上（确保分型已形成）
            if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
                continue

            # 检查买点
            if last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.BOTTOM:
                # 如果已持仓则跳过
                if code in positions and positions[code].volume > 0:
                    continue

                signals.append(CSignal(
                    code=code,
                    direction="buy",
                    percent=self.buy_percent,
                    reason=f"{last_bsp.type2str()}买点"
                ))

            # 检查卖点
            elif not last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.TOP:
                # 如果没有持仓则跳过
                if code not in positions or positions[code].volume == 0:
                    continue

                signals.append(CSignal(
                    code=code,
                    direction="sell",
                    percent=1.0,  # 全部卖出
                    reason=f"{last_bsp.type2str()}卖点"
                ))

        return signals


class CMAStrategy(CStrategy):
    """
    均线策略示例

    策略逻辑：
    - 短期均线上穿长期均线时买入（金叉）
    - 短期均线下穿长期均线时卖出（死叉）
    """

    def __init__(self, name: str = "均线策略",
                 short_period: int = 5,
                 long_period: int = 20,
                 buy_percent: float = 0.3):
        """
        Args:
            name: 策略名称
            short_period: 短期均线周期
            long_period: 长期均线周期
            buy_percent: 每次买入仓位比例
        """
        super().__init__(name)
        self.short_period = short_period
        self.long_period = long_period
        self.buy_percent = buy_percent

        # 用于记录上一次的均线状态
        self.context['last_ma_state'] = {}  # {code: 'above'/'below'}

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            kl_list = chan[0]

            # 确保有足够的K线
            if len(kl_list) < self.long_period:
                continue

            # 计算短期和长期均线
            short_ma = self._calculate_ma(kl_list, self.short_period)
            long_ma = self._calculate_ma(kl_list, self.long_period)

            # 获取上一次的状态
            last_state = self.context['last_ma_state'].get(code)

            # 判断当前状态
            current_state = 'above' if short_ma > long_ma else 'below'

            # 检测金叉（买入信号）
            if last_state == 'below' and current_state == 'above':
                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason=f"金叉: MA{self.short_period}上穿MA{self.long_period}"
                    ))

            # 检测死叉（卖出信号）
            elif last_state == 'above' and current_state == 'below':
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"死叉: MA{self.short_period}下穿MA{self.long_period}"
                    ))

            # 更新状态
            self.context['last_ma_state'][code] = current_state

        return signals

    def _calculate_ma(self, kl_list, period: int) -> float:
        """计算均线"""
        # 获取最近period根K线的收盘价
        closes = []
        count = 0
        for klc in reversed(kl_list):
            closes.append(klc[-1].close)
            count += 1
            if count >= period:
                break

        return sum(closes) / len(closes) if closes else 0.0
