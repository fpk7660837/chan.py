"""
EventDetector - 事件检测器

检测各种市场事件（买卖点、价格突破等）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from Chan import CChan
from Common.CEnum import BSP_TYPE, FX_TYPE


@dataclass
class CEvent:
    """事件对象"""

    type: str                  # 事件类型: bsp/price_break/stop_loss等
    code: str                  # 股票代码
    name: str                  # 股票名称
    level: str                 # 级别: high/medium/low
    title: str                 # 标题
    message: str               # 消息内容
    data: Dict                 # 附加数据
    timestamp: datetime        # 事件时间戳

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.title}: {self.message}"


class CEventDetector(ABC):
    """事件检测器基类"""

    @abstractmethod
    def detect(self, chan_dict: Dict[str, CChan]) -> List[CEvent]:
        """
        检测事件

        Args:
            chan_dict: {code: CChan对象}

        Returns:
            事件列表
        """
        pass


class CBSPDetector(CEventDetector):
    """买卖点检测器"""

    def __init__(self,
                 bsp_types: Optional[List[BSP_TYPE]] = None,
                 time_window_days: int = 3,
                 stock_names: Optional[Dict[str, str]] = None):
        """
        Args:
            bsp_types: 关注的买卖点类型
            time_window_days: 时间窗口（天），只关注最近N天的买卖点
            stock_names: 股票名称字典 {code: name}
        """
        if bsp_types is None:
            self.bsp_types = [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2]
        else:
            self.bsp_types = bsp_types

        self.time_window_days = time_window_days
        self.stock_names = stock_names or {}
        self.last_notified = {}  # {code_bsptype: CTime}

    def detect(self, chan_dict: Dict[str, CChan]) -> List[CEvent]:
        """检测买卖点"""
        from datetime import timedelta

        events = []
        cutoff_time = datetime.now() - timedelta(days=self.time_window_days)

        for code, chan in chan_dict.items():
            # 获取最近的买卖点
            bsp_list = chan.get_latest_bsp(number=5)

            for bsp in bsp_list:
                # 检查买卖点时间
                bsp_datetime = datetime(
                    bsp.klu.time.year,
                    bsp.klu.time.month,
                    bsp.klu.time.day
                )

                if bsp_datetime < cutoff_time:
                    continue

                # 检查类型
                if not any(t in bsp.type for t in self.bsp_types):
                    continue

                # 检查是否已通知
                key = f"{code}_{bsp.type2str()}_{bsp.klu.time}"
                if key in self.last_notified:
                    continue

                # 创建事件
                level = "high" if bsp.is_buy else "medium"
                direction = "买点" if bsp.is_buy else "卖点"

                events.append(CEvent(
                    type="bsp",
                    code=code,
                    name=self.stock_names.get(code, code),
                    level=level,
                    title=f"{direction}信号",
                    message=f"{code} {self.stock_names.get(code, '')} 出现{bsp.type2str()}{direction}",
                    data={
                        "bsp_type": bsp.type2str(),
                        "is_buy": bsp.is_buy,
                        "time": str(bsp.klu.time),
                        "price": bsp.klu.close,
                    },
                    timestamp=datetime.now()
                ))

                # 记录已通知
                self.last_notified[key] = bsp.klu.time

        return events


class CPriceBreakDetector(CEventDetector):
    """价格突破检测器"""

    def __init__(self,
                 break_type: str = "high",
                 lookback_days: int = 20,
                 stock_names: Optional[Dict[str, str]] = None):
        """
        Args:
            break_type: "high"/"low"/"both"
            lookback_days: 回溯天数
            stock_names: 股票名称字典
        """
        self.break_type = break_type
        self.lookback_days = lookback_days
        self.stock_names = stock_names or {}
        self.last_notified = {}  # {code_type: datetime}

    def detect(self, chan_dict: Dict[str, CChan]) -> List[CEvent]:
        """检测价格突破"""
        events = []

        for code, chan in chan_dict.items():
            kl_list = chan[0]

            if len(kl_list) < self.lookback_days + 1:
                continue

            # 当前K线
            current_klu = kl_list[-1][-1]

            # 前N天的最高价和最低价
            prev_high = max(klc.high for klc in kl_list[-(self.lookback_days+1):-1])
            prev_low = min(klc.low for klc in kl_list[-(self.lookback_days+1):-1])

            # 检测突破新高
            if self.break_type in ["high", "both"] and current_klu.high > prev_high:
                key = f"{code}_high"
                # 避免重复通知（同一天只通知一次）
                if key not in self.last_notified or \
                   (datetime.now() - self.last_notified[key]).days >= 1:

                    events.append(CEvent(
                        type="price_break",
                        code=code,
                        name=self.stock_names.get(code, code),
                        level="medium",
                        title="突破新高",
                        message=f"{code} {self.stock_names.get(code, '')} 价格突破{self.lookback_days}日新高",
                        data={
                            "price": current_klu.close,
                            "prev_high": prev_high,
                            "break_pct": (current_klu.high - prev_high) / prev_high * 100,
                        },
                        timestamp=datetime.now()
                    ))

                    self.last_notified[key] = datetime.now()

            # 检测跌破新低
            if self.break_type in ["low", "both"] and current_klu.low < prev_low:
                key = f"{code}_low"
                if key not in self.last_notified or \
                   (datetime.now() - self.last_notified[key]).days >= 1:

                    events.append(CEvent(
                        type="price_break",
                        code=code,
                        name=self.stock_names.get(code, code),
                        level="medium",
                        title="跌破新低",
                        message=f"{code} {self.stock_names.get(code, '')} 价格跌破{self.lookback_days}日新低",
                        data={
                            "price": current_klu.close,
                            "prev_low": prev_low,
                            "break_pct": (current_klu.low - prev_low) / prev_low * 100,
                        },
                        timestamp=datetime.now()
                    ))

                    self.last_notified[key] = datetime.now()

        return events


class CPositionMonitorDetector(CEventDetector):
    """持仓监控检测器（止损止盈）"""

    def __init__(self,
                 position_storage,
                 stop_loss: float = -0.05,
                 take_profit: float = 0.20,
                 stock_names: Optional[Dict[str, str]] = None):
        """
        Args:
            position_storage: 持仓存储对象
            stop_loss: 止损比例（负数）
            take_profit: 止盈比例（正数）
            stock_names: 股票名称字典
        """
        self.position_storage = position_storage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.stock_names = stock_names or {}
        self.last_notified = {}

    def detect(self, chan_dict: Dict[str, CChan]) -> List[CEvent]:
        """检测持仓状态"""
        events = []

        # 获取所有持仓
        positions = self.position_storage.get_all_positions()

        for code, position in positions.items():
            if code not in chan_dict:
                continue

            # 获取当前价格
            if len(chan_dict[code][0]) == 0:
                continue

            current_price = chan_dict[code][0][-1][-1].close
            profit_rate = (current_price - position.cost_price) / position.cost_price

            # 检测止损
            if profit_rate <= self.stop_loss:
                key = f"{code}_stop_loss"
                if key not in self.last_notified:
                    events.append(CEvent(
                        type="stop_loss",
                        code=code,
                        name=self.stock_names.get(code, code),
                        level="high",
                        title="止损提醒",
                        message=f"{code} {self.stock_names.get(code, '')} 触发止损线，当前亏损{profit_rate*100:.2f}%",
                        data={
                            "profit_rate": profit_rate,
                            "current_price": current_price,
                            "cost_price": position.cost_price,
                        },
                        timestamp=datetime.now()
                    ))
                    self.last_notified[key] = datetime.now()

            # 检测止盈
            elif profit_rate >= self.take_profit:
                key = f"{code}_take_profit"
                if key not in self.last_notified:
                    events.append(CEvent(
                        type="take_profit",
                        code=code,
                        name=self.stock_names.get(code, code),
                        level="medium",
                        title="止盈提醒",
                        message=f"{code} {self.stock_names.get(code, '')} 达到止盈目标，当前盈利{profit_rate*100:.2f}%",
                        data={
                            "profit_rate": profit_rate,
                            "current_price": current_price,
                            "cost_price": position.cost_price,
                        },
                        timestamp=datetime.now()
                    ))
                    self.last_notified[key] = datetime.now()

        return events
