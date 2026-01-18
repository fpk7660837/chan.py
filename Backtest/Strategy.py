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
import pandas as pd


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
    一类买卖点策略示例（优化版）

    策略逻辑：
    - 当出现买卖点信号时进行交易
    - 支持止盈止损机制
    - 扩大买卖点类型覆盖范围
    """

    def __init__(self, name: str = "一类买卖点策略",
                 buy_percent: float = 0.2,
                 bsp_types: Optional[List] = None,
                 stop_loss: float = -0.05,
                 take_profit: float = 0.20):
        """
        Args:
            name: 策略名称
            buy_percent: 每次买入仓位比例
            bsp_types: 关注的买卖点类型列表，默认关注T1、T1P、T2、T2P
            stop_loss: 止损比例，默认-5%
            take_profit: 止盈比例，默认+20%
        """
        super().__init__(name)
        self.buy_percent = buy_percent
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # 默认关注一二类买卖点（方案C：扩大买卖点类型）
        if bsp_types is None:
            from Common.CEnum import BSP_TYPE
            self.bsp_types = [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S]
        else:
            self.bsp_types = bsp_types

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            cur_lv_chan = chan[0]

            # 确保有足够的K线
            if len(cur_lv_chan) < 1:
                continue

            # 方案B：检查现有持仓的止盈止损
            if code in positions and positions[code].volume > 0:
                position = positions[code]
                current_price = cur_lv_chan[-1][-1].close  # 最新收盘价
                profit_rate = (current_price - position.cost_price) / position.cost_price

                # 止损检查
                if profit_rate <= self.stop_loss:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"止损{self.stop_loss*100:.1f}%"
                    ))
                    continue

                # 止盈检查
                if profit_rate >= self.take_profit:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"止盈+{self.take_profit*100:.1f}%"
                    ))
                    continue

            # 获取最近的买卖点
            bsp_list = chan.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            # 检查买卖点类型是否匹配
            if not any(t in last_bsp.type for t in self.bsp_types):
                continue

            # 方案A：简化买入条件（移除严格的位置和分型要求）
            if last_bsp.is_buy:
                # 如果已持仓则跳过
                if code in positions and positions[code].volume > 0:
                    continue

                signals.append(CSignal(
                    code=code,
                    direction="buy",
                    percent=self.buy_percent,
                    reason=f"{last_bsp.type2str()}买点"
                ))

            # 方案A：简化卖出条件
            else:  # 卖点
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


class CBSPStrategyWithSell(CStrategy):
    """
    增强版买卖点策略（带卖出逻辑）

    策略逻辑：
    - 买入：出现买卖点信号
    - 卖出：出现卖点 OR 止盈止损 OR 技术指标信号
    """

    def __init__(self, name: str = "增强买卖点策略",
                 buy_percent: float = 0.2,
                 bsp_types: Optional[List] = None,
                 stop_loss: float = -0.05,
                 take_profit: float = 0.20,
                 enable_ma_sell: bool = True,
                 ma_short: int = 5,
                 ma_long: int = 20):
        """
        Args:
            name: 策略名称
            buy_percent: 每次买入仓位比例
            bsp_types: 关注的买卖点类型列表
            stop_loss: 止损比例
            take_profit: 止盈比例
            enable_ma_sell: 是否启用均线卖出
            ma_short: 短期均线周期
            ma_long: 长期均线周期
        """
        super().__init__(name)
        self.buy_percent = buy_percent
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.enable_ma_sell = enable_ma_sell
        self.ma_short = ma_short
        self.ma_long = ma_long

        # 默认关注一二类买卖点
        if bsp_types is None:
            from Common.CEnum import BSP_TYPE
            self.bsp_types = [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S]
        else:
            self.bsp_types = bsp_types

        # 记录上一次均线状态
        self.context['last_ma_state'] = {}

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            cur_lv_chan = chan[0]

            # 确保有足够的K线
            if len(cur_lv_chan) < self.ma_long:
                continue

            current_price = cur_lv_chan[-1][-1].close

            # ========== 卖出逻辑 ==========
            if code in positions and positions[code].volume > 0:
                position = positions[code]
                profit_rate = (current_price - position.cost_price) / position.cost_price

                # 1. 止损检查
                if profit_rate <= self.stop_loss:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"止损{self.stop_loss*100:.1f}%"
                    ))
                    continue

                # 2. 止盈检查
                if profit_rate >= self.take_profit:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"止盈+{self.take_profit*100:.1f}%"
                    ))
                    continue

                # 3. 缠论卖点检查
                bsp_list = chan.get_latest_bsp(number=1)
                if bsp_list:
                    last_bsp = bsp_list[0]
                    if not last_bsp.is_buy and any(t in last_bsp.type for t in self.bsp_types):
                        signals.append(CSignal(
                            code=code,
                            direction="sell",
                            percent=1.0,
                            reason=f"{last_bsp.type2str()}卖点"
                        ))
                        continue

                # 4. 均线死叉卖出
                if self.enable_ma_sell:
                    ma_short = self._calculate_ma(cur_lv_chan, self.ma_short)
                    ma_long = self._calculate_ma(cur_lv_chan, self.ma_long)
                    current_state = 'above' if ma_short > ma_long else 'below'
                    last_state = self.context['last_ma_state'].get(code)

                    if last_state == 'above' and current_state == 'below':
                        signals.append(CSignal(
                            code=code,
                            direction="sell",
                            percent=1.0,
                            reason=f"死叉: MA{self.ma_short}下穿MA{self.ma_long}"
                        ))

                    self.context['last_ma_state'][code] = current_state
                    if current_state == 'below':
                        continue  # 死叉后跳过买入

            # ========== 买入逻辑 ==========
            # 只在未持仓或均线多头时买入
            if self.enable_ma_sell:
                ma_short = self._calculate_ma(cur_lv_chan, self.ma_short)
                ma_long = self._calculate_ma(cur_lv_chan, self.ma_long)
                if ma_short <= ma_long:
                    continue  # 均线空头，不买入

            # 检查买卖点
            bsp_list = chan.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            # 检查买卖点类型是否匹配
            if not any(t in last_bsp.type for t in self.bsp_types):
                continue

            # 买入信号
            if last_bsp.is_buy:
                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason=f"{last_bsp.type2str()}买点"
                    ))

        return signals

    def _calculate_ma(self, kl_list, period: int) -> float:
        """计算均线"""
        closes = []
        count = 0
        for klc in reversed(kl_list):
            closes.append(klc[-1].close)
            count += 1
            if count >= period:
                break
        return sum(closes) / len(closes) if closes else 0.0


class CMACDStrategy(CStrategy):
    """
    MACD策略

    策略逻辑：
    - MACD金叉（DIF上穿DEA）且DIF>0时买入
    - MACD死叉（DIF下穿DEA）时卖出
    """

    def __init__(self, name: str = "MACD策略",
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 buy_percent: float = 0.3):
        """
        Args:
            name: 策略名称
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            buy_percent: 每次买入仓位比例
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.buy_percent = buy_percent

        # 记录上一次的MACD状态
        self.context['last_macd_state'] = {}  # {code: {'dif': float, 'dea': float, 'golden': bool}}

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            kl_list = chan[0]

            # 确保有足够的K线
            min_period = self.slow_period + self.signal_period
            if len(kl_list) < min_period:
                continue

            # 计算MACD
            macd = self._calculate_macd(kl_list)
            if macd is None:
                continue

            dif, dea, histogram = macd

            # 获取上一次的状态
            last_state = self.context['last_macd_state'].get(code, {})
            last_dif = last_state.get('dif', 0)
            last_dea = last_state.get('dea', 0)

            # 判断金叉（买入信号）
            if last_dif <= last_dea and dif > dea and dif > 0:
                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason=f"MACD金叉 DIF={dif:.3f}>DEA={dea:.3f}"
                    ))

            # 判断死叉（卖出信号）
            elif last_dif >= last_dea and dif < dea:
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"MACD死叉 DIF={dif:.3f}<DEA={dea:.3f}"
                    ))

            # 更新状态
            self.context['last_macd_state'][code] = {'dif': dif, 'dea': dea}

        return signals

    def _calculate_macd(self, kl_list):
        """计算MACD指标"""
        import numpy as np

        # 获取收盘价序列（从旧到新）
        closes = []
        for klc in kl_list:
            closes.append(klc[-1].close)

        if len(closes) < self.slow_period + self.signal_period:
            return None

        closes = np.array(closes)

        # 计算EMA
        def ema(data, period):
            return data.ewm(span=period, adjust=False).mean()

        # 计算DIF
        ema_fast = ema(pd.Series(closes), self.fast_period)
        ema_slow = ema(pd.Series(closes), self.slow_period)
        dif = ema_fast - ema_slow

        # 计算DEA
        dea = ema(dif, self.signal_period)

        # 返回最新值
        return dif.iloc[-1], dea.iloc[-1], (dif - dea).iloc[-1]


class CEnhancedMACDStrategy(CStrategy):
    """
    增强版MACD策略

    策略逻辑：
    - MACD金叉买入，死叉卖出
    - 动态仓位：根据MACD强度调整仓位
    - 趋势过滤：只在DIF>0时加仓
    """

    def __init__(self, name: str = "增强MACD策略",
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 base_position: float = 0.15,
                 max_position: float = 0.25):
        """
        Args:
            name: 策略名称
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            base_position: 基础仓位比例
            max_position: 最大仓位比例
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.base_position = base_position
        self.max_position = max_position

        self.context['last_macd'] = {}

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            kl_list = chan[0]
            min_period = self.slow_period + self.signal_period

            if len(kl_list) < min_period:
                continue

            macd = self._calculate_macd(kl_list)
            if macd is None:
                continue

            dif, dea, histogram = macd
            last_state = self.context['last_macd'].get(code, {})
            last_dif = last_state.get('dif', 0)
            last_dea = last_state.get('dea', 0)

            # 金叉买入
            if last_dif <= last_dea and dif > dea:
                # 动态仓位：根据MACD强度调整
                if dif > 0:  # 多头市场，加大仓位
                    position_size = self.max_position
                else:  # 空头市场金叉，小仓位
                    position_size = self.base_position

                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=position_size,
                        reason=f"MACD金叉 DIF={dif:.3f} 仓位{position_size*100:.0f}%"
                    ))

            # 死叉卖出
            elif last_dif >= last_dea and dif < dea:
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"MACD死叉 DIF={dif:.3f}"
                    ))

            self.context['last_macd'][code] = {'dif': dif, 'dea': dea}

        return signals

    def _calculate_macd(self, kl_list):
        """计算MACD指标"""
        import numpy as np

        closes = [klc[-1].close for klc in kl_list]
        if len(closes) < self.slow_period + self.signal_period:
            return None

        closes = np.array(closes)

        def ema(data, period):
            return data.ewm(span=period, adjust=False).mean()

        ema_fast = ema(pd.Series(closes), self.fast_period)
        ema_slow = ema(pd.Series(closes), self.slow_period)
        dif = ema_fast - ema_slow
        dea = ema(dif, self.signal_period)

        return dif.iloc[-1], dea.iloc[-1], (dif - dea).iloc[-1]


class CRSIStrategy(CStrategy):
    """
    RSI策略

    策略逻辑：
    - RSI超卖(<30)买入
    - RSI超买(>70)卖出
    """

    def __init__(self, name: str = "RSI策略",
                 period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 buy_percent: float = 0.2):
        """
        Args:
            name: 策略名称
            period: RSI周期
            oversold: 超卖阈值
            overbought: 超买阈值
            buy_percent: 买入仓位比例
        """
        super().__init__(name)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.buy_percent = buy_percent

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            kl_list = chan[0]

            if len(kl_list) < self.period + 5:
                continue

            rsi = self._calculate_rsi(kl_list)
            if rsi is None:
                continue

            # RSI超卖买入
            if rsi < self.oversold:
                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason=f"RSI超卖 {rsi:.1f}"
                    ))

            # RSI超买卖出
            elif rsi > self.overbought:
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"RSI超买 {rsi:.1f}"
                    ))

        return signals

    def _calculate_rsi(self, kl_list):
        """计算RSI指标"""
        import numpy as np

        closes = [klc[-1].close for klc in kl_list]
        if len(closes) < self.period + 1:
            return None

        # 计算价格变化
        deltas = np.diff(closes)

        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # 计算平均涨跌幅
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class CDualTimeFrameMACDStrategy(CStrategy):
    """
    双时间周期MACD策略（周线+日线共振）

    策略逻辑：
    - 周线MACD金叉确认大趋势
    - 日线MACD金叉作为入场信号
    - 日线MACD死叉作为出场信号
    """

    def __init__(self, name: str = "双周期MACD策略",
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 buy_percent: float = 0.2):
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.buy_percent = buy_percent

        self.context['daily_macd'] = {}
        self.context['weekly_trend'] = {}  # 记录周线趋势

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            daily_chan = chan[0]

            if len(daily_chan) < self.slow_period + self.signal_period:
                continue

            # 计算日线MACD
            daily_macd = self._calculate_macd(daily_chan)
            if not daily_macd:
                continue

            dif_daily, dea_daily, _ = daily_macd
            last_daily = self.context['daily_macd'].get(code, {})
            last_dif_daily = last_daily.get('dif', 0)
            last_dea_daily = last_daily.get('dea', 0)

            # 判断周线趋势（通过日线数据模拟）
            # 简化方法：检查过去20天的DIF均值
            if len(daily_chan) >= 40:
                dif_trend = self._calculate_macd_trend(daily_chan)
                weekly_bullish = dif_trend > 0
            else:
                weekly_bullish = dif_daily > 0

            # 卖出：日线死叉
            if last_dif_daily >= last_dea_daily and dif_daily < dea_daily:
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"日线死叉 DIF={dif_daily:.3f}"
                    ))

            # 买入：日线金叉 + 周线多头
            elif last_dif_daily <= last_dea_daily and dif_daily > dea_daily:
                if weekly_bullish:  # 周线多头确认
                    if code not in positions or positions[code].volume == 0:
                        signals.append(CSignal(
                            code=code,
                            direction="buy",
                            percent=self.buy_percent,
                            reason=f"日线金叉+周线多头 DIF={dif_daily:.3f}"
                        ))

            self.context['daily_macd'][code] = {'dif': dif_daily, 'dea': dea_daily}

        return signals

    def _calculate_macd(self, kl_list):
        """计算MACD指标"""
        import numpy as np

        closes = [klc[-1].close for klc in kl_list]
        if len(closes) < self.slow_period + self.signal_period:
            return None

        closes = np.array(closes)

        def ema(data, period):
            return data.ewm(span=period, adjust=False).mean()

        ema_fast = ema(pd.Series(closes), self.fast_period)
        ema_slow = ema(pd.Series(closes), self.slow_period)
        dif = ema_fast - ema_slow
        dea = ema(dif, self.signal_period)

        return dif.iloc[-1], dea.iloc[-1], (dif - dea).iloc[-1]

    def _calculate_macd_trend(self, kl_list):
        """计算MACD DIF的趋势（模拟周线）"""
        import numpy as np

        closes = [klc[-1].close for klc in kl_list[-20:]]  # 最近20天
        closes = np.array(closes)

        def ema(data, period):
            return data.ewm(span=period, adjust=False).mean()

        ema_fast = ema(pd.Series(closes), self.fast_period)
        ema_slow = ema(pd.Series(closes), self.slow_period)
        dif = ema_fast - ema_slow

        return dif.mean()


class CHybridStrategy(CStrategy):
    """
    混合策略：缠论买卖点 + MACD确认

    策略逻辑：
    - 买入：缠论买点 AND MACD金叉或DIF>0
    - 卖出：缠论卖点 OR MACD死叉 OR 止盈止损
    """

    def __init__(self, name: str = "混合策略",
                 buy_percent: float = 0.2,
                 stop_loss: float = -0.08,
                 take_profit: float = 0.15):
        """
        Args:
            name: 策略名称
            buy_percent: 买入仓位比例
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        super().__init__(name)
        self.buy_percent = buy_percent
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # 记录MACD状态
        self.context['last_macd'] = {}

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            cur_lv_chan = chan[0]

            if len(cur_lv_chan) < 40:  # 需要足够的K线计算MACD
                continue

            current_price = cur_lv_chan[-1][-1].close

            # ========== 卖出逻辑 ==========
            if code in positions and positions[code].volume > 0:
                position = positions[code]
                profit_rate = (current_price - position.cost_price) / position.cost_price

                # 1. 止损
                if profit_rate <= self.stop_loss:
                    signals.append(CSignal(
                        code=code, direction="sell", percent=1.0,
                        reason=f"止损{self.stop_loss*100:.1f}%"
                    ))
                    continue

                # 2. 止盈
                if profit_rate >= self.take_profit:
                    signals.append(CSignal(
                        code=code, direction="sell", percent=1.0,
                        reason=f"止盈+{self.take_profit*100:.1f}%"
                    ))
                    continue

                # 3. MACD死叉卖出
                macd = self._calculate_macd(cur_lv_chan)
                if macd:
                    dif, dea, _ = macd
                    last_macd = self.context['last_macd'].get(code, {})
                    last_dif = last_macd.get('dif', 0)
                    last_dea = last_macd.get('dea', 0)

                    if last_dif >= last_dea and dif < dea:
                        signals.append(CSignal(
                            code=code, direction="sell", percent=1.0,
                            reason=f"MACD死叉卖出"
                        ))

                    self.context['last_macd'][code] = {'dif': dif, 'dea': dea}

                # 4. 缠论卖点
                bsp_list = chan.get_latest_bsp(number=1)
                if bsp_list:
                    last_bsp = bsp_list[0]
                    if not last_bsp.is_buy:
                        signals.append(CSignal(
                            code=code, direction="sell", percent=1.0,
                            reason=f"{last_bsp.type2str()}卖点"
                        ))
                        continue

            # ========== 买入逻辑 ==========
            # 1. 检查缠论买点
            bsp_list = chan.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]
            if not last_bsp.is_buy:
                continue

            # 2. MACD确认（金叉或DIF>0）
            macd = self._calculate_macd(cur_lv_chan)
            if not macd:
                continue

            dif, dea, histogram = macd

            # MACD多头信号：金叉或DIF>0
            if dif > dea or dif > 0:
                if code not in positions or positions[code].volume == 0:
                    reason = f"{last_bsp.type2str()}买点+MACD确认(DIF={dif:.3f})"
                    signals.append(CSignal(
                        code=code, direction="buy", percent=self.buy_percent, reason=reason
                    ))

        return signals

    def _calculate_macd(self, kl_list):
        """计算MACD指标"""
        import numpy as np

        closes = [klc[-1].close for klc in kl_list]
        if len(closes) < 34:
            return None

        closes = np.array(closes)

        def ema(data, period):
            return data.ewm(span=period, adjust=False).mean()

        ema_fast = ema(pd.Series(closes), 12)
        ema_slow = ema(pd.Series(closes), 26)
        dif = ema_fast - ema_slow
        dea = ema(dif, 9)

        return dif.iloc[-1], dea.iloc[-1], (dif - dea).iloc[-1]


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
