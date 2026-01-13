"""
买卖点策略示例

基于缠论买卖点的交易策略
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Backtest.Strategy import CStrategy, CSignal
from Common.CEnum import BSP_TYPE, FX_TYPE


class CBSPStrategy(CStrategy):
    """
    一类买卖点策略

    策略逻辑：
    - 当出现一类买点且底分型形成后买入
    - 当出现一类卖点且顶分型形成后卖出
    """

    def __init__(self,
                 name: str = "一类买卖点策略",
                 buy_percent: float = 0.2,
                 buy_types: list = None,
                 sell_types: list = None):
        """
        Args:
            name: 策略名称
            buy_percent: 每次买入仓位比例
            buy_types: 关注的买点类型（默认T1和T1P）
            sell_types: 关注的卖点类型（默认T1和T1P）
        """
        super().__init__(name)
        self.buy_percent = buy_percent

        if buy_types is None:
            self.buy_types = [BSP_TYPE.T1, BSP_TYPE.T1P]
        else:
            self.buy_types = buy_types

        if sell_types is None:
            self.sell_types = [BSP_TYPE.T1, BSP_TYPE.T1P]
        else:
            self.sell_types = sell_types

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            # 获取最新买卖点
            bsp_list = chan.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]
            cur_lv_chan = chan[0]

            # 确保有足够的K线
            if len(cur_lv_chan) < 2:
                continue

            # 检查买卖点是否在倒数第二根合并K线上
            if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
                continue

            # 买入信号
            if last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.BOTTOM:
                # 检查买点类型
                if any(t in last_bsp.type for t in self.buy_types):
                    # 如果已持仓则跳过
                    if code in positions and positions[code].volume > 0:
                        continue

                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason=f"{last_bsp.type2str()}买点"
                    ))

            # 卖出信号
            elif not last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.TOP:
                # 检查卖点类型
                if any(t in last_bsp.type for t in self.sell_types):
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


class CMultiLevelBSPStrategy(CStrategy):
    """
    多级别买卖点策略

    策略逻辑：
    - 大级别确认趋势方向
    - 小级别寻找买卖点
    """

    def __init__(self,
                 name: str = "多级别买卖点策略",
                 buy_percent: float = 0.25):
        super().__init__(name)
        self.buy_percent = buy_percent

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            # 检查是否有多级别数据
            if len(chan.lv_list) < 2:
                continue

            # 大级别（第一个级别）
            big_lv = chan[0]
            # 小级别（第二个级别）
            small_lv = chan[1]

            # 大级别买卖点
            big_bsp_list = chan.get_latest_bsp(idx=0, number=1)
            # 小级别买卖点
            small_bsp_list = chan.get_latest_bsp(idx=1, number=1)

            if not big_bsp_list or not small_bsp_list:
                continue

            big_bsp = big_bsp_list[0]
            small_bsp = small_bsp_list[0]

            # 买入条件：大级别买点 + 小级别买点
            if (big_bsp.is_buy and small_bsp.is_buy and
                code not in positions or positions[code].volume == 0):

                signals.append(CSignal(
                    code=code,
                    direction="buy",
                    percent=self.buy_percent,
                    reason=f"多级别买点: 大级别{big_bsp.type2str()}, 小级别{small_bsp.type2str()}"
                ))

            # 卖出条件：大级别卖点或小级别卖点
            elif ((not big_bsp.is_buy or not small_bsp.is_buy) and
                  code in positions and positions[code].volume > 0):

                signals.append(CSignal(
                    code=code,
                    direction="sell",
                    percent=1.0,
                    reason=f"多级别卖点"
                ))

        return signals
