"""
MACD策略示例

基于MACD指标的交易策略
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Backtest.Strategy import CStrategy, CSignal


class CMACDStrategy(CStrategy):
    """
    MACD金叉死叉策略

    策略逻辑：
    - MACD金叉时买入
    - MACD死叉时卖出
    """

    def __init__(self,
                 name: str = "MACD策略",
                 buy_percent: float = 0.3):
        super().__init__(name)
        self.buy_percent = buy_percent
        # 记录上一次的MACD状态
        self.context['last_macd_state'] = {}  # {code: 'above'/'below'}

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            kl_list = chan[0]

            # 确保有足够的K线
            if len(kl_list) < 2:
                continue

            # 获取最新的MACD值
            current_klc = kl_list[-1]
            prev_klc = kl_list[-2]

            # 检查是否有MACD数据
            if not hasattr(current_klc, 'metric') or 'macd' not in current_klc.metric:
                continue

            current_macd = current_klc.metric.get('macd', {})
            prev_macd = prev_klc.metric.get('macd', {})

            if not current_macd or not prev_macd:
                continue

            current_dif = current_macd.get('dif', 0)
            current_dea = current_macd.get('dea', 0)
            prev_dif = prev_macd.get('dif', 0)
            prev_dea = prev_macd.get('dea', 0)

            # 判断当前状态
            current_state = 'above' if current_dif > current_dea else 'below'
            prev_state = 'above' if prev_dif > prev_dea else 'below'

            # 检测金叉（买入信号）
            if prev_state == 'below' and current_state == 'above':
                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason="MACD金叉"
                    ))

            # 检测死叉（卖出信号）
            elif prev_state == 'above' and current_state == 'below':
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason="MACD死叉"
                    ))

        return signals


class CMACDDivergenceStrategy(CStrategy):
    """
    MACD背驰策略

    策略逻辑：
    - 结合缠论背驰和MACD背驰
    - 背驰时买入/卖出
    """

    def __init__(self,
                 name: str = "MACD背驰策略",
                 buy_percent: float = 0.25):
        super().__init__(name)
        self.buy_percent = buy_percent

    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            # 获取最新买卖点
            bsp_list = chan.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            # 检查是否有背驰信息
            # 这里简化处理，实际应该检查笔的MACD面积
            kl_list = chan[0]
            if len(kl_list) < 2:
                continue

            # 买入信号：买点 + 底背驰
            if last_bsp.is_buy:
                if code not in positions or positions[code].volume == 0:
                    signals.append(CSignal(
                        code=code,
                        direction="buy",
                        percent=self.buy_percent,
                        reason=f"背驰买点: {last_bsp.type2str()}"
                    ))

            # 卖出信号：卖点 + 顶背驰
            elif not last_bsp.is_buy:
                if code in positions and positions[code].volume > 0:
                    signals.append(CSignal(
                        code=code,
                        direction="sell",
                        percent=1.0,
                        reason=f"背驰卖点: {last_bsp.type2str()}"
                    ))

        return signals
