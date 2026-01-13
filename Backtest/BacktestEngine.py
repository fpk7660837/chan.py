"""
BacktestEngine - 回测引擎核心

负责驱动整个回测流程
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CTime import CTime
from Common.CEnum import AUTYPE

from Backtest.BacktestConfig import CBacktestConfig
from Backtest.Strategy import CStrategy, CSignal
from Backtest.Position import CPosition, CPositionManager
from Backtest.Trade import CTrade


@dataclass
class CBacktestResult:
    """回测结果"""

    # 基本信息
    strategy_name: str                  # 策略名称
    start_time: str                     # 回测开始时间
    end_time: str                       # 回测结束时间
    initial_capital: float              # 初始资金

    # 交易记录
    trades: List[CTrade] = field(default_factory=list)  # 所有交易记录

    # 权益曲线
    equity_curve: List[tuple] = field(default_factory=list)  # [(时间, 总资产, 持仓市值, 现金)]

    # 绩效指标
    metrics: Dict = field(default_factory=dict)

    def add_trade(self, trade: CTrade):
        """添加交易记录"""
        self.trades.append(trade)

    def add_equity_point(self, time: CTime, total_value: float, positions_value: float, cash: float):
        """添加权益曲线点"""
        self.equity_curve.append((time, total_value, positions_value, cash))


class CBacktestEngine:
    """
    回测引擎核心类

    负责：
    - 驱动CChan逐步加载历史数据
    - 在每个时间点调用策略决策
    - 管理持仓和资金
    - 记录所有交易
    - 计算绩效指标
    """

    def __init__(self, config: CBacktestConfig):
        self.config = config
        self.position_manager = CPositionManager(config.initial_capital)
        self.result: Optional[CBacktestResult] = None

        # 统计信息
        self.total_bars = 0      # 总K线数
        self.current_time: Optional[CTime] = None

    def run(self, strategy: CStrategy, code_list: List[str]) -> CBacktestResult:
        """
        执行回测

        Args:
            strategy: 策略对象
            code_list: 股票代码列表

        Returns:
            CBacktestResult: 回测结果
        """
        print(f"\n=== 开始回测 ===")
        print(f"策略: {strategy.name}")
        print(f"股票池: {code_list}")
        print(f"时间范围: {self.config.begin_time} ~ {self.config.end_time or '至今'}")
        print(f"初始资金: {self.config.initial_capital:.2f}")
        print()

        # 初始化回测结果
        self.result = CBacktestResult(
            strategy_name=strategy.name,
            start_time=self.config.begin_time,
            end_time=self.config.end_time or datetime.now().strftime("%Y-%m-%d"),
            initial_capital=self.config.initial_capital,
        )

        # 调用策略的开始回调
        strategy.on_backtest_start()

        # 初始化CChan实例
        print("正在加载数据...")
        chan_dict = self._init_chan_dict(code_list)

        # 创建迭代器字典
        iterators = {}
        for code, chan in chan_dict.items():
            iterators[code] = chan.step_load()

        # 逐步推进回测
        print("正在回测...\n")
        self._backtest_loop(strategy, chan_dict, iterators)

        # 计算绩效指标
        print("\n正在计算绩效指标...")
        self._calculate_performance()

        # 调用策略的结束回调
        strategy.on_backtest_end(self.result)

        print("\n=== 回测完成 ===")
        return self.result

    def _init_chan_dict(self, code_list: List[str]) -> Dict[str, CChan]:
        """初始化CChan字典"""
        chan_dict = {}

        # 准备缠论配置
        chan_config = CChanConfig(self.config.chan_config)

        for code in code_list:
            try:
                chan = CChan(
                    code=code,
                    begin_time=self.config.begin_time,
                    end_time=self.config.end_time,
                    data_src=self.config.data_src,
                    lv_list=self.config.lv_list,
                    config=chan_config,
                    autype=AUTYPE.QFQ,
                )
                chan_dict[code] = chan
                print(f"  ✓ {code} 数据加载成功")
            except Exception as e:
                print(f"  ✗ {code} 数据加载失败: {e}")

        if not chan_dict:
            raise Exception("没有成功加载任何股票数据")

        return chan_dict

    def _backtest_loop(self, strategy: CStrategy, chan_dict: Dict[str, CChan], iterators: Dict):
        """回测主循环"""

        while iterators:
            # 推进所有迭代器
            finished_codes = []
            for code in list(iterators.keys()):
                try:
                    # 获取下一个快照
                    chan_snapshot = next(iterators[code])

                    # 更新当前时间
                    if len(chan_snapshot[0]) > 0:
                        last_klu = chan_snapshot[0][-1][-1]
                        self.current_time = last_klu.time

                    # 更新chan_dict中的对象
                    chan_dict[code] = chan_snapshot

                except StopIteration:
                    # 该股票数据已结束
                    finished_codes.append(code)

            # 移除已结束的迭代器
            for code in finished_codes:
                del iterators[code]

            # 如果所有股票都结束了，退出循环
            if not iterators:
                break

            self.total_bars += 1

            # 更新持仓价格
            self._update_positions_price(chan_dict)

            # 调用策略生成信号
            signals = strategy.on_bar(
                chan_dict=chan_dict,
                positions=self.position_manager.positions,
                timestamp=self.current_time
            )

            # 执行交易信号
            if signals:
                self._execute_signals(signals, chan_dict, strategy)

            # 记录权益曲线
            if self.current_time:
                self.result.add_equity_point(
                    self.current_time,
                    self.position_manager.get_total_value(),
                    self.position_manager.get_positions_value(),
                    self.position_manager.cash
                )

            # 打印进度
            if self.config.print_progress and self.total_bars % self.config.progress_interval == 0:
                print(f"进度: {self.total_bars} bars, "
                      f"时间: {self.current_time}, "
                      f"总资产: {self.position_manager.get_total_value():.2f}, "
                      f"收益率: {self.position_manager.get_total_profit_rate()*100:.2f}%")

        # 最后一天结算（更新可用数量）
        self.position_manager.update_available()

    def _update_positions_price(self, chan_dict: Dict[str, CChan]):
        """更新所有持仓的价格"""
        prices = {}
        for code, chan in chan_dict.items():
            if len(chan[0]) > 0:
                # 使用最新K线的收盘价
                prices[code] = chan[0][-1][-1].close

        self.position_manager.update_all_prices(prices, self.current_time)

    def _execute_signals(self, signals: List[CSignal], chan_dict: Dict[str, CChan], strategy: CStrategy):
        """执行交易信号"""

        for signal in signals:
            try:
                if signal.direction in ['buy']:
                    self._execute_buy(signal, chan_dict, strategy)
                elif signal.direction in ['sell', 'close']:
                    self._execute_sell(signal, chan_dict, strategy)
            except Exception as e:
                print(f"执行信号失败 {signal}: {e}")

    def _execute_buy(self, signal: CSignal, chan_dict: Dict[str, CChan], strategy: CStrategy):
        """执行买入信号"""

        code = signal.code
        if code not in chan_dict:
            return

        chan = chan_dict[code]
        if len(chan[0]) == 0:
            return

        # 获取价格（下一根K线开盘价/收盘价）
        if self.config.match_mode == "current_close":
            # 当前K线收盘价（存在未来函数风险）
            price = chan[0][-1][-1].close
        else:
            # 简化处理：使用当前收盘价作为下一根开盘价的近似
            price = chan[0][-1][-1].close

        # 计算买入数量
        if signal.volume is not None:
            volume = signal.volume
        else:
            # 按仓位百分比计算
            total_value = self.position_manager.get_total_value()
            target_value = total_value * signal.percent
            volume = int(target_value / price / 100) * 100  # 取整到100股

        if volume == 0:
            return

        # 计算成本
        actual_price, commission, tax, total_cost = self.config.calculate_total_cost(
            price, volume, 'buy'
        )

        # 检查是否可以买入
        if not self.position_manager.can_buy(code, actual_price, volume, total_cost):
            return

        # 执行买入
        self.position_manager.buy(code, volume, actual_price, commission, self.current_time)

        # 记录交易
        trade = CTrade(
            code=code,
            direction='buy',
            volume=volume,
            price=actual_price,
            time=self.current_time,
            datetime=datetime.now(),
            commission=commission,
            tax=tax,
            slippage=abs(actual_price - price) * volume,
            reason=signal.reason,
        )
        self.result.add_trade(trade)

        # 调用策略回调
        strategy.on_trade(trade)

        if self.config.print_progress:
            print(f"  {trade}")

    def _execute_sell(self, signal: CSignal, chan_dict: Dict[str, CChan], strategy: CStrategy):
        """执行卖出信号"""

        code = signal.code
        if code not in chan_dict:
            return

        chan = chan_dict[code]
        if len(chan[0]) == 0:
            return

        # 检查是否有持仓
        if not self.position_manager.has_position(code):
            return

        # 获取价格
        if self.config.match_mode == "current_close":
            price = chan[0][-1][-1].close
        else:
            price = chan[0][-1][-1].close

        # 计算卖出数量
        position = self.position_manager.get_position(code)
        if signal.volume is not None:
            volume = min(signal.volume, position.available)
        else:
            # 按仓位百分比计算
            volume = int(position.available * signal.percent / 100) * 100

        if volume == 0:
            return

        # 计算成本
        actual_price, commission, tax, total_cost = self.config.calculate_total_cost(
            price, volume, 'sell'
        )

        # 检查是否可以卖出
        if not self.position_manager.can_sell(code, volume):
            return

        # 执行卖出
        realized_profit, realized_profit_rate = self.position_manager.sell(
            code, volume, actual_price, commission, tax
        )

        # 记录交易
        trade = CTrade(
            code=code,
            direction='sell',
            volume=volume,
            price=actual_price,
            time=self.current_time,
            datetime=datetime.now(),
            commission=commission,
            tax=tax,
            slippage=abs(actual_price - price) * volume,
            reason=signal.reason,
            profit=realized_profit,
            profit_rate=realized_profit_rate,
        )
        self.result.add_trade(trade)

        # 调用策略回调
        strategy.on_trade(trade)

        if self.config.print_progress:
            print(f"  {trade}")

    def _calculate_performance(self):
        """计算绩效指标"""
        from Backtest.Performance import CPerformance

        performance = CPerformance(self.result, self.config)
        self.result.metrics = performance.calculate_metrics()

        # 打印绩效摘要
        self._print_performance_summary()

    def _print_performance_summary(self):
        """打印绩效摘要"""
        metrics = self.result.metrics

        print("\n" + "="*60)
        print("绩效摘要".center(60))
        print("="*60)

        print(f"\n【收益指标】")
        print(f"  初始资金:     {self.config.initial_capital:>12.2f}")
        print(f"  最终资产:     {metrics.get('final_value', 0):>12.2f}")
        print(f"  累计收益:     {metrics.get('total_profit', 0):>12.2f}")
        print(f"  累计收益率:   {metrics.get('total_return', 0)*100:>11.2f}%")
        print(f"  年化收益率:   {metrics.get('annual_return', 0)*100:>11.2f}%")

        print(f"\n【风险指标】")
        print(f"  最大回撤:     {metrics.get('max_drawdown', 0)*100:>11.2f}%")
        print(f"  夏普比率:     {metrics.get('sharpe_ratio', 0):>12.2f}")

        print(f"\n【交易统计】")
        print(f"  总交易次数:   {metrics.get('trade_count', 0):>12}")
        print(f"  胜率:         {metrics.get('win_rate', 0)*100:>11.2f}%")
        print(f"  盈亏比:       {metrics.get('profit_loss_ratio', 0):>12.2f}")
        print(f"  平均持仓天数: {metrics.get('avg_hold_days', 0):>12.1f}")

        print("="*60 + "\n")
