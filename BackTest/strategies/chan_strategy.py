import backtrader as bt
from Chan import CChan, CChanConfig
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, DATA_FIELD
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit

class ChanStrategy(bt.Strategy):
    """
    缠论三买策略
    """
    params = (
        ('begin_time', '2018-01-01'),
        ('end_time', '2023-12-31'),
        ('buy_type', '3a'),      # 买入信号类型
        ('sell_type', '3b'),     # 卖出信号类型
        ('position_size', 1.0),  # 仓位大小(0-1)
    )

    def __init__(self):
        self.order = None
        self.chan = CChan(
            code=self.data._name,  # 股票代码
            begin_time=self.params.begin_time,
            end_time=self.params.end_time,
            data_src=DATA_SRC.BAO_STOCK,
            lv_list=[KL_TYPE.K_30M],
            config=CChanConfig({
                "bi_strict": True,
                "trigger_step": True,  # 改为True，使用step_load逐步加载
                "skip_step": 0,
                "divergence_rate": float("inf"),
                "bsp2_follow_1": True,  # 二买必须跟随一买
                "bsp3_follow_1": True,  # 三买必须跟随一买
                "min_zs_cnt": 0,
                "bs1_peak": False,
                "macd_algo": "peak",
                "bs_type": self.params.buy_type,
                "print_warning": True,  # 打印警告信息
                "zs_algo": "normal",
            })
        )
        
        # 记录交易
        self.trades = []
        
        # 打印初始化信息
        self.log(f'策略初始化: 股票代码={self.data._name}, 买入类型={self.params.buy_type}, 卖出类型={self.params.sell_type}')
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
            else:
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
                
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.trades.append({
            'open_date': bt.num2date(trade.dtopen),
            'close_date': bt.num2date(trade.dtclose),
            'open_price': trade.price,
            'close_price': trade.pnlcomm / trade.size + trade.price,
            'pnl': trade.pnlcomm,
            'size': trade.size
        })
        
        self.log(f'交易利润: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}')
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()}, {txt}')
        
    def next(self):
        # 如果有订单正在执行，不操作
        if self.order:
            return

        # 获取当前K线的数据
        current_datetime = self.data.datetime.datetime(0)
        kline_dict = {
            DATA_FIELD.FIELD_TIME: CTime(current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.hour, current_datetime.minute),
            DATA_FIELD.FIELD_OPEN: self.data.open[0],
            DATA_FIELD.FIELD_HIGH: self.data.high[0],
            DATA_FIELD.FIELD_LOW: self.data.low[0],
            DATA_FIELD.FIELD_CLOSE: self.data.close[0],
            DATA_FIELD.FIELD_VOLUME: self.data.volume[0],
        }
        
        # 创建K线对象
        current_kline = CKLine_Unit(kline_dict)
        
        # 输入当前K线到Chan对象
        self.chan.trigger_load({KL_TYPE.K_30M: [current_kline]})
        
        # 获取买卖点信号
        signals = self.chan.get_bsp()
        
        # 处理买卖点信号
        if signals and len(signals) > 0:
            # 只处理当前K线产生的买卖点
            current_signals = [s for s in signals if s.klu.time == current_kline.time]
            
            if current_signals:  # 如果当前K线有买卖点
                last_signal = current_signals[-1]  # 获取最新的买卖点
                self.log(f'当前K线产生买卖点信号: {last_signal}')
                
                # 判断是否为买点
                if last_signal.is_buy:
                    # 检查是否为指定的买点类型
                    signal_types = [t.value for t in last_signal.type]
                    if self.params.buy_type in signal_types:
                        if not self.position:  # 没有持仓时才买入
                            size = self.broker.getcash() * self.params.position_size / self.data.close[0]
                            self.log(f'发现买入信号 {self.params.buy_type}, 下单数量: {size:.2f}')
                            self.order = self.buy(size=size)  # 买入
                        else:
                            self.log(f'已有持仓，忽略买入信号 {self.params.buy_type}')
                # 判断是否为卖点
                else:
                    # 检查是否为指定的卖点类型
                    signal_types = [t.value for t in last_signal.type]
                    if self.params.sell_type in signal_types:
                        if self.position:  # 有持仓时才卖出
                            self.log(f'发现卖出信号 {self.params.sell_type}, 准备卖出')
                            self.order = self.sell()  # 卖出
                        else:
                            self.log(f'无持仓，忽略卖出信号 {self.params.sell_type}')
            else:
                self.log('当前K线未产生买卖点信号')
        else:
            self.log('当前无买卖点信号')
                
    def stop(self):
        # 回测结束时的汇总
        if len(self.trades) > 0:
            total_pnl = sum(trade['pnl'] for trade in self.trades)
            win_trades = [trade for trade in self.trades if trade['pnl'] > 0]
            win_rate = len(win_trades) / len(self.trades) if self.trades else 0
            
            self.log(f'回测结束: 总交易次数={len(self.trades)}, 胜率={win_rate:.2%}, 总收益={total_pnl:.2f}')
