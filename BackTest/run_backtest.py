import os
import sys
import backtrader as bt
from datetime import datetime
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from strategies.chan_strategy import ChanStrategy

def run_backtest(
    code="sz.300896",
    start_date='2024-06-01',
    end_date=None,
    cash=100000.0,
    commission=0.001,
    position_size=1.0,
    buy_type='3a',
    sell_type='3b'
):
    """
    运行回测
    :param code: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param cash: 初始资金
    :param commission: 手续费率
    :param position_size: 仓位大小(0-1)
    :param buy_type: 买入信号类型
    :param sell_type: 卖出信号类型
    """
    # 创建cerebro引擎
    cerebro = bt.Cerebro()
    
    # 加载数据
    data_loader = DataLoader(data_dir='BackTest/data')
    data = data_loader.get_stock_data(
        code=code,
        start_date=start_date,
        end_date=end_date,
        force_download=True,
        frequency='30'  # 使用30分钟K线
    )
    
    # 转换为backtrader数据格式
    feed = bt.feeds.PandasData(
        dataname=data,
        name=code,
        datetime=None,  # 使用索引作为日期时间
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    cerebro.adddata(feed)
    
    # 设置初始资金
    cerebro.broker.setcash(cash)
    
    # 设置手续费
    cerebro.broker.setcommission(commission=commission)
    
    # 添加策略
    cerebro.addstrategy(
        ChanStrategy,
        begin_time=start_date,
        end_time=end_date,
        buy_type=buy_type,
        sell_type=sell_type,
        position_size=position_size
    )
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, stddev_sample=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 运行回测
    print(f'初始资金: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    
    # 输出分析结果
    print(f'最终资金: {cerebro.broker.getvalue():.2f}')
    print(f'总收益率: {strat.analyzers.returns.get_analysis()["rtot"]:.2%}')
    
    # 处理夏普比率
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()
    if sharpe_ratio and "sharperatio" in sharpe_ratio and sharpe_ratio["sharperatio"] is not None:
        print(f'夏普比率: {sharpe_ratio["sharperatio"]:.2f}')
    else:
        print('夏普比率: 0.00')
    
    print(f'最大回撤: {strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]:.2%}')
    
    # 保存交易记录
    if strat.trades:
        trades_df = pd.DataFrame(strat.trades)
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        trades_df.to_csv(
            os.path.join(results_dir, f'trades_{code}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'),
            index=False
        )
        
        # 打印交易统计
        print("\n交易统计:")
        print(f"总交易次数: {len(strat.trades)}")
        win_trades = [t for t in strat.trades if t['pnl'] > 0]
        print(f"胜率: {len(win_trades)/len(strat.trades):.2%}")
        print(f"平均收益: {sum(t['pnl'] for t in strat.trades)/len(strat.trades):.2f}")
    
if __name__ == "__main__":
    # 运行回测示例
    run_backtest(
        code="sz.300896",        # 平安银行
        start_date="2024-09-01", # 起始日期，使用更短的时间范围进行测试
        end_date="2024-12-31",   # 结束日期
        cash=100000.0,           # 初始资金
        commission=0.001,        # 手续费率
        position_size=1.0,       # 仓位大小
        buy_type='1',          # 买入信号类型
        sell_type='3b'          # 卖出信号类型
    )
