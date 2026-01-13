"""
回测示例 - 使用一类买卖点策略

演示如何使用回测系统进行策略测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Backtest.BacktestEngine import CBacktestEngine
from Backtest.BacktestConfig import CBacktestConfig
from Backtest.Strategy import CBSPStrategy
from Common.CEnum import DATA_SRC, KL_TYPE


def main():
    """主函数"""

    # 配置回测参数
    config = CBacktestConfig(
        initial_capital=100000.0,      # 初始资金10万
        commission_rate=0.0003,        # 万三手续费
        slippage_rate=0.001,           # 0.1%滑点
        stamp_tax_rate=0.001,          # 千一印花税
        max_position_per_stock=0.3,    # 单只最大30%仓位
        max_total_position=0.95,       # 总仓位95%
        begin_time="2023-01-01",       # 回测开始时间
        end_time="2024-12-31",         # 回测结束时间
        data_src=DATA_SRC.AKSHARE,
        lv_list=[KL_TYPE.K_DAY],
        chan_config={
            "trigger_step": True,      # 开启逐步回测
            "bi_strict": True,         # 笔严格模式
            "divergence_rate": 0.9,    # 背驰比例
            "bs_type": "1,1p,2",       # 关注的买卖点类型
            "print_warning": False,    # 不打印警告
        },
        print_progress=True,
        progress_interval=50,
    )

    # 创建策略
    strategy = CBSPStrategy(
        name="一类买卖点策略",
        buy_percent=0.2,  # 每次买入20%仓位
    )

    # 创建回测引擎
    engine = CBacktestEngine(config)

    # 运行回测（测试单只股票）
    stock_list = ["000001"]  # 平安银行
    # stock_list = ["000001", "600519", "000858"]  # 多只股票测试

    try:
        result = engine.run(strategy, stock_list)

        # 绘制权益曲线
        print("\n正在生成图表...")
        from Backtest.Performance import CPerformance
        performance = CPerformance(result, config)

        # 保存图表
        performance.plot_equity_curve("backtest_equity_curve.png")
        performance.plot_drawdown("backtest_drawdown.png")

        # 打印详细交易记录
        print("\n" + "="*60)
        print("详细交易记录".center(60))
        print("="*60)
        for i, trade in enumerate(result.trades, 1):
            print(f"{i}. {trade}")

        print("\n回测完成！")

    except Exception as e:
        print(f"\n回测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
