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
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
from strategy_configs import get_config, BACKTEST_CONFIG


def main():
    """主函数"""

    # 使用预配置的最优平衡配置
    strategy_config = get_config("balanced")

    # 配置回测参数
    config = CBacktestConfig(
        initial_capital=BACKTEST_CONFIG["initial_capital"],
        commission_rate=BACKTEST_CONFIG["commission_rate"],
        slippage_rate=BACKTEST_CONFIG["slippage_rate"],
        stamp_tax_rate=BACKTEST_CONFIG["stamp_tax_rate"],
        max_position_per_stock=BACKTEST_CONFIG["max_position_per_stock"],
        max_total_position=BACKTEST_CONFIG["max_total_position"],
        begin_time=BACKTEST_CONFIG["begin_time"],
        end_time=BACKTEST_CONFIG["end_time"],
        data_src=DATA_SRC.AKSHARE,
        lv_list=[KL_TYPE.K_DAY],
        chan_config=BACKTEST_CONFIG["chan_config"],
        print_progress=BACKTEST_CONFIG["print_progress"],
        progress_interval=BACKTEST_CONFIG["progress_interval"],
    )

    # 创建策略
    strategy = CBSPStrategy(**strategy_config["strategy_params"])

    # 创建回测引擎
    engine = CBacktestEngine(config)

    # 股票池
    stock_list = strategy_config["stock_list"]

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
