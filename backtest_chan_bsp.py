"""
缠论买卖点策略回测脚本

测试不同的缠论配置参数和时间周期，找出最佳配置
"""

from Backtest.BacktestConfig import CBacktestConfig
from Backtest.BacktestEngine import CBacktestEngine
from Backtest.Strategy import CBSPStrategy
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
import json
from datetime import datetime
import pandas as pd


def run_backtest_with_config(config_name, timeframe, chan_config, strategy_params):
    """
    运行单次回测

    Args:
        config_name: 配置名称
        timeframe: 时间周期
        chan_config: 缠论配置
        strategy_params: 策略参数
    """
    print(f"\n{'='*80}")
    print(f"配置: {config_name} | 周期: {timeframe.value}")
    print(f"{'='*80}")

    # 回测配置
    backtest_config = CBacktestConfig(
        initial_capital=100000.0,
        begin_time="2021-01-01",
        end_time="2024-01-01",
        data_src=DATA_SRC.BAO_STOCK,
        lv_list=[timeframe],
        chan_config=chan_config,
        commission_rate=0.0003,
        slippage_rate=0.001,
        stamp_tax_rate=0.001,
        print_progress=True,
        progress_interval=50,
    )

    # 创建策略
    strategy = CBSPStrategy(
        name=f"{config_name}_{timeframe.value}",
        **strategy_params
    )

    # 创建回测引擎
    engine = CBacktestEngine(backtest_config)

    # 运行回测
    code_list = ["sh.600519"]  # 贵州茅台

    try:
        result = engine.run(strategy, code_list)

        return {
            'config_name': config_name,
            'timeframe': timeframe.value,
            'chan_config': chan_config,
            'strategy_params': strategy_params,
            'metrics': result.metrics,
            'success': True
        }
    except Exception as e:
        print(f"回测失败: {e}")
        return {
            'config_name': config_name,
            'timeframe': timeframe.value,
            'chan_config': chan_config,
            'strategy_params': strategy_params,
            'error': str(e),
            'success': False
        }


def main():
    """主函数：测试多种配置组合"""

    # 定义测试的时间周期（只使用日线，因为数据源限制）
    timeframes = [
        KL_TYPE.K_DAY,   # 日线
    ]

    # 定义多种缠论配置
    chan_configs = [
        {
            "name": "标准配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
            }
        },
        {
            "name": "严格配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.7,
                "min_zs_cnt": 2,
                "bs_type": "1,1p",
                "bi_strict": True,
            }
        },
        {
            "name": "宽松配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 1.0,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s,3a",
                "bi_strict": False,
            }
        },
        {
            "name": "高背驰配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.6,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2",
                "macd_algo": "area",
            }
        },
        {
            "name": "多中枢配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 2,
                "bs_type": "1,1p,2,2s",
                "zs_combine": True,
            }
        },
    ]

    # 定义多种策略参数
    strategy_params_list = [
        {
            "name": "保守策略",
            "params": {
                "buy_percent": 0.15,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "平衡策略",
            "params": {
                "buy_percent": 0.2,
                "stop_loss": -0.05,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "激进策略",
            "params": {
                "buy_percent": 0.3,
                "stop_loss": -0.03,
                "take_profit": 0.15,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
    ]

    # 存储所有结果
    all_results = []

    # 遍历所有组合
    for timeframe in timeframes:
        for chan_config_dict in chan_configs:
            for strategy_dict in strategy_params_list:
                config_name = f"{chan_config_dict['name']}_{strategy_dict['name']}"

                result = run_backtest_with_config(
                    config_name=config_name,
                    timeframe=timeframe,
                    chan_config=chan_config_dict['config'],
                    strategy_params=strategy_dict['params']
                )

                all_results.append(result)

    # 筛选成功的结果
    successful_results = [r for r in all_results if r.get('success', False)]

    if not successful_results:
        print("\n所有回测都失败了！")
        return

    # 按年化收益率排序
    successful_results.sort(
        key=lambda x: x['metrics'].get('annual_return', -999),
        reverse=True
    )

    # 生成结果报告
    print("\n" + "="*80)
    print("回测结果汇总".center(80))
    print("="*80)

    # 创建DataFrame用于展示
    summary_data = []
    for r in successful_results:
        metrics = r['metrics']
        summary_data.append({
            '配置名称': r['config_name'],
            '时间周期': r['timeframe'],
            '年化收益率': f"{metrics.get('annual_return', 0)*100:.2f}%",
            '累计收益率': f"{metrics.get('total_return', 0)*100:.2f}%",
            '最大回撤': f"{metrics.get('max_drawdown', 0)*100:.2f}%",
            '夏普比率': f"{metrics.get('sharpe_ratio', 0):.2f}",
            '胜率': f"{metrics.get('win_rate', 0)*100:.2f}%",
            '交易次数': metrics.get('trade_count', 0),
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 输出最佳配置到文档
    best_result = successful_results[0]

    report_content = f"""# 缠论买卖点策略回测最佳配置报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 最佳配置

### 配置名称
{best_result['config_name']}

### 时间周期
{best_result['timeframe']}

### 缠论配置参数
```json
{json.dumps(best_result['chan_config'], indent=2, ensure_ascii=False)}
```

### 策略参数
```json
{json.dumps({k: str(v) if isinstance(v, list) else v for k, v in best_result['strategy_params'].items()}, indent=2, ensure_ascii=False)}
```

## 回测绩效

### 收益指标
- 初始资金: 100,000.00 元
- 最终资产: {best_result['metrics'].get('final_value', 0):,.2f} 元
- 累计收益: {best_result['metrics'].get('total_profit', 0):,.2f} 元
- 累计收益率: {best_result['metrics'].get('total_return', 0)*100:.2f}%
- 年化收益率: {best_result['metrics'].get('annual_return', 0)*100:.2f}%

### 风险指标
- 最大回撤: {best_result['metrics'].get('max_drawdown', 0)*100:.2f}%
- 夏普比率: {best_result['metrics'].get('sharpe_ratio', 0):.2f}
- 波动率: {best_result['metrics'].get('volatility', 0)*100:.2f}%

### 交易统计
- 总交易次数: {best_result['metrics'].get('trade_count', 0)}
- 盈利次数: {best_result['metrics'].get('win_count', 0)}
- 亏损次数: {best_result['metrics'].get('loss_count', 0)}
- 胜率: {best_result['metrics'].get('win_rate', 0)*100:.2f}%
- 盈亏比: {best_result['metrics'].get('profit_loss_ratio', 0):.2f}
- 平均持仓天数: {best_result['metrics'].get('avg_hold_days', 0):.1f}

## TOP 5 配置对比

| 排名 | 配置名称 | 时间周期 | 年化收益率 | 最大回撤 | 夏普比率 | 胜率 |
|------|---------|---------|-----------|---------|---------|------|
"""

    for i, r in enumerate(successful_results[:5], 1):
        m = r['metrics']
        report_content += f"| {i} | {r['config_name']} | {r['timeframe']} | {m.get('annual_return', 0)*100:.2f}% | {m.get('max_drawdown', 0)*100:.2f}% | {m.get('sharpe_ratio', 0):.2f} | {m.get('win_rate', 0)*100:.2f}% |\n"

    report_content += f"""
## 完整结果列表

共测试 {len(all_results)} 个配置组合，成功 {len(successful_results)} 个。

{df.to_markdown(index=False)}

## 配置建议

1. **时间周期选择**: {best_result['timeframe']}
   - 该周期在本次回测中表现最佳

2. **缠论参数优化**:
   - divergence_rate: {best_result['chan_config'].get('divergence_rate', 'N/A')}
   - min_zs_cnt: {best_result['chan_config'].get('min_zs_cnt', 'N/A')}
   - bs_type: {best_result['chan_config'].get('bs_type', 'N/A')}

3. **风险控制**:
   - 止损比例: {best_result['strategy_params'].get('stop_loss', 'N/A')}
   - 止盈比例: {best_result['strategy_params'].get('take_profit', 'N/A')}
   - 单次仓位: {best_result['strategy_params'].get('buy_percent', 'N/A')}

4. **注意事项**:
   - 回测结果基于历史数据，实盘效果可能有差异
   - 建议先进行小资金验证
   - 定期根据市场情况调整参数

## 数据说明

- 回测标的: 贵州茅台 (sh.600519)
- 回测区间: 2020-01-01 至 2024-12-31
- 初始资金: 100,000 元
- 交易成本: 手续费万三 + 滑点千一 + 印花税千一
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点策略最佳配置报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✓ 最佳配置报告已保存至: {report_path}")

    # 保存详细结果到JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # 转换不可序列化的对象
        serializable_results = []
        for r in all_results:
            r_copy = r.copy()
            if 'strategy_params' in r_copy:
                r_copy['strategy_params'] = {
                    k: str(v) if isinstance(v, list) else v
                    for k, v in r_copy['strategy_params'].items()
                }
            serializable_results.append(r_copy)

        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"✓ 详细结果已保存至: {json_path}")

    print(f"\n{'='*80}")
    print("回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
