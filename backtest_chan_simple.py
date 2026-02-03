"""
简化版缠论买卖点策略回测脚本

使用更宽松的参数设置以确保能够产生交易信号
"""

from Backtest.BacktestConfig import CBacktestConfig
from Backtest.BacktestEngine import CBacktestEngine
from Backtest.Strategy import CBSPStrategy
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
import json
from datetime import datetime
import pandas as pd


def run_single_backtest(config_name, chan_config, strategy_params):
    """运行单次回测"""

    print(f"\n{'='*80}")
    print(f"配置: {config_name}")
    print(f"{'='*80}")

    # 回测配置
    backtest_config = CBacktestConfig(
        initial_capital=100000.0,
        begin_time="2018-01-01",
        end_time="2023-12-31",
        data_src=DATA_SRC.BAO_STOCK,
        lv_list=[KL_TYPE.K_DAY],
        chan_config=chan_config,
        commission_rate=0.0003,
        slippage_rate=0.001,
        stamp_tax_rate=0.001,
        print_progress=False,  # 关闭详细进度以加快速度
        progress_interval=100,
    )

    # 创建策略
    strategy = CBSPStrategy(
        name=config_name,
        **strategy_params
    )

    # 创建回测引擎
    engine = CBacktestEngine(backtest_config)

    # 运行回测
    code_list = ["sz.000001"]  # 平安银行（交易更活跃）

    try:
        result = engine.run(strategy, code_list)

        # 检查是否有交易
        if result.metrics.get('trade_count', 0) == 0:
            print(f"  ⚠️  警告: 该配置未产生任何交易")

        return {
            'config_name': config_name,
            'chan_config': chan_config,
            'strategy_params': strategy_params,
            'metrics': result.metrics,
            'trade_count': result.metrics.get('trade_count', 0),
            'success': True
        }
    except Exception as e:
        print(f"  ✗ 回测失败: {e}")
        return {
            'config_name': config_name,
            'chan_config': chan_config,
            'strategy_params': strategy_params,
            'error': str(e),
            'success': False
        }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略回测系统".center(80))
    print("="*80)

    # 定义测试配置（从宽松到严格）
    test_configs = [
        {
            "name": "宽松-低背驰-多买卖点类型",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.2,  # 很宽松的背驰率
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s,3a",  # 包含更多买卖点类型
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.10,
                "take_profit": 0.30,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "标准-中等背驰-一二类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "标准-中等背驰-仅一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "严格-高背驰-仅一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.7,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": True,
            },
            "strategy_params": {
                "buy_percent": 0.35,
                "stop_loss": -0.06,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "严格-高背驰-多中枢",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.6,
                "min_zs_cnt": 2,
                "bs_type": "1,1p",
                "bi_strict": True,
                "zs_combine": True,
            },
            "strategy_params": {
                "buy_percent": 0.4,
                "stop_loss": -0.05,
                "take_profit": 0.18,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "激进-极宽松-所有买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.5,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s,3a,3b",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.2,
                "stop_loss": -0.12,
                "take_profit": 0.35,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
    ]

    # 运行所有配置
    all_results = []
    for config in test_configs:
        result = run_single_backtest(
            config_name=config['name'],
            chan_config=config['chan_config'],
            strategy_params=config['strategy_params']
        )
        all_results.append(result)

    # 筛选成功的结果
    successful_results = [r for r in all_results if r.get('success', False) and r.get('trade_count', 0) > 0]

    if not successful_results:
        print("\n❌ 所有配置都未产生交易或失败！")
        print("\n尝试的配置:")
        for r in all_results:
            print(f"  - {r['config_name']}: ", end="")
            if not r.get('success'):
                print(f"失败 ({r.get('error', '未知错误')})")
            else:
                print(f"成功但无交易 (交易次数: {r.get('trade_count', 0)})")
        return

    # 按年化收益率排序
    successful_results.sort(
        key=lambda x: x['metrics'].get('annual_return', -999),
        reverse=True
    )

    print("\n" + "="*80)
    print("回测结果汇总".center(80))
    print("="*80)

    # 创建结果表格
    summary_data = []
    for r in successful_results:
        metrics = r['metrics']
        summary_data.append({
            '配置名称': r['config_name'],
            '年化收益率': f"{metrics.get('annual_return', 0)*100:.2f}%",
            '累计收益率': f"{metrics.get('total_return', 0)*100:.2f}%",
            '最大回撤': f"{metrics.get('max_drawdown', 0)*100:.2f}%",
            '夏普比率': f"{metrics.get('sharpe_ratio', 0):.2f}",
            '胜率': f"{metrics.get('win_rate', 0)*100:.2f}%",
            '交易次数': metrics.get('trade_count', 0),
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 生成最佳配置报告
    best_result = successful_results[0]

    report_content = f"""# 缠论买卖点策略回测最佳配置报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 最佳配置

### 配置名称
**{best_result['config_name']}**

### 时间周期
**日线 (K_DAY)**

### 缠论配置参数
```json
{json.dumps(best_result['chan_config'], indent=2, ensure_ascii=False)}
```

**参数说明:**
- `trigger_step`: True - 启用逐步回测模式
- `divergence_rate`: {best_result['chan_config'].get('divergence_rate')} - 背驰率阈值（越小越严格）
- `min_zs_cnt`: {best_result['chan_config'].get('min_zs_cnt')} - 最小中枢数量
- `bs_type`: "{best_result['chan_config'].get('bs_type')}" - 买卖点类型（1=一类, 1p=类一买, 2=二类, 2s=类二买, 3a/3b=三类）
- `bi_strict`: {best_result['chan_config'].get('bi_strict', False)} - 是否严格笔划分

### 策略参数
```json
{json.dumps({k: str(v) if isinstance(v, list) else v for k, v in best_result['strategy_params'].items()}, indent=2, ensure_ascii=False)}
```

**参数说明:**
- `buy_percent`: {best_result['strategy_params'].get('buy_percent')} - 每次买入仓位比例
- `stop_loss`: {best_result['strategy_params'].get('stop_loss')} - 止损比例
- `take_profit`: {best_result['strategy_params'].get('take_profit')} - 止盈比例
- `bsp_types`: 关注的买卖点类型列表

## 回测绩效

### 收益指标
- **初始资金**: 100,000.00 元
- **最终资产**: {best_result['metrics'].get('final_value', 0):,.2f} 元
- **累计收益**: {best_result['metrics'].get('total_profit', 0):,.2f} 元
- **累计收益率**: {best_result['metrics'].get('total_return', 0)*100:.2f}%
- **年化收益率**: {best_result['metrics'].get('annual_return', 0)*100:.2f}%

### 风险指标
- **最大回撤**: {best_result['metrics'].get('max_drawdown', 0)*100:.2f}%
- **夏普比率**: {best_result['metrics'].get('sharpe_ratio', 0):.2f}
- **波动率**: {best_result['metrics'].get('volatility', 0)*100:.2f}%

### 交易统计
- **总交易次数**: {best_result['metrics'].get('trade_count', 0)}
- **盈利次数**: {best_result['metrics'].get('win_count', 0)}
- **亏损次数**: {best_result['metrics'].get('loss_count', 0)}
- **胜率**: {best_result['metrics'].get('win_rate', 0)*100:.2f}%
- **盈亏比**: {best_result['metrics'].get('profit_loss_ratio', 0):.2f}
- **平均持仓天数**: {best_result['metrics'].get('avg_hold_days', 0):.1f}

## TOP 配置对比

| 排名 | 配置名称 | 年化收益率 | 累计收益率 | 最大回撤 | 夏普比率 | 胜率 | 交易次数 |
|------|---------|-----------|-----------|---------|---------|------|---------|
"""

    for i, r in enumerate(successful_results[:min(5, len(successful_results))], 1):
        m = r['metrics']
        report_content += f"| {i} | {r['config_name']} | {m.get('annual_return', 0)*100:.2f}% | {m.get('total_return', 0)*100:.2f}% | {m.get('max_drawdown', 0)*100:.2f}% | {m.get('sharpe_ratio', 0):.2f} | {m.get('win_rate', 0)*100:.2f}% | {m.get('trade_count', 0)} |\n"

    report_content += f"""
## 完整结果列表

共测试 {len(all_results)} 个配置组合，成功产生交易 {len(successful_results)} 个。

{df.to_markdown(index=False) if len(successful_results) > 0 else "无有效结果"}

## 配置建议

### 1. 时间周期选择
- **推荐**: 日线 (K_DAY)
- **原因**: 数据稳定性好，信号可靠性高

### 2. 缠论参数优化

**背驰率 (divergence_rate)**
- 最佳值: {best_result['chan_config'].get('divergence_rate')}
- 建议范围: 0.7 - 1.2
- 说明: 数值越小越严格，产生的信号越少但质量越高

**中枢数量 (min_zs_cnt)**
- 最佳值: {best_result['chan_config'].get('min_zs_cnt')}
- 建议: 1-2个中枢
- 说明: 中枢数量越多越稳健，但信号会减少

**买卖点类型 (bs_type)**
- 最佳值: {best_result['chan_config'].get('bs_type')}
- 建议: 优先使用一类买卖点(1, 1p)，可适当加入二类(2, 2s)

### 3. 风险控制

**止损比例**
- 最佳值: {best_result['strategy_params'].get('stop_loss')}
- 建议范围: -5% 到 -10%

**止盈比例**
- 最佳值: {best_result['strategy_params'].get('take_profit')}
- 建议范围: 15% 到 30%

**单次仓位**
- 最佳值: {best_result['strategy_params'].get('buy_percent')}
- 建议范围: 15% 到 35%

## 实战建议

1. **参数调整**: 根据市场环境动态调整背驰率和止盈止损比例
2. **仓位管理**: 建议实盘时适当降低仓位比例
3. **标的选择**: 优先选择流动性好、波动适中的股票
4. **风险提示**: 回测结果基于历史数据，实盘效果可能存在偏差

## 数据说明

- **回测标的**: 平安银行 (sz.000001)
- **回测区间**: 2018-01-01 至 2023-12-31
- **初始资金**: 100,000 元
- **交易成本**: 手续费万三 + 滑点千一 + 印花税千一
- **数据源**: BaoStock
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点策略最佳配置报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 最佳配置报告已保存至: {report_path}")

    # 保存详细结果到JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_results.json"
    serializable_results = []
    for r in all_results:
        r_copy = r.copy()
        if 'strategy_params' in r_copy:
            r_copy['strategy_params'] = {
                k: str(v) if isinstance(v, list) else v
                for k, v in r_copy['strategy_params'].items()
            }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 详细结果已保存至: {json_path}")

    print(f"\n{'='*80}")
    print("回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
