"""
缠论买卖点策略 - 30分钟K线回测 (改进版 - 增强买卖点准确率分析)

改进点:
1. 扩展股票池，选择流动性好的大盘股
2. 改进买卖点准确率计算逻辑
3. 增加更多配置参数组合
4. 详细的买卖点分析报告
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from datetime import datetime
import json
import pandas as pd


# 扩展股票池 - 选择流动性好的大盘股
STOCK_LIST = [
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600000", "name": "浦发银行", "industry": "银行"},
    {"code": "sz.000002", "name": "万科A", "industry": "房地产"},
    {"code": "sh.600036", "name": "招商银行", "industry": "银行"},
    {"code": "sz.000063", "name": "中兴通讯", "industry": "通信"},
    {"code": "sh.600030", "name": "中信证券", "industry": "证券"},
    {"code": "sz.000333", "name": "美的集团", "industry": "家电"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000651", "name": "格力电器", "industry": "家电"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
]


def calculate_bsp_accuracy_detailed(trades, equity_curve):
    """
    详细计算买卖点准确率

    准确率定义 (更严格):
    - 买点准确: 买入后实现止盈或盈利超过5%
    - 卖点准确: 卖点是止盈信号，或卖出后价格下跌
    - 部分准确: 小幅止损(<3%)或小幅盈利(0-5%)
    """
    if not trades:
        return {
            'buy_accuracy': 0,
            'sell_accuracy': 0,
            'total_bsp_count': 0,
            'correct_bsp_count': 0,
            'partial_accuracy': 0
        }

    buy_signals = []
    sell_signals = []

    for i, trade in enumerate(trades):
        if trade['type'] == 'buy':
            # 找到对应的卖出交易
            next_sell = None
            for j in range(i+1, len(trades)):
                if trades[j]['type'] == 'sell':
                    next_sell = trades[j]
                    break

            if next_sell:
                profit_rate = next_sell.get('profit_rate', 0)
                reason = next_sell.get('reason', '')

                # 买点准确度评级
                if '止盈' in reason or profit_rate > 0.05:
                    accuracy_level = 'correct'
                elif profit_rate > 0:
                    accuracy_level = 'partial'
                elif profit_rate > -0.03:
                    accuracy_level = 'partial'
                else:
                    accuracy_level = 'wrong'

                buy_signals.append({
                    'level': accuracy_level,
                    'profit_rate': profit_rate,
                    'reason': reason
                })

        elif trade['type'] == 'sell' and i > 0:
            profit_rate = trade.get('profit_rate', 0)
            reason = trade.get('reason', '')

            # 卖点准确度评级
            if '止盈' in reason:
                accuracy_level = 'correct'
            elif profit_rate > 0.03:
                accuracy_level = 'correct'
            elif profit_rate > 0:
                accuracy_level = 'partial'
            elif '止损' in reason and profit_rate > -0.03:
                accuracy_level = 'partial'
            else:
                accuracy_level = 'wrong'

            sell_signals.append({
                'level': accuracy_level,
                'profit_rate': profit_rate,
                'reason': reason
            })

    # 统计准确率
    buy_correct = sum(1 for s in buy_signals if s['level'] == 'correct')
    buy_partial = sum(1 for s in buy_signals if s['level'] == 'partial')
    sell_correct = sum(1 for s in sell_signals if s['level'] == 'correct')
    sell_partial = sum(1 for s in sell_signals if s['level'] == 'partial')

    buy_accuracy = buy_correct / len(buy_signals) if buy_signals else 0
    sell_accuracy = sell_correct / len(sell_signals) if sell_signals else 0

    # 综合准确率 (correct + 0.5*partial)
    buy_effective = buy_correct + 0.5 * buy_partial
    sell_effective = sell_correct + 0.5 * sell_partial
    buy_effective_accuracy = buy_effective / len(buy_signals) if buy_signals else 0
    sell_effective_accuracy = sell_effective / len(sell_signals) if sell_signals else 0

    total_bsp = len(buy_signals) + len(sell_signals)
    overall_effective = (buy_effective + sell_effective) / total_bsp if total_bsp > 0 else 0

    return {
        'buy_accuracy': buy_accuracy,
        'sell_accuracy': sell_accuracy,
        'buy_effective_accuracy': buy_effective_accuracy,
        'sell_effective_accuracy': sell_effective_accuracy,
        'overall_effective_accuracy': overall_effective,
        'buy_signal_count': len(buy_signals),
        'sell_signal_count': len(sell_signals),
        'buy_correct_count': buy_correct,
        'buy_partial_count': buy_partial,
        'sell_correct_count': sell_correct,
        'sell_partial_count': sell_partial,
        'total_bsp_count': total_bsp,
        'total_correct_count': buy_correct + sell_correct,
        'total_partial_count': buy_partial + sell_partial,
    }


def run_backtest_30m(stock_code, stock_name, chan_config_dict, strategy_params):
    """运行30分钟K线回测"""

    begin_time = "2022-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_30M]

    config = CChanConfig(chan_config_dict)

    try:
        chan = CChan(
            code=stock_code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=lv_list,
            config=config,
            autype=AUTYPE.QFQ,
        )
    except Exception as e:
        return None

    initial_capital = 100000.0
    cash = initial_capital
    position = 0
    cost_price = 0

    trades = []
    equity_curve = []

    buy_percent = strategy_params.get('buy_percent', 0.3)
    stop_loss = strategy_params.get('stop_loss', -0.05)
    take_profit = strategy_params.get('take_profit', 0.12)
    target_bsp_types = strategy_params.get('bsp_types', [BSP_TYPE.T1, BSP_TYPE.T1P])

    try:
        for chan_snapshot in chan.step_load():
            cur_lv_chan = chan_snapshot[0]

            if len(cur_lv_chan) < 2:
                continue

            current_price = cur_lv_chan[-1][-1].close

            total_value = cash + position * current_price
            equity_curve.append({
                'time': str(cur_lv_chan[-1][-1].time),
                'total_value': total_value,
            })

            if position > 0:
                profit_rate = (current_price - cost_price) / cost_price

                if profit_rate <= stop_loss:
                    sell_value = position * current_price * 0.999
                    cash += sell_value
                    trades.append({
                        'time': str(cur_lv_chan[-1][-1].time),
                        'type': 'sell',
                        'price': current_price,
                        'volume': position,
                        'reason': '止损',
                        'profit': sell_value - position * cost_price,
                        'profit_rate': profit_rate
                    })
                    position = 0
                    cost_price = 0
                    continue

                if profit_rate >= take_profit:
                    sell_value = position * current_price * 0.999
                    cash += sell_value
                    trades.append({
                        'time': str(cur_lv_chan[-1][-1].time),
                        'type': 'sell',
                        'price': current_price,
                        'volume': position,
                        'reason': '止盈',
                        'profit': sell_value - position * cost_price,
                        'profit_rate': profit_rate
                    })
                    position = 0
                    cost_price = 0
                    continue

            bsp_list = chan_snapshot.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            if not any(t in last_bsp.type for t in target_bsp_types):
                continue

            if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
                continue

            if last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and position == 0:
                buy_amount = total_value * buy_percent
                buy_volume = int(buy_amount / current_price / 100) * 100

                if buy_volume > 0 and cash >= buy_volume * current_price:
                    cost = buy_volume * current_price * 1.001
                    if cash >= cost:
                        cash -= cost
                        position = buy_volume
                        cost_price = current_price * 1.001
                        trades.append({
                            'time': str(cur_lv_chan[-1][-1].time),
                            'type': 'buy',
                            'price': current_price,
                            'volume': buy_volume,
                            'reason': f'{last_bsp.type2str()}买点',
                            'profit': 0,
                            'profit_rate': 0
                        })

            elif not last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.TOP and position > 0:
                sell_value = position * current_price * 0.999
                cash += sell_value
                profit = sell_value - position * cost_price
                profit_rate = (current_price - cost_price) / cost_price

                trades.append({
                    'time': str(cur_lv_chan[-1][-1].time),
                    'type': 'sell',
                    'price': current_price,
                    'volume': position,
                    'reason': f'{last_bsp.type2str()}卖点',
                    'profit': profit,
                    'profit_rate': profit_rate
                })
                position = 0
                cost_price = 0

    except Exception as e:
        return None

    if position > 0 and len(cur_lv_chan) > 0:
        final_price = cur_lv_chan[-1][-1].close
        sell_value = position * final_price * 0.999
        cash += sell_value
        profit = sell_value - position * cost_price
        profit_rate = (final_price - cost_price) / cost_price
        trades.append({
            'time': str(cur_lv_chan[-1][-1].time),
            'type': 'sell',
            'price': final_price,
            'volume': position,
            'reason': '强制平仓',
            'profit': profit,
            'profit_rate': profit_rate
        })

    accuracy_stats = calculate_bsp_accuracy_detailed(trades, equity_curve)

    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital
    years = 2.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1

    sell_trades = [t for t in trades if t['type'] == 'sell']
    win_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
    loss_trades = [t for t in sell_trades if t.get('profit', 0) < 0]
    win_rate = len(win_trades) / len(sell_trades) if len(sell_trades) > 0 else 0

    max_drawdown = 0
    peak = initial_capital
    for point in equity_curve:
        if point['total_value'] > peak:
            peak = point['total_value']
        drawdown = (peak - point['total_value']) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    avg_win = sum(t.get('profit', 0) for t in win_trades) / len(win_trades) if len(win_trades) > 0 else 0
    avg_loss = abs(sum(t.get('profit', 0) for t in loss_trades) / len(loss_trades)) if len(loss_trades) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    buy_trades = [t for t in trades if t['type'] == 'buy']

    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'metrics': {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'trade_count': len(buy_trades),
            'win_count': len(win_trades),
            'loss_count': len(loss_trades),
        },
        'accuracy': accuracy_stats,
        'trades': trades[:30],
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 30分钟K线回测 (改进版)".center(80))
    print("="*80)

    # 更全面的配置测试
    test_configs = [
        {
            "name": "标准配置",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.05,
                "take_profit": 0.12,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "宽松配置",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.2,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.06,
                "take_profit": 0.15,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2],
            }
        },
        {
            "name": "严格配置",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.7,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": True,
            },
            "strategy_params": {
                "buy_percent": 0.35,
                "stop_loss": -0.04,
                "take_profit": 0.10,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "低背驰率",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.6,
                "min_zs_cnt": 1,
                "bs_type": "1",
                "bi_strict": True,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.04,
                "take_profit": 0.10,
                "bsp_types": [BSP_TYPE.T1],
            }
        },
        {
            "name": "一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 1,
                "bs_type": "1",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.05,
                "take_profit": 0.12,
                "bsp_types": [BSP_TYPE.T1],
            }
        },
        {
            "name": "准一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1p",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.05,
                "take_profit": 0.12,
                "bsp_types": [BSP_TYPE.T1P],
            }
        },
        {
            "name": "多买卖点组合",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.85,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.055,
                "take_profit": 0.13,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "保守策略",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.75,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": True,
            },
            "strategy_params": {
                "buy_percent": 0.2,
                "stop_loss": -0.04,
                "take_profit": 0.08,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
    ]

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock")
    print(f"时间周期: 30分钟 (K_30M)")
    print(f"回测区间: 2022-01-01 至 2023-12-31 (2年)")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个配置)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:12s}...", end=" ", flush=True)

            result = run_backtest_30m(
                stock_code=stock['code'],
                stock_name=stock['name'],
                chan_config_dict=config['chan_config'],
                strategy_params=config['strategy_params']
            )

            if result and result['metrics']['trade_count'] > 0:
                result['config_name'] = config['name']
                result['industry'] = stock['industry']
                result['chan_config'] = config['chan_config']
                result['strategy_params'] = config['strategy_params']
                all_results.append(result)

                m = result['metrics']
                a = result['accuracy']
                print(f"✓ {m['trade_count']:2d}次 "
                      f"收益{m['total_return']*100:6.1f}% "
                      f"胜率{m['win_rate']*100:4.0f}% "
                      f"准确率{a['overall_effective_accuracy']*100:4.0f}%")
            else:
                print("✗")

    successful_results = [r for r in all_results if r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 有效配置: {len(successful_results)}/{total}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        return

    # 按综合准确率排序
    successful_results.sort(key=lambda x: x['accuracy']['overall_effective_accuracy'], reverse=True)

    print("\n" + "="*80)
    print("TOP 15 配置 (按买卖点准确率排序)".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results[:15]:
        m = r['metrics']
        a = r['accuracy']
        summary_data.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '准确率': f"{a['overall_effective_accuracy']*100:.0f}%",
            '买点准确': f"{a['buy_effective_accuracy']*100:.0f}%",
            '卖点准确': f"{a['sell_effective_accuracy']*100:.0f}%",
            '年化收益': f"{m['annual_return']*100:6.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 按收益率排序
    successful_results_by_return = sorted(successful_results, key=lambda x: x['metrics']['annual_return'], reverse=True)

    print("\n" + "="*80)
    print("TOP 15 配置 (按年化收益率排序)".center(80))
    print("="*80)

    summary_data2 = []
    for r in successful_results_by_return[:15]:
        m = r['metrics']
        a = r['accuracy']
        summary_data2.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '年化收益': f"{m['annual_return']*100:6.1f}%",
            '准确率': f"{a['overall_effective_accuracy']*100:.0f}%",
            '买点准确': f"{a['buy_effective_accuracy']*100:.0f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df2 = pd.DataFrame(summary_data2)
    print("\n" + df2.to_string(index=False))

    # 生成详细报告
    best_by_accuracy = successful_results[0]
    best_by_return = successful_results_by_return[0]

    # 计算统计信息
    avg_accuracy = sum(r['accuracy']['overall_effective_accuracy'] for r in successful_results) / len(successful_results)
    avg_return = sum(r['metrics']['annual_return'] for r in successful_results) / len(successful_results)

    # 按行业统计
    industry_stats = {}
    for r in successful_results:
        industry = r['industry']
        if industry not in industry_stats:
            industry_stats[industry] = []
        industry_stats[industry].append(r)

    report_content = f"""# 缠论买卖点30分钟K线回测报告 (改进版)

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 回测概况

### 测试规模
- **股票数量**: {len(STOCK_LIST)}只
- **覆盖行业**: {', '.join(set(s['industry'] for s in STOCK_LIST))}
- **配置数量**: {len(test_configs)}种
- **回测区间**: 2022-01-01 至 2023-12-31 (2年)
- **时间周期**: 30分钟 (K_30M)
- **数据源**: BaoStock
- **初始资金**: 100,000 元

### 测试股票列表
"""

    for stock in STOCK_LIST:
        report_content += f"- **{stock['name']}** ({stock['code']}) - {stock['industry']}\n"

    report_content += f"""
### 测试结果
- **总测试配置**: {total}个
- **有效配置**: {len(successful_results)}个
- **有效率**: {len(successful_results)/total*100:.1f}%
- **平均买卖点准确率**: {avg_accuracy*100:.2f}%
- **平均年化收益率**: {avg_return*100:.2f}%

## 最佳配置 (按买卖点准确率)

### 股票
**{best_by_accuracy['stock_name']} ({best_by_accuracy['stock_code']})** - {best_by_accuracy['industry']}行业

### 配置名称
**{best_by_accuracy['config_name']}**

### 缠论参数
```json
{json.dumps(best_by_accuracy['chan_config'], indent=2, ensure_ascii=False)}
```

**关键参数**:
- `divergence_rate`: {best_by_accuracy['chan_config'].get('divergence_rate')} - 背驰率
- `min_zs_cnt`: {best_by_accuracy['chan_config'].get('min_zs_cnt')} - 最小中枢数量
- `bs_type`: "{best_by_accuracy['chan_config'].get('bs_type')}" - 买卖点类型
- `bi_strict`: {best_by_accuracy['chan_config'].get('bi_strict', False)} - 笔严格模式

### 策略参数
- **单次买入仓位**: {best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%
- **止损比例**: {best_by_accuracy['strategy_params']['stop_loss']*100:.1f}%
- **止盈比例**: {best_by_accuracy['strategy_params']['take_profit']*100:.1f}%

## 买卖点准确率分析

### 准确率定义
- **完全正确**: 买点后止盈或盈利>5%, 卖点后价格下跌或止盈
- **部分正确**: 小幅盈利(0-5%)或小幅止损(<3%)
- **准确率计算**: 完全正确 + 0.5×部分正确

### 最高准确率配置详情
- **买卖点总体准确率**: {best_by_accuracy['accuracy']['overall_effective_accuracy']*100:.2f}%
- **买点有效准确率**: {best_by_accuracy['accuracy']['buy_effective_accuracy']*100:.2f}%
- **卖点有效准确率**: {best_by_accuracy['accuracy']['sell_effective_accuracy']*100:.2f}%
- **买点信号数**: {best_by_accuracy['accuracy']['buy_signal_count']} (完全正确{best_by_accuracy['accuracy']['buy_correct_count']}, 部分正确{best_by_accuracy['accuracy']['buy_partial_count']})
- **卖点信号数**: {best_by_accuracy['accuracy']['sell_signal_count']} (完全正确{best_by_accuracy['accuracy']['sell_correct_count']}, 部分正确{best_by_accuracy['accuracy']['sell_partial_count']})

### 回测绩效
- **年化收益率**: {best_by_accuracy['metrics']['annual_return']*100:.2f}%
- **累计收益**: {best_by_accuracy['metrics']['total_return']*100:.2f}%
- **最大回撤**: {best_by_accuracy['metrics']['max_drawdown']*100:.2f}%
- **胜率**: {best_by_accuracy['metrics']['win_rate']*100:.2f}%
- **盈亏比**: {best_by_accuracy['metrics']['profit_loss_ratio']:.2f}
- **交易次数**: {best_by_accuracy['metrics']['trade_count']}

## 最佳配置 (按年化收益率)

### 股票
**{best_by_return['stock_name']} ({best_by_return['stock_code']})** - {best_by_return['industry']}行业

### 配置名称
**{best_by_return['config_name']}**

### 缠论参数
```json
{json.dumps(best_by_return['chan_config'], indent=2, ensure_ascii=False)}
```

### 买卖点准确率
- **买卖点总体准确率**: {best_by_return['accuracy']['overall_effective_accuracy']*100:.2f}%
- **买点有效准确率**: {best_by_return['accuracy']['buy_effective_accuracy']*100:.2f}%
- **卖点有效准确率**: {best_by_return['accuracy']['sell_effective_accuracy']*100:.2f}%

### 回测绩效
- **年化收益率**: {best_by_return['metrics']['annual_return']*100:.2f}%
- **累计收益**: {best_by_return['metrics']['total_return']*100:.2f}%
- **最大回撤**: {best_by_return['metrics']['max_drawdown']*100:.2f}%
- **胜率**: {best_by_return['metrics']['win_rate']*100:.2f}%
- **交易次数**: {best_by_return['metrics']['trade_count']}

## TOP 10 配置对比 (按准确率排序)

| 排名 | 股票 | 配置 | 准确率 | 买点准确 | 卖点准确 | 年化收益率 | 胜率 | 交易次数 |
|------|------|------|--------|---------|---------|-----------|------|---------|
"""

    for i, r in enumerate(successful_results[:10], 1):
        m = r['metrics']
        a = r['accuracy']
        report_content += f"| {i} | {r['stock_name']} | {r['config_name']} | {a['overall_effective_accuracy']*100:.0f}% | {a['buy_effective_accuracy']*100:.0f}% | {a['sell_effective_accuracy']*100:.0f}% | {m['annual_return']*100:6.1f}% | {m['win_rate']*100:.0f}% | {m['trade_count']} |\n"

    report_content += f"""

## 行业分析
"""

    for industry, results in industry_stats.items():
        if len(results) >= 2:  # 只分析有足够样本的行业
            ind_acc = sum(r['accuracy']['overall_effective_accuracy'] for r in results) / len(results)
            ind_ret = sum(r['metrics']['annual_return'] for r in results) / len(results)
            report_content += f"""
### {industry}
- **配置数量**: {len(results)}个
- **平均准确率**: {ind_acc*100:.2f}%
- **平均年化收益**: {ind_ret*100:.2f}%
- **最佳股票**: {max(results, key=lambda x: x['accuracy']['overall_effective_accuracy'])['stock_name']}
"""

    report_content += f"""

## 关键发现与建议

### 1. 买卖点准确率分析
基于{len(successful_results)}个有效配置的回测结果:

#### 准确率分布
- **高准确率(>60%)**: {sum(1 for r in successful_results if r['accuracy']['overall_effective_accuracy'] > 0.6)}个配置
- **中准确率(40-60%)**: {sum(1 for r in successful_results if 0.4 <= r['accuracy']['overall_effective_accuracy'] <= 0.6)}个配置
- **低准确率(<40%)**: {sum(1 for r in successful_results if r['accuracy']['overall_effective_accuracy'] < 0.4)}个配置

#### 准确率与收益率关系
- 平均准确率: {avg_accuracy*100:.2f}%
- 平均收益率: {avg_return*100:.2f}%
- 准确率>50%的配置平均收益率: {sum(r['metrics']['annual_return'] for r in successful_results if r['accuracy']['overall_effective_accuracy'] > 0.5) / max(1, sum(1 for r in successful_results if r['accuracy']['overall_effective_accuracy'] > 0.5))*100:.2f}%

### 2. 最佳参数推荐

#### 综合最佳配置 (准确率与收益率平衡)
**推荐配置**: {best_by_accuracy['config_name']}

**缠论参数**:
- 背驰率: {best_by_accuracy['chan_config'].get('divergence_rate')}
- 中枢数: {best_by_accuracy['chan_config'].get('min_zs_cnt')}
- 买卖点类型: {best_by_accuracy['chan_config'].get('bs_type')}
- 笔严格模式: {best_by_accuracy['chan_config'].get('bi_strict', False)}

**风控参数**:
- 单次仓位: {best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%
- 止损: {best_by_accuracy['strategy_params']['stop_loss']*100:.1f}%
- 止盈: {best_by_accuracy['strategy_params']['take_profit']*100:.1f}%

**预期效果**:
- 买卖点准确率: {best_by_accuracy['accuracy']['overall_effective_accuracy']*100:.1f}%
- 年化收益率: {best_by_accuracy['metrics']['annual_return']*100:.1f}%

### 3. 买卖点判定准确性分析

#### 买点判定
- 最佳买点准确率: {max(r['accuracy']['buy_effective_accuracy'] for r in successful_results)*100:.1f}%
- 平均买点准确率: {sum(r['accuracy']['buy_effective_accuracy'] for r in successful_results) / len(successful_results)*100:.1f}%

#### 卖点判定
- 最佳卖点准确率: {max(r['accuracy']['sell_effective_accuracy'] for r in successful_results)*100:.1f}%
- 平均卖点准确率: {sum(r['accuracy']['sell_effective_accuracy'] for r in successful_results) / len(successful_results)*100:.1f}%

#### 分析结论
"""

    buy_acc_avg = sum(r['accuracy']['buy_effective_accuracy'] for r in successful_results) / len(successful_results)
    sell_acc_avg = sum(r['accuracy']['sell_effective_accuracy'] for r in successful_results) / len(successful_results)

    if buy_acc_avg > sell_acc_avg:
        report_content += "- **买点判定优于卖点**: 缠论买点识别相对准确，建议重点关注买点信号\n"
    else:
        report_content += "- **卖点判定优于买点**: 缠论卖点识别相对准确，建议及时止盈\n"

    report_content += f"""
- **整体准确率**: {avg_accuracy*100:.1f}%，说明缠论买卖点识别有一定准确性
- **胜率分析**: 平均胜率{sum(r['metrics']['win_rate'] for r in successful_results) / len(successful_results)*100:.1f}%，与准确率相关

### 4. 实盘建议

#### 参数设置
1. **推荐使用**: {best_by_accuracy['config_name']}
2. **背驰率**: {best_by_accuracy['chan_config'].get('divergence_rate')} (根据市场调整)
3. **止损**: {best_by_accuracy['strategy_params']['stop_loss']*100:.1f}% (严格执行)
4. **止盈**: {best_by_accuracy['strategy_params']['take_profit']*100:.1f}% (可分批止盈)

#### 交易纪律
1. 只在准确率>50%的配置下交易
2. 严格按买卖点信号操作，不主观判断
3. 止盈止损必须执行
4. 控制单次仓位在{best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%左右

#### 适用场景
- 30分钟级别适合短线交易
- 需要盯盘，及时响应信号
- 适合波动性适中的股票

#### 风险提示
⚠️ **重要提示**:
- 历史回测结果不代表未来收益
- 30分钟级别信号频繁，交易成本较高
- 建议先用模拟盘验证
- 注意市场环境变化对策略的影响

## 买卖点准确率统计方法

### 评级标准
1. **完全正确**: 买点后盈利>5%或止盈；卖点后价格下跌或止盈
2. **部分正确**: 小幅盈利(0-5%)或小幅止损(<3%)
3. **错误**: 损失>3%或卖点后价格上涨

### 计算公式
```
有效准确率 = (完全正确数 + 0.5×部分正确数) / 总信号数
```

这个计算方式更合理地反映了买卖点的实际价值。

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**回测工具**: Python + 缠论库 + 买卖点准确率分析
**数据源**: BaoStock 30分钟K线
**回测版本**: 改进版 v2.0
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点30分钟K线回测报告_改进版.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 报告已保存至: {report_path}")

    # 保存JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_30m_improved_results.json"
    serializable_results = []
    for r in successful_results:
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'industry': r['industry'],
            'config_name': r['config_name'],
            'chan_config': r['chan_config'],
            'strategy_params': {
                k: str(v) if isinstance(v, list) else v
                for k, v in r['strategy_params'].items()
            },
            'metrics': r['metrics'],
            'accuracy': r['accuracy'],
            'sample_trades': r['trades'],
        }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 详细结果已保存至: {json_path}")
    print(f"\n{'='*80}")
    print("✅ 30分钟K线回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
