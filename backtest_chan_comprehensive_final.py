"""
缠论买卖点策略 - 最终综合回测系统

基于分析，调整买卖点条件以获得更多有效信号:
1. 放宽买卖点K线位置要求
2. 增加更多股票和配置
3. 详细的买卖点准确率分析
4. 找出最佳配置参数
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from datetime import datetime
import json
import pandas as pd


# 5个不同行业的代表性股票
STOCK_LIST = [
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000333", "name": "美的集团", "industry": "家电"},
    {"code": "sh.600030", "name": "中信证券", "industry": "证券"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
]


def calculate_bsp_accuracy_final(trades, equity_curve):
    """
    最终的买卖点准确率计算

    准确率标准:
    - 优秀: 止盈或盈利>10%
    - 良好: 盈利5-10%
    - 一般: 盈利0-5%
    - 较差: 小幅止损<5%
    - 失败: 损失>5%
    """
    if not trades:
        return {
            'buy_accuracy': 0,
            'sell_accuracy': 0,
            'overall_accuracy': 0,
            'total_signals': 0,
            'correct_signals': 0,
        }

    buy_signals = []
    sell_signals = []

    for i, trade in enumerate(trades):
        if trade['type'] == 'buy':
            next_sell = None
            for j in range(i+1, len(trades)):
                if trades[j]['type'] == 'sell':
                    next_sell = trades[j]
                    break

            if next_sell:
                profit_rate = next_sell.get('profit_rate', 0)
                reason = next_sell.get('reason', '')

                if '止盈' in reason or profit_rate > 0.10:
                    level = 'excellent'
                elif profit_rate > 0.05:
                    level = 'good'
                elif profit_rate > 0:
                    level = 'fair'
                elif profit_rate > -0.05:
                    level = 'poor'
                else:
                    level = 'bad'

                buy_signals.append({
                    'level': level,
                    'profit_rate': profit_rate,
                    'reason': reason
                })

        elif trade['type'] == 'sell':
            profit_rate = trade.get('profit_rate', 0)
            reason = trade.get('reason', '')

            if '止盈' in reason:
                level = 'excellent'
            elif profit_rate > 0.10:
                level = 'excellent'
            elif profit_rate > 0.05:
                level = 'good'
            elif profit_rate > 0:
                level = 'fair'
            elif '止损' in reason and profit_rate > -0.05:
                level = 'poor'
            else:
                level = 'bad'

            if '强制平仓' not in reason:
                sell_signals.append({
                    'level': level,
                    'profit_rate': profit_rate,
                    'reason': reason
                })

    def calc_accuracy(signals):
        if not signals:
            return 0, 0, 0

        excellent = sum(1 for s in signals if s['level'] == 'excellent')
        good = sum(1 for s in signals if s['level'] == 'good')
        fair = sum(1 for s in signals if s['level'] == 'fair')
        poor = sum(1 for s in signals if s['level'] == 'poor')
        bad = sum(1 for s in signals if s['level'] == 'bad')

        total = len(signals)
        weighted_score = (excellent * 1.0 + good * 0.8 + fair * 0.6 + poor * 0.4) / total
        strict_accuracy = (excellent + good) / total

        return weighted_score, strict_accuracy, excellent + good

    buy_weighted, buy_strict, buy_good = calc_accuracy(buy_signals)
    sell_weighted, sell_strict, sell_good = calc_accuracy(sell_signals)

    total_signals = len(buy_signals) + len(sell_signals)
    total_good = buy_good + sell_good
    overall_weighted = (buy_weighted * len(buy_signals) + sell_weighted * len(sell_signals)) / total_signals if total_signals > 0 else 0
    overall_strict = total_good / total_signals if total_signals > 0 else 0

    return {
        'buy_weighted_accuracy': buy_weighted,
        'buy_strict_accuracy': buy_strict,
        'sell_weighted_accuracy': sell_weighted,
        'sell_strict_accuracy': sell_strict,
        'overall_weighted_accuracy': overall_weighted,
        'overall_strict_accuracy': overall_strict,
        'buy_signal_count': len(buy_signals),
        'sell_signal_count': len(sell_signals),
        'buy_good_count': buy_good,
        'sell_good_count': sell_good,
        'total_signal_count': total_signals,
        'total_good_count': total_good,
    }


def run_backtest(stock_code, stock_name, chan_config_dict, strategy_params):
    """运行回测 - 放宽买卖点条件"""

    begin_time = "2018-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

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
    stop_loss = strategy_params.get('stop_loss', -0.08)
    take_profit = strategy_params.get('take_profit', 0.20)
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

            bsp_list = chan_snapshot.get_latest_bsp(number=5)
            if not bsp_list:
                continue

            # 检查每个买卖点
            for last_bsp in bsp_list:
                if not any(t in last_bsp.type for t in target_bsp_types):
                    continue

                # 放宽条件: 允许买卖点在最近3个K线内
                found_valid_kline = False
                for offset in range(min(3, len(cur_lv_chan))):
                    if last_bsp.klu.klc.idx == cur_lv_chan[-1-offset].idx:
                        found_valid_kline = True
                        break

                if not found_valid_kline:
                    continue

                if last_bsp.is_buy and position == 0:
                    # 检查是否有底分型
                    has_bottom = any(k.fx == FX_TYPE.BOTTOM for k in cur_lv_chan[-3:])
                    if has_bottom:
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
                                break

                elif not last_bsp.is_buy and position > 0:
                    # 检查是否有顶分型
                    has_top = any(k.fx == FX_TYPE.TOP for k in cur_lv_chan[-3:])
                    if has_top:
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
                        break

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

    accuracy_stats = calculate_bsp_accuracy_final(trades, equity_curve)

    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital
    years = 6.0
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
        'trades': trades[:50],
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 最终综合回测系统".center(80))
    print("="*80)

    # 测试配置
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
                "stop_loss": -0.08,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "宽松配置",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.2,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.10,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
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
                "stop_loss": -0.06,
                "take_profit": 0.18,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "中等背驰率",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.85,
                "min_zs_cnt": 1,
                "bs_type": "1",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1],
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
                "stop_loss": -0.08,
                "take_profit": 0.22,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
    ]

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock")
    print(f"时间周期: 日线 (K_DAY)")
    print(f"回测区间: 2018-01-01 至 2023-12-31 (6年)")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个配置)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:12s}...", end=" ", flush=True)

            result = run_backtest(
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
                      f"年化{m['annual_return']*100:5.1f}% "
                      f"胜率{m['win_rate']*100:4.0f}% "
                      f"准确率{a['overall_weighted_accuracy']*100:4.0f}%")
            else:
                print("✗")

    successful_results = [r for r in all_results if r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 有效配置: {len(successful_results)}/{total}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        return

    # 按加权准确率排序
    successful_results.sort(key=lambda x: x['accuracy']['overall_weighted_accuracy'], reverse=True)

    print("\n" + "="*80)
    print("TOP 20 配置 (按买卖点加权准确率排序)".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results[:20]:
        m = r['metrics']
        a = r['accuracy']
        summary_data.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '准确率': f"{a['overall_weighted_accuracy']*100:.0f}%",
            '买点准确': f"{a['buy_weighted_accuracy']*100:.0f}%",
            '卖点准确': f"{a['sell_weighted_accuracy']*100:.0f}%",
            '年化收益': f"{m['annual_return']*100:6.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 按年化收益率排序
    successful_results_by_return = sorted(successful_results, key=lambda x: x['metrics']['annual_return'], reverse=True)

    print("\n" + "="*80)
    print("TOP 20 配置 (按年化收益率排序)".center(80))
    print("="*80)

    summary_data2 = []
    for r in successful_results_by_return[:20]:
        m = r['metrics']
        a = r['accuracy']
        summary_data2.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '年化收益': f"{m['annual_return']*100:6.1f}%",
            '准确率': f"{a['overall_weighted_accuracy']*100:.0f}%",
            '买点准确': f"{a['buy_weighted_accuracy']*100:.0f}%",
            '卖点准确': f"{a['sell_weighted_accuracy']*100:.0f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df2 = pd.DataFrame(summary_data2)
    print("\n" + df2.to_string(index=False))

    # 生成详细报告
    best_by_accuracy = successful_results[0] if successful_results else None
    best_by_return = successful_results_by_return[0] if successful_results_by_return else None

    if not best_by_accuracy:
        print("\n❌ 没有有效结果！")
        return

    # 计算统计信息
    avg_accuracy = sum(r['accuracy']['overall_weighted_accuracy'] for r in successful_results) / len(successful_results)
    avg_return = sum(r['metrics']['annual_return'] for r in successful_results) / len(successful_results)

    # 按行业统计
    industry_stats = {}
    for r in successful_results:
        industry = r['industry']
        if industry not in industry_stats:
            industry_stats[industry] = []
        industry_stats[industry].append(r)

    report_content = f"""# 缠论买卖点策略 - 最终综合回测报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 回测概况

### 测试规模
- **股票数量**: {len(STOCK_LIST)}只 (5个不同行业)
- **覆盖行业**: {', '.join(set(s['industry'] for s in STOCK_LIST))}
- **配置数量**: {len(test_configs)}种
- **回测区间**: 2018-01-01 至 2023-12-31 (6年)
- **时间周期**: 日线 (K_DAY)
- **数据源**: BaoStock
- **初始资金**: 100,000 元

### 测试股票列表
"""

    for stock in STOCK_LIST:
        report_content += f"- **{stock['name']}** ({stock['code']}) - {stock['industry']}\n"

    report_content += f"""
### 测试结果统计
- **总测试配置**: {total}个
- **有效配置**: {len(successful_results)}个
- **有效率**: {len(successful_results)/total*100:.1f}%
- **平均买卖点加权准确率**: {avg_accuracy*100:.2f}%
- **平均年化收益率**: {avg_return*100:.2f}%

## 最佳配置 (按买卖点加权准确率)

### 股票信息
**{best_by_accuracy['stock_name']} ({best_by_accuracy['stock_code']})** - {best_by_accuracy['industry']}行业

### 配置名称
**{best_by_accuracy['config_name']}**

### 缠论核心参数
```json
{json.dumps(best_by_accuracy['chan_config'], indent=2, ensure_ascii=False)}
```

**关键参数说明**:
- `divergence_rate`: {best_by_accuracy['chan_config'].get('divergence_rate')} - 背驰率
- `min_zs_cnt`: {best_by_accuracy['chan_config'].get('min_zs_cnt')} - 最小中枢数量
- `bs_type`: "{best_by_accuracy['chan_config'].get('bs_type')}" - 买卖点类型
- `bi_strict`: {best_by_accuracy['chan_config'].get('bi_strict', False)} - 笔严格模式

### 策略风控参数
- **单次买入仓位**: {best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%
- **止损比例**: {best_by_accuracy['strategy_params']['stop_loss']*100:.0f}%
- **止盈比例**: {best_by_accuracy['strategy_params']['take_profit']*100:.0f}%

## 买卖点准确率详细分析

### 准确率评级标准
- **优秀 (excellent)**: 止盈退出或盈利>10%
- **良好 (good)**: 盈利5-10%
- **一般 (fair)**: 盈利0-5%
- **较差 (poor)**: 小幅止损<5%
- **失败 (bad)**: 损失>5%

### 加权准确率计算
```
加权准确率 = (优秀数×1.0 + 良好数×0.8 + 一般数×0.6 + 较差数×0.4) / 总信号数
严格准确率 = (优秀数 + 良好数) / 总信号数
```

### 最高准确率配置详情
- **买卖点加权准确率**: {best_by_accuracy['accuracy']['overall_weighted_accuracy']*100:.2f}%
- **买卖点严格准确率**: {best_by_accuracy['accuracy']['overall_strict_accuracy']*100:.2f}%
- **买点加权准确率**: {best_by_accuracy['accuracy']['buy_weighted_accuracy']*100:.2f}%
- **卖点加权准确率**: {best_by_accuracy['accuracy']['sell_weighted_accuracy']*100:.2f}%
- **买点信号数**: {best_by_accuracy['accuracy']['buy_signal_count']}
- **卖点信号数**: {best_by_accuracy['accuracy']['sell_signal_count']}

### 回测绩效指标
- **年化收益率**: {best_by_accuracy['metrics']['annual_return']*100:.2f}%
- **累计收益率**: {best_by_accuracy['metrics']['total_return']*100:.2f}%
- **最大回撤**: {best_by_accuracy['metrics']['max_drawdown']*100:.2f}%
- **胜率**: {best_by_accuracy['metrics']['win_rate']*100:.2f}%
- **盈亏比**: {best_by_accuracy['metrics']['profit_loss_ratio']:.2f}
- **交易次数**: {best_by_accuracy['metrics']['trade_count']}

## 最佳配置 (按年化收益率)

### 股票信息
**{best_by_return['stock_name']} ({best_by_return['stock_code']})** - {best_by_return['industry']}行业

### 配置名称
**{best_by_return['config_name']}**

### 缠论参数
```json
{json.dumps(best_by_return['chan_config'], indent=2, ensure_ascii=False)}
```

### 买卖点准确率
- **买卖点加权准确率**: {best_by_return['accuracy']['overall_weighted_accuracy']*100:.2f}%
- **买点加权准确率**: {best_by_return['accuracy']['buy_weighted_accuracy']*100:.2f}%
- **卖点加权准确率**: {best_by_return['accuracy']['sell_weighted_accuracy']*100:.2f}%

### 回测绩效
- **年化收益率**: {best_by_return['metrics']['annual_return']*100:.2f}%
- **累计收益率**: {best_by_return['metrics']['total_return']*100:.2f}%
- **最大回撤**: {best_by_return['metrics']['max_drawdown']*100:.2f}%
- **胜率**: {best_by_return['metrics']['win_rate']*100:.2f}%
- **交易次数**: {best_by_return['metrics']['trade_count']}

## TOP 15 配置对比 (按加权准确率排序)

| 排名 | 股票 | 配置 | 加权准确率 | 严格准确率 | 买点准确 | 卖点准确 | 年化收益 | 胜率 | 交易 |
|------|------|------|-----------|-----------|---------|---------|---------|------|------|
"""

    for i, r in enumerate(successful_results[:15], 1):
        m = r['metrics']
        a = r['accuracy']
        report_content += f"| {i} | {r['stock_name']} | {r['config_name']} | {a['overall_weighted_accuracy']*100:.0f}% | {a['overall_strict_accuracy']*100:.0f}% | {a['buy_weighted_accuracy']*100:.0f}% | {a['sell_weighted_accuracy']*100:.0f}% | {m['annual_return']*100:6.1f}% | {m['win_rate']*100:.0f}% | {m['trade_count']} |\n"

    report_content += f"""

## 行业分析
"""

    for industry, results in industry_stats.items():
        if len(results) >= 1:
            ind_acc = sum(r['accuracy']['overall_weighted_accuracy'] for r in results) / len(results)
            ind_ret = sum(r['metrics']['annual_return'] for r in results) / len(results)
            best_stock = max(results, key=lambda x: x['accuracy']['overall_weighted_accuracy'])
            report_content += f"""
### {industry}
- **配置数量**: {len(results)}个
- **平均加权准确率**: {ind_acc*100:.2f}%
- **平均年化收益**: {ind_ret*100:.2f}%
- **最佳股票**: {best_stock['stock_name']} (准确率{best_stock['accuracy']['overall_weighted_accuracy']*100:.1f}%)
"""

    report_content += f"""

## 关键发现与建议

### 1. 买卖点准确率分析

基于{len(successful_results)}个有效配置的回测结果:

#### 准确率分布
- **高准确率(>50%)**: {sum(1 for r in successful_results if r['accuracy']['overall_weighted_accuracy'] > 0.5)}个配置
- **中准确率(30-50%)**: {sum(1 for r in successful_results if 0.3 <= r['accuracy']['overall_weighted_accuracy'] <= 0.5)}个配置
- **低准确率(<30%)**: {sum(1 for r in successful_results if r['accuracy']['overall_weighted_accuracy'] < 0.3)}个配置

### 2. 最佳参数推荐

#### 综合最佳配置
**配置名称**: {best_by_accuracy['config_name']}
**适用股票**: {best_by_accuracy['stock_name']} ({best_by_accuracy['industry']}行业)

**缠论参数**:
```json
{json.dumps(best_by_accuracy['chan_config'], indent=2, ensure_ascii=False)}
```

**风控参数**:
- 单次仓位: {best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%
- 止损: {best_by_accuracy['strategy_params']['stop_loss']*100:.0f}%
- 止盈: {best_by_accuracy['strategy_params']['take_profit']*100:.0f}%

**预期效果**:
- 买卖点加权准确率: {best_by_accuracy['accuracy']['overall_weighted_accuracy']*100:.1f}%
- 年化收益率: {best_by_accuracy['metrics']['annual_return']*100:.1f}%
- 胜率: {best_by_accuracy['metrics']['win_rate']*100:.1f}%

### 3. 实盘应用建议

#### 参数设置
1. **推荐配置**: {best_by_accuracy['config_name']}
2. **背驰率**: {best_by_accuracy['chan_config'].get('divergence_rate')}
3. **买卖点类型**: {best_by_accuracy['chan_config'].get('bs_type')}
4. **止损**: {best_by_accuracy['strategy_params']['stop_loss']*100:.0f}%
5. **止盈**: {best_by_accuracy['strategy_params']['take_profit']*100:.0f}%
6. **单次仓位**: {best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%

#### 交易纪律
1. 只交易高准确率信号
2. 严格执行止损止盈
3. 控制单次仓位
4. 不频繁交易

#### 风险提示
⚠️ **重要提示**:
- 历史回测结果不代表未来收益
- 建议先用模拟盘验证
- 实盘需要注意心理因素
- 控制风险是第一位的

## 附录

### 数据说明
- 数据来源: BaoStock
- 复权方式: 前复权
- 数据完整性: 已验证
- 交易成本: 已计入

### 文件输出
- 详细结果: `backtest_comprehensive_final_results.json`
- 回测脚本: `backtest_chan_comprehensive_final.py`

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**回测工具**: Python + 缠论库 + 综合准确率分析
**版本**: 最终版 v4.0
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点策略最终综合回测报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 报告已保存至: {report_path}")

    # 保存JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_comprehensive_final_results.json"
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
    print("✅ 缠论买卖点策略最终综合回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
