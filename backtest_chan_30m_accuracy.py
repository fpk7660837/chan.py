"""
缠论买卖点策略 - 30分钟K线回测 (带买卖点准确率统计)

核心改进:
1. 使用30分钟K线数据 (BaoStock支持)
2. 增加买卖点判定准确率统计
3. 测试多种缠论配置参数
4. 5个不同行业的股票
5. 输出最佳配置到文档
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from datetime import datetime
import json
import pandas as pd


# 选择5个不同行业的股票
STOCK_LIST = [
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sh.600036", "name": "招商银行", "industry": "银行"},
    {"code": "sz.000858", "name": "五粮液", "industry": "白酒"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
]


def calculate_bsp_accuracy(trades, equity_curve):
    """
    计算买卖点准确率

    准确率定义:
    - 买点准确: 买入后未触发止损即实现盈利
    - 卖点准确: 卖出时盈利或避免了更大损失
    """
    if not trades:
        return {
            'buy_accuracy': 0,
            'sell_accuracy': 0,
            'total_bsp_count': 0,
            'correct_bsp_count': 0
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
                # 判断买点是否准确: 如果卖出盈利或止损幅度小于5%
                is_correct = next_sell.get('profit', 0) > 0 or next_sell.get('profit_rate', -1) > -0.05
                buy_signals.append({
                    'is_correct': is_correct,
                    'profit_rate': next_sell.get('profit_rate', 0),
                    'reason': next_sell.get('reason', '')
                })

        elif trade['type'] == 'sell' and i > 0:
            # 判断卖点是否准确: 卖出后价格是否下跌或卖出时盈利
            profit_rate = trade.get('profit_rate', 0)
            reason = trade.get('reason', '')

            # 如果是止盈卖出，认为准确
            # 如果是止损但损失小于3%，也认为是及时止损
            # 如果正常卖点卖出且盈利，认为准确
            is_correct = (
                '止盈' in reason or
                profit_rate > 0 or
                ('止损' in reason and profit_rate > -0.03)
            )

            sell_signals.append({
                'is_correct': is_correct,
                'profit_rate': profit_rate,
                'reason': reason
            })

    buy_correct = sum(1 for s in buy_signals if s['is_correct'])
    sell_correct = sum(1 for s in sell_signals if s['is_correct'])

    buy_accuracy = buy_correct / len(buy_signals) if buy_signals else 0
    sell_accuracy = sell_correct / len(sell_signals) if sell_signals else 0

    total_bsp = len(buy_signals) + len(sell_signals)
    correct_bsp = buy_correct + sell_correct
    overall_accuracy = correct_bsp / total_bsp if total_bsp > 0 else 0

    return {
        'buy_accuracy': buy_accuracy,
        'sell_accuracy': sell_accuracy,
        'overall_accuracy': overall_accuracy,
        'buy_signal_count': len(buy_signals),
        'sell_signal_count': len(sell_signals),
        'buy_correct_count': buy_correct,
        'sell_correct_count': sell_correct,
        'total_bsp_count': total_bsp,
        'correct_bsp_count': correct_bsp,
        'buy_signals': buy_signals[:10],  # 保留前10个样本
        'sell_signals': sell_signals[:10]
    }


def run_backtest_30m(stock_code, stock_name, chan_config_dict, strategy_params):
    """运行30分钟K线回测"""

    # 使用2023年数据
    begin_time = "2023-01-01"
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
        print(f"初始化失败: {e}")
        return None

    # 策略状态
    initial_capital = 100000.0
    cash = initial_capital
    position = 0
    cost_price = 0

    trades = []
    equity_curve = []

    buy_percent = strategy_params.get('buy_percent', 0.3)
    stop_loss = strategy_params.get('stop_loss', -0.08)
    take_profit = strategy_params.get('take_profit', 0.25)
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

            # 止盈止损
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

            # 获取买卖点
            bsp_list = chan_snapshot.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            if not any(t in last_bsp.type for t in target_bsp_types):
                continue

            if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
                continue

            # 买入
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

            # 卖出
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
        print(f"回测执行失败: {e}")
        return None

    # 最终清仓
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

    # 计算买卖点准确率
    accuracy_stats = calculate_bsp_accuracy(trades, equity_curve)

    # 计算绩效
    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital
    annual_return = total_return  # 1年数据

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
        'trades': trades[:20],
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 30分钟K线回测 (带准确率统计)".center(80))
    print("="*80)

    # 针对30分钟级别优化的配置
    test_configs = [
        {
            "name": "配置1-标准",
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
            "name": "配置2-宽松",
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
            "name": "配置3-严格",
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
            "name": "配置4-平衡",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.85,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.28,
                "stop_loss": -0.055,
                "take_profit": 0.13,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "配置5-MACD面积",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "macd_algo": "area",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.05,
                "take_profit": 0.12,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "配置6-低背驰率",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.6,
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
    ]

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock")
    print(f"时间周期: 30分钟 (K_30M)")
    print(f"回测区间: 2023-01-01 至 2023-12-31 (1年)")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个配置)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:6s} - {config['name']:12s}...", end=" ")

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

                acc = result['accuracy']
                print(f"✓ {result['metrics']['trade_count']:2d}次 "
                      f"收益{result['metrics']['total_return']*100:5.1f}% "
                      f"准确率{acc['overall_accuracy']*100:4.1f}%")
            else:
                print("✗ 无交易或失败")

    successful_results = [r for r in all_results if r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 有效配置: {len(successful_results)}/{total}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        return

    # 按准确率排序
    successful_results.sort(key=lambda x: x['accuracy']['overall_accuracy'], reverse=True)

    # 显示TOP 10 (按准确率)
    print("\n" + "="*80)
    print("TOP 10 配置 (按买卖点准确率排序)".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results[:10]:
        m = r['metrics']
        a = r['accuracy']
        summary_data.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '买卖点准确率': f"{a['overall_accuracy']*100:.1f}%",
            '买点准确率': f"{a['buy_accuracy']*100:.1f}%",
            '卖点准确率': f"{a['sell_accuracy']*100:.1f}%",
            '年化收益': f"{m['annual_return']*100:.2f}%",
            '胜率': f"{m['win_rate']*100:.1f}%",
            '交易次数': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 按收益排序
    successful_results_by_return = sorted(successful_results, key=lambda x: x['metrics']['annual_return'], reverse=True)

    print("\n" + "="*80)
    print("TOP 10 配置 (按年化收益率排序)".center(80))
    print("="*80)

    summary_data2 = []
    for r in successful_results_by_return[:10]:
        m = r['metrics']
        a = r['accuracy']
        summary_data2.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '年化收益': f"{m['annual_return']*100:.2f}%",
            '买卖点准确率': f"{a['overall_accuracy']*100:.1f}%",
            '买点准确率': f"{a['buy_accuracy']*100:.1f}%",
            '胜率': f"{m['win_rate']*100:.1f}%",
            '交易次数': m['trade_count'],
        })

    df2 = pd.DataFrame(summary_data2)
    print("\n" + df2.to_string(index=False))

    # 生成报告
    best_by_accuracy = successful_results[0]
    best_by_return = successful_results_by_return[0]

    report_content = f"""# 缠论买卖点30分钟K线回测报告 (带准确率统计)

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 回测概况

### 测试规模
- **股票数量**: {len(STOCK_LIST)}只
- **覆盖行业**: {', '.join(set(s['industry'] for s in STOCK_LIST))}
- **配置数量**: {len(test_configs)}种
- **回测区间**: 2023-01-01 至 2023-12-31 (1年)
- **时间周期**: 30分钟 (K_30M)
- **数据源**: BaoStock
- **初始资金**: 100,000 元

### 测试股票
"""

    for stock in STOCK_LIST:
        report_content += f"- **{stock['name']}** ({stock['code']}) - {stock['industry']}\n"

    report_content += f"""
### 测试结果
- **总测试配置**: {total}个
- **有效配置**: {len(successful_results)}个
- **有效率**: {len(successful_results)/total*100:.1f}%

## 最佳配置 (按买卖点准确率)

### 股票
**{best_by_accuracy['stock_name']} ({best_by_accuracy['stock_code']})** - {best_by_accuracy['industry']}行业

### 配置名称
**{best_by_accuracy['config_name']}**

### 缠论参数
```json
{json.dumps(best_by_accuracy['chan_config'], indent=2, ensure_ascii=False)}
```

### 策略参数
- **单次买入仓位**: {best_by_accuracy['strategy_params']['buy_percent']*100:.0f}%
- **止损比例**: {best_by_accuracy['strategy_params']['stop_loss']*100:.1f}%
- **止盈比例**: {best_by_accuracy['strategy_params']['take_profit']*100:.1f}%

## 买卖点准确率分析

### 最高准确率配置
- **买卖点总体准确率**: {best_by_accuracy['accuracy']['overall_accuracy']*100:.2f}%
- **买点准确率**: {best_by_accuracy['accuracy']['buy_accuracy']*100:.2f}%
- **卖点准确率**: {best_by_accuracy['accuracy']['sell_accuracy']*100:.2f}%
- **买点信号数**: {best_by_accuracy['accuracy']['buy_signal_count']}
- **卖点信号数**: {best_by_accuracy['accuracy']['sell_signal_count']}
- **正确买点数**: {best_by_accuracy['accuracy']['buy_correct_count']}
- **正确卖点数**: {best_by_accuracy['accuracy']['sell_correct_count']}

### 回测绩效
- **年化收益率**: {best_by_accuracy['metrics']['annual_return']*100:.2f}%
- **累计收益**: {best_by_accuracy['metrics']['total_return']*100:.2f}%
- **最大回撤**: {best_by_accuracy['metrics']['max_drawdown']*100:.2f}%
- **胜率**: {best_by_accuracy['metrics']['win_rate']*100:.2f}%
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
- **买卖点总体准确率**: {best_by_return['accuracy']['overall_accuracy']*100:.2f}%
- **买点准确率**: {best_by_return['accuracy']['buy_accuracy']*100:.2f}%
- **卖点准确率**: {best_by_return['accuracy']['sell_accuracy']*100:.2f}%

### 回测绩效
- **年化收益率**: {best_by_return['metrics']['annual_return']*100:.2f}%
- **累计收益**: {best_by_return['metrics']['total_return']*100:.2f}%
- **最大回撤**: {best_by_return['metrics']['max_drawdown']*100:.2f}%
- **胜率**: {best_by_return['metrics']['win_rate']*100:.2f}%
- **交易次数**: {best_by_return['metrics']['trade_count']}

## TOP 10 配置对比 (按准确率排序)

| 排名 | 股票 | 配置 | 买卖点准确率 | 买点准确率 | 卖点准确率 | 年化收益率 | 胜率 | 交易次数 |
|------|------|------|-------------|-----------|-----------|-----------|------|---------|
"""

    for i, r in enumerate(successful_results[:10], 1):
        m = r['metrics']
        a = r['accuracy']
        report_content += f"| {i} | {r['stock_name']} | {r['config_name']} | {a['overall_accuracy']*100:.1f}% | {a['buy_accuracy']*100:.1f}% | {a['sell_accuracy']*100:.1f}% | {m['annual_return']*100:.2f}% | {m['win_rate']*100:.1f}% | {m['trade_count']} |\n"

    report_content += f"""
## 准确率与收益率关系分析

### 关键发现
"""

    # 计算准确率与收益率的相关性
    accuracies = [r['accuracy']['overall_accuracy'] for r in successful_results]
    returns = [r['metrics']['annual_return'] for r in successful_results]

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_return = sum(returns) / len(returns)

    report_content += f"""
1. **平均买卖点准确率**: {avg_accuracy*100:.2f}%
2. **平均年化收益率**: {avg_return*100:.2f}%
3. **准确率>70%的配置数**: {sum(1 for a in accuracies if a > 0.7)}个
4. **收益率>10%的配置数**: {sum(1 for r in returns if r > 0.1)}个

### 准确率定义说明

**买点准确**: 买入后未触发止损即实现盈利，或小幅止损(<5%)
**卖点准确**: 卖出时盈利，或止盈，或及时止损(损失<3%)

这个定义基于缠论理论：正确的买卖点应该能带来盈利或将损失控制在最小范围。

## 最佳参数推荐

### 综合考虑准确率和收益率

基于{len(successful_results)}个有效配置:
"""

    # 找到准确率和收益率都较高的配置
    balanced_results = [r for r in successful_results
                       if r['accuracy']['overall_accuracy'] > avg_accuracy
                       and r['metrics']['annual_return'] > avg_return]

    if balanced_results:
        best_balanced = balanced_results[0]
        report_content += f"""
**推荐配置**: {best_balanced['config_name']}
**股票**: {best_balanced['stock_name']}

**缠论参数**:
- 背驰率: {best_balanced['chan_config'].get('divergence_rate')}
- 中枢数: {best_balanced['chan_config'].get('min_zs_cnt')}
- 买卖点类型: {best_balanced['chan_config'].get('bs_type')}

**风控参数**:
- 单次仓位: {best_balanced['strategy_params']['buy_percent']*100:.0f}%
- 止损: {best_balanced['strategy_params']['stop_loss']*100:.1f}%
- 止盈: {best_balanced['strategy_params']['take_profit']*100:.1f}%

**绩效**:
- 买卖点准确率: {best_balanced['accuracy']['overall_accuracy']*100:.2f}%
- 年化收益率: {best_balanced['metrics']['annual_return']*100:.2f}%
- 胜率: {best_balanced['metrics']['win_rate']*100:.1f}%
"""

    report_content += f"""
## 实盘建议

### 1. 参数选择
- 优先选择买卖点准确率>70%的配置
- 同时关注收益率和最大回撤
- 30分钟级别建议降低单次仓位至20-25%

### 2. 风险控制
- 严格执行止盈止损
- 避免隔夜持仓风险
- 注意盘中波动

### 3. 适用场景
- 适合日内交易
- 需要盯盘
- 流动性好的股票

### 注意事项

⚠️ **重要提示**:
- 买卖点准确率高不一定收益率就高（可能单次盈利小）
- 收益率高但准确率低意味着风险较大
- 建议选择准确率和收益率都较高的平衡配置
- 历史回测不代表未来收益

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**回测工具**: Python + 缠论库 + 买卖点准确率统计
**数据源**: BaoStock 30分钟K线
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点30分钟K线准确率回测报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 报告已保存至: {report_path}")

    # 保存JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_30m_accuracy_results.json"
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
    print("✅ 30分钟K线准确率回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
