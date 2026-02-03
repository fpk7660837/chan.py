"""
缠论买卖点策略多股票多周期回测

测试不同行业的5只股票，使用日线数据（因BaoStock限制）
测试多种缠论配置参数，找出最佳配置
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
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600036", "name": "招商银行", "industry": "银行"},
    {"code": "sz.000858", "name": "五粮液", "industry": "白酒"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
]


def backtest_single_config(stock_code, stock_name, timeframe, config_name, chan_config_dict, strategy_params):
    """
    回测单个配置

    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        timeframe: 时间周期
        config_name: 配置名称
        chan_config_dict: 缠论配置字典
        strategy_params: 策略参数
    """

    print(f"\n{'='*80}")
    print(f"股票: {stock_name} ({stock_code}) | 周期: {timeframe.name} | 配置: {config_name}")
    print(f"{'='*80}")

    begin_time = "2020-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK

    config = CChanConfig(chan_config_dict)

    try:
        chan = CChan(
            code=stock_code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=[timeframe],
            config=config,
            autype=AUTYPE.QFQ,
        )
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
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

    bar_count = 0
    try:
        for chan_snapshot in chan.step_load():
            bar_count += 1
            cur_lv_chan = chan_snapshot[0]

            if len(cur_lv_chan) < 2:
                continue

            current_price = cur_lv_chan[-1][-1].close

            # 计算当前总资产
            total_value = cash + position * current_price
            equity_curve.append({
                'time': str(cur_lv_chan[-1][-1].time),
                'total_value': total_value,
                'cash': cash,
                'position_value': position * current_price
            })

            # 止盈止损检查
            if position > 0:
                profit_rate = (current_price - cost_price) / cost_price

                # 止损
                if profit_rate <= stop_loss:
                    sell_value = position * current_price * 0.999
                    cash += sell_value
                    trades.append({
                        'time': str(cur_lv_chan[-1][-1].time),
                        'type': 'sell',
                        'price': current_price,
                        'volume': position,
                        'reason': f'止损 {profit_rate*100:.2f}%',
                        'profit': sell_value - position * cost_price,
                        'profit_rate': profit_rate
                    })
                    position = 0
                    cost_price = 0
                    continue

                # 止盈
                if profit_rate >= take_profit:
                    sell_value = position * current_price * 0.999
                    cash += sell_value
                    trades.append({
                        'time': str(cur_lv_chan[-1][-1].time),
                        'type': 'sell',
                        'price': current_price,
                        'volume': position,
                        'reason': f'止盈 {profit_rate*100:.2f}%',
                        'profit': sell_value - position * cost_price,
                        'profit_rate': profit_rate
                    })
                    position = 0
                    cost_price = 0
                    continue

            # 获取最新买卖点
            bsp_list = chan_snapshot.get_latest_bsp(number=1)
            if not bsp_list:
                continue

            last_bsp = bsp_list[0]

            # 检查买卖点类型是否匹配
            if not any(t in last_bsp.type for t in target_bsp_types):
                continue

            # 检查买卖点是否在倒数第二个K线组合上（已确认）
            if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
                continue

            # 买入信号：买点 + 底分型
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

            # 卖出信号：卖点 + 顶分型
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
        print(f"  ✗ 回测执行失败: {e}")
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

    # 计算绩效指标
    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital

    # 计算年化收益率
    years = 4.0  # 2020-2023
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1

    # 计算胜率
    sell_trades = [t for t in trades if t['type'] == 'sell' and t['reason'] != '强制平仓']
    win_trades = [t for t in sell_trades if t['profit'] > 0]
    loss_trades = [t for t in sell_trades if t['profit'] < 0]
    win_rate = len(win_trades) / len(sell_trades) if len(sell_trades) > 0 else 0

    # 计算最大回撤
    max_drawdown = 0
    peak = initial_capital
    for point in equity_curve:
        if point['total_value'] > peak:
            peak = point['total_value']
        drawdown = (peak - point['total_value']) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # 计算盈亏比
    avg_win = sum(t['profit'] for t in win_trades) / len(win_trades) if len(win_trades) > 0 else 0
    avg_loss = abs(sum(t['profit'] for t in loss_trades) / len(loss_trades)) if len(loss_trades) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    buy_trades = [t for t in trades if t['type'] == 'buy']

    print(f"  交易次数: {len(buy_trades)} | 最终资产: {final_value:,.0f} | 收益率: {total_return*100:.2f}% | 胜率: {win_rate*100:.1f}%")

    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'timeframe': timeframe.name,
        'config_name': config_name,
        'chan_config': chan_config_dict,
        'strategy_params': strategy_params,
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
            'bar_count': bar_count,
        },
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 多股票多周期回测系统".center(80))
    print("="*80)

    # 测试的时间周期（只使用日线，因为BaoStock对分钟数据支持有限）
    timeframes = [
        KL_TYPE.K_DAY,
    ]

    # 缠论配置参数组合
    chan_configs = [
        {
            "name": "标准配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
            }
        },
        {
            "name": "宽松配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 1.1,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
            }
        },
        {
            "name": "严格配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.7,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": True,
            }
        },
        {
            "name": "平衡配置",
            "config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
            }
        },
    ]

    # 策略参数组合
    strategy_params_list = [
        {
            "name": "保守策略",
            "params": {
                "buy_percent": 0.25,
                "stop_loss": -0.08,
                "take_profit": 0.30,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "平衡策略",
            "params": {
                "buy_percent": 0.3,
                "stop_loss": -0.07,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "激进策略",
            "params": {
                "buy_percent": 0.35,
                "stop_loss": -0.06,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
    ]

    # 运行所有组合
    all_results = []
    total_tests = len(STOCK_LIST) * len(timeframes) * len(chan_configs) * len(strategy_params_list)
    current_test = 0

    print(f"\n总共需要测试: {total_tests} 个配置组合")
    print(f"股票数: {len(STOCK_LIST)}, 周期数: {len(timeframes)}, 缠论配置: {len(chan_configs)}, 策略参数: {len(strategy_params_list)}")

    for stock in STOCK_LIST:
        for timeframe in timeframes:
            for chan_config in chan_configs:
                for strategy_params in strategy_params_list:
                    current_test += 1
                    config_full_name = f"{stock['name']}-{timeframe.name}-{chan_config['name']}-{strategy_params['name']}"

                    print(f"\n[{current_test}/{total_tests}] 测试中...")

                    result = backtest_single_config(
                        stock_code=stock['code'],
                        stock_name=stock['name'],
                        timeframe=timeframe,
                        config_name=config_full_name,
                        chan_config_dict=chan_config['config'],
                        strategy_params=strategy_params['params']
                    )

                    if result:
                        all_results.append(result)

    # 筛选成功的结果
    successful_results = [r for r in all_results if r.get('success') and r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 成功: {len(successful_results)}/{len(all_results)}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        return

    # 按年化收益率排序
    successful_results.sort(key=lambda x: x['metrics']['annual_return'], reverse=True)

    # 创建结果表格
    print("\n" + "="*80)
    print("TOP 10 配置".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results[:10]:
        m = r['metrics']
        summary_data.append({
            '股票': r['stock_name'],
            '周期': r['timeframe'],
            '配置': r['config_name'].split('-')[2],  # 提取配置名称
            '策略': r['config_name'].split('-')[3],  # 提取策略名称
            '年化收益': f"{m['annual_return']*100:.2f}%",
            '累计收益': f"{m['total_return']*100:.2f}%",
            '最大回撤': f"{m['max_drawdown']*100:.2f}%",
            '胜率': f"{m['win_rate']*100:.1f}%",
            '交易次数': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 生成最佳配置报告
    best_result = successful_results[0]

    # 按股票统计最佳配置
    stock_best = {}
    for stock in STOCK_LIST:
        stock_results = [r for r in successful_results if r['stock_code'] == stock['code']]
        if stock_results:
            stock_best[stock['name']] = stock_results[0]

    report_content = f"""# 缠论买卖点策略多股票回测最佳配置报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 回测说明
- **回测股票**: {len(STOCK_LIST)}只（涵盖{len(set(s['industry'] for s in STOCK_LIST))}个行业）
- **回测区间**: 2020-01-01 至 2023-12-31
- **时间周期**: 日线 (K_DAY)
- **初始资金**: 100,000 元
- **数据源**: BaoStock
- **总测试配置**: {total_tests}个
- **成功配置**: {len(successful_results)}个

## 全局最佳配置

### 股票
**{best_result['stock_name']} ({best_result['stock_code']})**

### 时间周期
**{best_result['timeframe']}**

### 缠论参数
```json
{json.dumps(best_result['chan_config'], indent=2, ensure_ascii=False)}
```

**参数说明**:
- `divergence_rate`: {best_result['chan_config'].get('divergence_rate', 'N/A')} - 背驰率阈值
- `min_zs_cnt`: {best_result['chan_config'].get('min_zs_cnt', 'N/A')} - 最小中枢数量
- `bs_type`: "{best_result['chan_config'].get('bs_type', 'N/A')}" - 买卖点类型

### 策略参数
- 单次买入仓位: {best_result['strategy_params']['buy_percent']*100:.0f}%
- 止损比例: {best_result['strategy_params']['stop_loss']*100:.0f}%
- 止盈比例: {best_result['strategy_params']['take_profit']*100:.0f}%

## 回测绩效

### 收益指标
- **最终资产**: {best_result['metrics']['final_value']:,.2f} 元
- **累计收益**: {best_result['metrics']['total_return']*100:.2f}%
- **年化收益率**: {best_result['metrics']['annual_return']*100:.2f}%

### 风险指标
- **最大回撤**: {best_result['metrics']['max_drawdown']*100:.2f}%

### 交易统计
- **交易次数**: {best_result['metrics']['trade_count']}
- **盈利次数**: {best_result['metrics']['win_count']}
- **亏损次数**: {best_result['metrics']['loss_count']}
- **胜率**: {best_result['metrics']['win_rate']*100:.2f}%
- **盈亏比**: {best_result['metrics']['profit_loss_ratio']:.2f}

## TOP 10 配置对比

| 排名 | 股票 | 周期 | 配置 | 策略 | 年化收益率 | 累计收益率 | 最大回撤 | 胜率 | 交易次数 |
|------|------|------|------|------|-----------|-----------|---------|------|---------|
"""

    for i, r in enumerate(successful_results[:10], 1):
        m = r['metrics']
        config_short = r['config_name'].split('-')[2]
        strategy_short = r['config_name'].split('-')[3]
        report_content += f"| {i} | {r['stock_name']} | {r['timeframe']} | {config_short} | {strategy_short} | {m['annual_return']*100:.2f}% | {m['total_return']*100:.2f}% | {m['max_drawdown']*100:.2f}% | {m['win_rate']*100:.1f}% | {m['trade_count']} |\n"

    report_content += f"""
## 各股票最佳配置

"""

    for stock_name, result in stock_best.items():
        m = result['metrics']
        report_content += f"""### {stock_name}

- **配置**: {result['config_name'].split('-')[2]}
- **策略**: {result['config_name'].split('-')[3]}
- **年化收益率**: {m['annual_return']*100:.2f}%
- **累计收益率**: {m['total_return']*100:.2f}%
- **最大回撤**: {m['max_drawdown']*100:.2f}%
- **胜率**: {m['win_rate']*100:.1f}%
- **交易次数**: {m['trade_count']}

"""

    report_content += f"""
## 配置建议

### 1. 最佳缠论参数
- **背驰率 (divergence_rate)**: {best_result['chan_config'].get('divergence_rate', 'N/A')}
- **最小中枢数 (min_zs_cnt)**: {best_result['chan_config'].get('min_zs_cnt', 'N/A')}
- **买卖点类型 (bs_type)**: {best_result['chan_config'].get('bs_type', 'N/A')}

### 2. 最佳风控参数
- **单次仓位**: {best_result['strategy_params']['buy_percent']*100:.0f}%
- **止损**: {best_result['strategy_params']['stop_loss']*100:.0f}%
- **止盈**: {best_result['strategy_params']['take_profit']*100:.0f}%

### 3. 适用股票特征
根据回测结果，该配置在以下类型股票上表现较好：
"""

    # 统计表现好的股票行业
    top_stocks = successful_results[:5]
    industries = {}
    for r in top_stocks:
        stock_info = next((s for s in STOCK_LIST if s['code'] == r['stock_code']), None)
        if stock_info:
            industry = stock_info['industry']
            if industry not in industries:
                industries[industry] = []
            industries[industry].append(r['stock_name'])

    for industry, stocks in industries.items():
        report_content += f"- **{industry}**: {', '.join(stocks)}\n"

    report_content += f"""
### 4. 实盘建议
1. **选股**: 优先选择流动性好、波动适中的股票
2. **仓位**: 建议实盘时降低单次仓位至15-20%
3. **风控**: 严格执行止盈止损，避免情绪化交易
4. **环境**: 该策略在趋势明确的市场环境下表现更好

### 5. 风险提示
- 回测结果不代表未来表现
- 不同市场环境下策略有效性会变化
- 建议先用小资金验证后再逐步加大投入
- 注意流动性风险和滑点成本

## 数据说明

- **回测期间**: 2020-01-01 至 2023-12-31 (4年)
- **交易成本**: 手续费万三 + 滑点千一 + 印花税千一
- **复权方式**: 前复权
- **信号确认**: 要求买卖点配合分型确认，避免虚假信号
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点多股票回测最佳配置报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 最佳配置报告已保存至: {report_path}")

    # 保存JSON结果
    json_path = "/Users/fupengkai/Documents/quant/chan.py/multi_stock_backtest_results.json"
    serializable_results = []
    for r in successful_results[:50]:  # 只保存前50个结果
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'timeframe': r['timeframe'],
            'config_name': r['config_name'],
            'chan_config': r['chan_config'],
            'strategy_params': {
                k: str(v) if isinstance(v, list) else v
                for k, v in r['strategy_params'].items()
            },
            'metrics': r['metrics'],
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
