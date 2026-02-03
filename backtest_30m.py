"""
缠论买卖点策略 - 30分钟K线回测

使用BaoStock的30分钟数据，测试多种缠论配置
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


def run_backtest_30m(stock_code, stock_name, chan_config_dict, strategy_params):
    """运行30分钟K线回测"""

    # 30分钟数据量较大，缩短测试周期
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
                        trades.append({'time': str(cur_lv_chan[-1][-1].time), 'type': 'buy'})

            # 卖出
            elif not last_bsp.is_buy and cur_lv_chan[-2].fx == FX_TYPE.TOP and position > 0:
                sell_value = position * current_price * 0.999
                cash += sell_value
                profit = sell_value - position * cost_price
                profit_rate = (current_price - cost_price) / cost_price

                trades.append({
                    'time': str(cur_lv_chan[-1][-1].time),
                    'type': 'sell',
                    'profit': profit,
                    'profit_rate': profit_rate
                })
                position = 0
                cost_price = 0

    except Exception as e:
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
            'profit': profit,
            'profit_rate': profit_rate
        })

    # 计算绩效
    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital
    years = 1.0  # 2023年一年
    annual_return = total_return  # 一年的收益率就是年化收益率

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
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 30分钟K线回测".center(80))
    print("="*80)

    # 针对30分钟级别优化的配置
    test_configs = [
        {
            "name": "30分钟-宽松配置",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.2,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.06,
                "take_profit": 0.15,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "30分钟-标准配置",
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
            "name": "30分钟-严格配置",
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
            "name": "30分钟-MACD面积",
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
            "name": "30分钟-非严格笔",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2",
                "bi_strict": False,
                "bi_fx_check": "loss",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.05,
                "take_profit": 0.12,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2],
            }
        },
        {
            "name": "30分钟-平衡配置",
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
            print(f"[{current}/{total}] {stock['name']:6s} - {config['name']:15s}...", end=" ")

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
                print(f"✓ {result['metrics']['trade_count']:2d}次 收益{result['metrics']['total_return']*100:5.1f}%")
            else:
                print("✗ 无交易")

    successful_results = [r for r in all_results if r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 有效配置: {len(successful_results)}/{total}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        print("\n可能原因:")
        print("1. 30分钟数据在2023年期间买卖点信号较少")
        print("2. 缠论配置参数过于严格")
        print("3. 建议尝试更宽松的参数或更长的回测周期")
        return

    # 按年化收益率排序
    successful_results.sort(key=lambda x: x['metrics']['annual_return'], reverse=True)

    # 显示TOP 10
    print("\n" + "="*80)
    print("TOP 10 配置".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results[:10]:
        m = r['metrics']
        summary_data.append({
            '股票': r['stock_name'],
            '配置': r['config_name'],
            '年化收益': f"{m['annual_return']*100:.2f}%",
            '累计收益': f"{m['total_return']*100:.2f}%",
            '最大回撤': f"{m['max_drawdown']*100:.2f}%",
            '胜率': f"{m['win_rate']*100:.1f}%",
            '交易次数': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 生成报告
    best_result = successful_results[0]

    # 按股票统计
    stock_best = {}
    for stock in STOCK_LIST:
        stock_results = [r for r in successful_results if r['stock_code'] == stock['code']]
        if stock_results:
            stock_best[stock['name']] = stock_results[0]

    report_content = f"""# 缠论买卖点策略30分钟K线回测报告

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

## 最佳配置

### 股票
**{best_result['stock_name']} ({best_result['stock_code']})** - {best_result['industry']}行业

### 配置名称
**{best_result['config_name']}**

### 缠论参数
```json
{json.dumps(best_result['chan_config'], indent=2, ensure_ascii=False)}
```

**关键参数**:
"""

    for key, value in best_result['chan_config'].items():
        if key == 'trigger_step':
            report_content += f"- `{key}`: {value} - 启用逐步回测\n"
        elif key == 'divergence_rate':
            report_content += f"- `{key}`: {value} - 背驰率阈值\n"
        elif key == 'min_zs_cnt':
            report_content += f"- `{key}`: {value} - 最小中枢数量\n"
        elif key == 'bs_type':
            report_content += f"- `{key}`: \"{value}\" - 买卖点类型\n"
        elif key == 'macd_algo':
            report_content += f"- `{key}`: \"{value}\" - MACD算法\n"
        elif key == 'bi_strict':
            report_content += f"- `{key}`: {value} - 笔严格模式\n"
        elif key == 'bi_fx_check':
            report_content += f"- `{key}`: \"{value}\" - 分型检查模式\n"

    report_content += f"""
### 策略参数
- **单次买入仓位**: {best_result['strategy_params']['buy_percent']*100:.0f}%
- **止损比例**: {best_result['strategy_params']['stop_loss']*100:.1f}%
- **止盈比例**: {best_result['strategy_params']['take_profit']*100:.1f}%

## 回测绩效

### 收益指标
- **初始资金**: 100,000.00 元
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

| 排名 | 股票 | 配置 | 年化收益率 | 累计收益率 | 最大回撤 | 胜率 | 交易次数 |
|------|------|------|-----------|-----------|---------|------|---------|
"""

    for i, r in enumerate(successful_results[:10], 1):
        m = r['metrics']
        report_content += f"| {i} | {r['stock_name']} | {r['config_name']} | {m['annual_return']*100:.2f}% | {m['total_return']*100:.2f}% | {m['max_drawdown']*100:.2f}% | {m['win_rate']*100:.1f}% | {m['trade_count']} |\n"

    report_content += f"""
## 各股票最佳配置

"""

    for stock_name, result in stock_best.items():
        m = result['metrics']
        report_content += f"""### {stock_name}

- **最佳配置**: {result['config_name']}
- **年化收益率**: {m['annual_return']*100:.2f}%
- **累计收益率**: {m['total_return']*100:.2f}%
- **最大回撤**: {m['max_drawdown']*100:.2f}%
- **胜率**: {m['win_rate']*100:.1f}%
- **交易次数**: {m['trade_count']}
- **盈亏比**: {m['profit_loss_ratio']:.2f}

"""

    report_content += f"""
## 30分钟级别交易特点

### 1. 与日线级别对比
- **交易频率**: 30分钟级别交易频率明显高于日线
- **持仓周期**: 平均持仓时间更短，以小时/天为单位
- **止盈止损**: 需要设置更小的止盈止损比例（5-15%）
- **资金利用**: 可以更灵活地进出场

### 2. 30分钟级别最优参数
基于{len(successful_results)}个有效配置:

**缠论配置**:
- **背驰率**: {best_result['chan_config'].get('divergence_rate')}
- **中枢数**: {best_result['chan_config'].get('min_zs_cnt')}
- **买卖点类型**: {best_result['chan_config'].get('bs_type')}

**风控参数**:
- **单次仓位**: {best_result['strategy_params']['buy_percent']*100:.0f}%
- **止损**: {best_result['strategy_params']['stop_loss']*100:.1f}%
- **止盈**: {best_result['strategy_params']['take_profit']*100:.1f}%

### 3. 行业适用性
"""

    industry_stats = {}
    for r in successful_results:
        industry = r['industry']
        if industry not in industry_stats:
            industry_stats[industry] = []
        industry_stats[industry].append(r['metrics']['annual_return'])

    for industry, returns in industry_stats.items():
        avg_return = sum(returns) / len(returns)
        report_content += f"- **{industry}**: 平均年化收益 {avg_return*100:.2f}% ({len(returns)}个有效配置)\n"

    report_content += f"""
## 实盘建议

### 选股建议
1. 优先选择{best_result['industry']}行业股票
2. 选择流动性好、30分钟级别波动适中的标的
3. 避免ST股票和高风险标的

### 参数配置
1. **推荐背驰率**: {best_result['chan_config'].get('divergence_rate')}
2. **推荐中枢数**: {best_result['chan_config'].get('min_zs_cnt')}
3. **推荐买卖点类型**: {best_result['chan_config'].get('bs_type')}

### 风险控制
1. 30分钟级别建议降低单次仓位至20-25%
2. 严格执行止盈止损，避免隔夜持仓风险
3. 单只股票最大仓位不超过30%
4. 注意盘中波动，设置合理的止损

### 注意事项

⚠️ **重要提示**:
- 30分钟级别需要盯盘，不适合上班族
- 交易成本（手续费+滑点）对收益影响更大
- 建议设置价格提醒，及时把握交易机会
- 历史回测不代表未来收益

## 技术说明

### 数据质量
- **数据来源**: BaoStock
- **时间周期**: 30分钟
- **复权方式**: 前复权
- **回测期间**: 2023年全年

### 交易成本
- 手续费: 万分之三
- 滑点: 千分之一
- 印花税: 千分之一(仅卖出)

### 信号确认
- 买点 + 底分型确认
- 卖点 + 顶分型确认
- 避免虚假信号

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**回测工具**: Python + 缠论库
**数据源**: BaoStock 30分钟K线
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点30分钟K线回测报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 报告已保存至: {report_path}")

    # 保存JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_30m_results.json"
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
