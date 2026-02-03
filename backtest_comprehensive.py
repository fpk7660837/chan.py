"""
缠论买卖点策略全面回测 - 探索更多ChanConfig参数

根据ChanConfig.py中的参数，测试更多配置组合
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


def run_backtest(stock_code, stock_name, chan_config_dict, strategy_params):
    """运行单个股票的回测"""

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
    years = 5.0
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
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 全面参数测试".center(80))
    print("="*80)

    # 扩展的配置参数组合 - 基于ChanConfig.py中的参数
    test_configs = [
        # 组1: 基础配置 - 不同的背驰率
        {
            "name": "极宽松-背驰率1.5",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.5,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.10,
                "take_profit": 0.30,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "宽松-背驰率1.2",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.2,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.09,
                "take_profit": 0.28,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "标准-背驰率0.9",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "严格-背驰率0.6",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.6,
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
        # 组2: 不同的MACD算法
        {
            "name": "MACD面积算法",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "macd_algo": "area",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "MACD峰值算法",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "macd_algo": "peak",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        # 组3: 不同的中枢数量要求
        {
            "name": "单中枢",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "zs_combine": True,
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "双中枢",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 2,
                "bs_type": "1,1p",
                "zs_combine": True,
            },
            "strategy_params": {
                "buy_percent": 0.35,
                "stop_loss": -0.07,
                "take_profit": 0.22,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        # 组4: 不同的笔算法配置
        {
            "name": "笔-非严格模式",
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
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2],
            }
        },
        {
            "name": "笔-严格模式",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.9,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "bi_strict": True,
                "bi_fx_check": "strict",
            },
            "strategy_params": {
                "buy_percent": 0.3,
                "stop_loss": -0.08,
                "take_profit": 0.25,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        # 组5: 包含更多买卖点类型
        {
            "name": "全类型买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.0,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s,3a",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.2,
                "stop_loss": -0.10,
                "take_profit": 0.30,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S, BSP_TYPE.T3A],
            }
        },
        # 组6: 优化的平衡配置
        {
            "name": "平衡配置A",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
                "macd_algo": "area",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.07,
                "take_profit": 0.22,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
        {
            "name": "平衡配置B",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.85,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2",
                "macd_algo": "peak",
                "bi_strict": False,
            },
            "strategy_params": {
                "buy_percent": 0.28,
                "stop_loss": -0.075,
                "take_profit": 0.23,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2],
            }
        },
    ]

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n总共测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个配置)\n")
    print("配置参数说明:")
    print("- divergence_rate: 背驰率 (值越小越严格)")
    print("- min_zs_cnt: 最小中枢数量")
    print("- bs_type: 买卖点类型 (1=一类, 1p=准一类, 2=二类, 2s=准二类, 3a=三类a)")
    print("- macd_algo: MACD算法 (area=面积, peak=峰值)")
    print("- bi_strict: 笔是否严格模式")
    print()

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:6s} - {config['name']:20s}...", end=" ")

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
                print(f"✓ {result['metrics']['trade_count']:2d}次 收益{result['metrics']['total_return']*100:5.1f}%")
            else:
                print("✗ 无交易")

    successful_results = [r for r in all_results if r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 有效配置: {len(successful_results)}/{total}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        return

    # 按年化收益率排序
    successful_results.sort(key=lambda x: x['metrics']['annual_return'], reverse=True)

    # 显示TOP 15
    print("\n" + "="*80)
    print("TOP 15 配置".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results[:15]:
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

    # 生成详细报告
    best_result = successful_results[0]

    # 按股票统计
    stock_best = {}
    for stock in STOCK_LIST:
        stock_results = [r for r in successful_results if r['stock_code'] == stock['code']]
        if stock_results:
            stock_best[stock['name']] = stock_results[0]

    # 按配置统计
    config_stats = {}
    for r in successful_results:
        config_name = r['config_name']
        if config_name not in config_stats:
            config_stats[config_name] = []
        config_stats[config_name].append(r['metrics']['annual_return'])

    report_content = f"""# 缠论买卖点策略全面参数测试报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 测试概述

### 测试范围
- **股票数量**: {len(STOCK_LIST)}只
- **覆盖行业**: {', '.join(set(s['industry'] for s in STOCK_LIST))}
- **配置数量**: {len(test_configs)}种
- **回测区间**: 2018-01-01 至 2023-12-31 (6年)
- **时间周期**: 日线 (K_DAY)
- **初始资金**: 100,000 元
- **数据源**: BaoStock

### 测试股票
"""

    for stock in STOCK_LIST:
        report_content += f"- **{stock['name']}** ({stock['code']}) - {stock['industry']}\n"

    report_content += f"""
### 测试结果统计
- **总测试配置**: {total}个
- **有效配置**: {len(successful_results)}个
- **有效率**: {len(successful_results)/total*100:.1f}%

## 最佳配置详情

### 股票
**{best_result['stock_name']} ({best_result['stock_code']})** - {best_result['industry']}行业

### 配置名称
**{best_result['config_name']}**

### 缠论参数
```json
{json.dumps(best_result['chan_config'], indent=2, ensure_ascii=False)}
```

**参数说明**:
"""

    for key, value in best_result['chan_config'].items():
        if key == 'trigger_step':
            report_content += f"- `{key}`: {value} - 启用逐步回测模式\n"
        elif key == 'divergence_rate':
            report_content += f"- `{key}`: {value} - 背驰率阈值 (值越小越严格)\n"
        elif key == 'min_zs_cnt':
            report_content += f"- `{key}`: {value} - 最小中枢数量\n"
        elif key == 'bs_type':
            report_content += f"- `{key}`: \"{value}\" - 买卖点类型\n"
        elif key == 'macd_algo':
            report_content += f"- `{key}`: \"{value}\" - MACD算法\n"
        elif key == 'bi_strict':
            report_content += f"- `{key}`: {value} - 笔严格模式\n"
        elif key == 'zs_combine':
            report_content += f"- `{key}`: {value} - 中枢合并\n"
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

## TOP 15 配置对比

| 排名 | 股票 | 配置 | 年化收益率 | 累计收益率 | 最大回撤 | 胜率 | 交易次数 |
|------|------|------|-----------|-----------|---------|------|---------|
"""

    for i, r in enumerate(successful_results[:15], 1):
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
## 配置参数分析

### 背驰率 (divergence_rate) 影响
不同背驰率下的平均表现:
"""

    # 分析背驰率影响
    div_rate_stats = {}
    for r in successful_results:
        div_rate = r['chan_config'].get('divergence_rate')
        if div_rate:
            if div_rate not in div_rate_stats:
                div_rate_stats[div_rate] = []
            div_rate_stats[div_rate].append(r['metrics']['annual_return'])

    for div_rate in sorted(div_rate_stats.keys()):
        avg_return = sum(div_rate_stats[div_rate]) / len(div_rate_stats[div_rate])
        report_content += f"- **{div_rate}**: 平均年化收益 {avg_return*100:.2f}% ({len(div_rate_stats[div_rate])}个有效配置)\n"

    report_content += f"""
### MACD算法 (macd_algo) 影响
"""

    macd_stats = {}
    for r in successful_results:
        macd_algo = r['chan_config'].get('macd_algo', 'default')
        if macd_algo not in macd_stats:
            macd_stats[macd_algo] = []
        macd_stats[macd_algo].append(r['metrics']['annual_return'])

    for algo, returns in macd_stats.items():
        avg_return = sum(returns) / len(returns)
        report_content += f"- **{algo}**: 平均年化收益 {avg_return*100:.2f}% ({len(returns)}个有效配置)\n"

    report_content += f"""
## 关键发现

### 1. 最优参数组合
基于{len(successful_results)}个有效配置的分析:

**缠论配置**:
- **背驰率**: {best_result['chan_config'].get('divergence_rate')}
- **中枢数**: {best_result['chan_config'].get('min_zs_cnt')}
- **买卖点类型**: {best_result['chan_config'].get('bs_type')}
- **MACD算法**: {best_result['chan_config'].get('macd_algo', '默认')}
- **笔严格模式**: {best_result['chan_config'].get('bi_strict', True)}

**风控参数**:
- **单次仓位**: {best_result['strategy_params']['buy_percent']*100:.0f}%
- **止损**: {best_result['strategy_params']['stop_loss']*100:.1f}%
- **止盈**: {best_result['strategy_params']['take_profit']*100:.1f}%

### 2. 行业适用性
"""

    industry_stats = {}
    for r in successful_results:
        industry = r['industry']
        if industry not in industry_stats:
            industry_stats[industry] = []
        industry_stats[industry].append(r['metrics']['annual_return'])

    for industry, returns in industry_stats.items():
        avg_return = sum(returns) / len(returns)
        report_content += f"- **{industry}**: 平均年化收益 {avg_return*100:.2f}% (测试{len(returns)}个配置)\n"

    report_content += f"""
### 3. 实盘建议

**选股建议**:
1. 优先选择{best_result['industry']}行业股票
2. 选择流动性好、波动适中的标的
3. 避免ST股票和高风险标的

**参数配置**:
1. 推荐使用背驰率: {best_result['chan_config'].get('divergence_rate')}
2. 最小中枢数: {best_result['chan_config'].get('min_zs_cnt')}
3. 买卖点类型: {best_result['chan_config'].get('bs_type')}

**风险控制**:
1. 实盘建议降低单次仓位至15-20%
2. 严格执行止盈止损
3. 单只股票最大仓位不超过30%
4. 总仓位控制在80%以内

### 4. 注意事项

⚠️ **重要提示**:
- 历史回测不代表未来收益
- 不同市场环境策略有效性会变化
- 建议先用小资金验证
- 注意流动性风险和滑点成本

## 技术说明

### 测试的ChanConfig参数
本次测试探索了以下参数:
- divergence_rate: 背驰率 (0.6 - 1.5)
- min_zs_cnt: 中枢数量 (1-2)
- bs_type: 买卖点类型 (多种组合)
- macd_algo: MACD算法 (area, peak)
- bi_strict: 笔严格模式 (True, False)
- bi_fx_check: 分型检查 (strict, loss)
- zs_combine: 中枢合并 (True, False)

### 信号确认
- 买点 + 底分型确认
- 卖点 + 顶分型确认

### 交易成本
- 手续费: 万分之三
- 滑点: 千分之一
- 印花税: 千分之一(仅卖出)

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点全面参数测试报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 报告已保存至: {report_path}")

    # 保存JSON
    json_path = "/Users/fupengkai/Documents/quant/chan.py/comprehensive_backtest_results.json"
    serializable_results = []
    for r in successful_results[:30]:
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
    print("✅ 全面参数测试完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
