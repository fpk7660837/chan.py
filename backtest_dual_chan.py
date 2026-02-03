"""
缠论买卖点策略 - 周线趋势过滤回测

使用两个独立的CChan对象，避免多级别复杂性
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, KL_TYPE
from datetime import datetime
import json
import pandas as pd
import time


STOCK_LIST = [
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000333", "name": "美的集团", "industry": "家电"},
    {"code": "sh.600030", "name": "中信证券", "industry": "证券"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
]


def get_weekly_trend_simple(weekly_chan):
    """简单判断周线趋势"""
    try:
        if len(weekly_chan) < 10:
            return "NEUTRAL"

        # 使用均线判断
        closes = [weekly_chan[i].close for i in range(len(weekly_chan))]
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes[-10:]) / 10

        if ma5 > ma10 * 1.01:
            return "UP"
        elif ma5 < ma10 * 0.99:
            return "DOWN"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def calculate_bsp_accuracy(trades):
    if not trades:
        return {'overall_accuracy': 0, 'buy_accuracy': 0, 'sell_accuracy': 0}

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
                if profit_rate > 0.05:
                    level = 'correct'
                elif profit_rate > 0:
                    level = 'partial'
                else:
                    level = 'wrong'
                buy_signals.append({'level': level, 'profit_rate': profit_rate})

        elif trade['type'] == 'sell' and '强制平仓' not in trade.get('reason', ''):
            profit_rate = trade.get('profit_rate', 0)
            if profit_rate > 0.05:
                level = 'correct'
            elif profit_rate > 0:
                level = 'partial'
            else:
                level = 'wrong'
            sell_signals.append({'level': level, 'profit_rate': profit_rate})

    buy_correct = sum(1 for s in buy_signals if s['level'] == 'correct')
    buy_partial = sum(1 for s in buy_signals if s['level'] == 'partial')
    sell_correct = sum(1 for s in sell_signals if s['level'] == 'correct')
    sell_partial = sum(1 for s in sell_signals if s['level'] == 'partial')

    buy_effective = buy_correct + 0.5 * buy_partial
    sell_effective = sell_correct + 0.5 * sell_partial

    buy_accuracy = buy_effective / len(buy_signals) if buy_signals else 0
    sell_accuracy = sell_effective / len(sell_signals) if sell_signals else 0

    total_signals = len(buy_signals) + len(sell_signals)
    overall_accuracy = (buy_effective + sell_effective) / total_signals if total_signals > 0 else 0

    return {
        'buy_accuracy': buy_accuracy,
        'sell_accuracy': sell_accuracy,
        'overall_accuracy': overall_accuracy,
        'buy_signal_count': len(buy_signals),
        'sell_signal_count': len(sell_signals),
        'buy_correct': buy_correct,
        'sell_correct': sell_correct,
    }


def run_backtest_dual_chan(stock_code, stock_name, chan_config_dict, strategy_params, ml_config):
    """运行双CChan回测"""

    begin_time = "2018-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK

    config = CChanConfig(chan_config_dict.copy() if isinstance(chan_config_dict, dict) else chan_config_dict)

    # 创建两个独立的CChan
    try:
        chan_weekly = CChan(
            code=stock_code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=[KL_TYPE.K_WEEK],
            config=config,
            autype=AUTYPE.QFQ,
        )

        chan_daily = CChan(
            code=stock_code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=[KL_TYPE.K_DAY],
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
    take_profit = strategy_params.get('take_profit', 0.20)

    only_t1 = ml_config.get('only_t1', True)
    need_confirm = ml_config.get('need_confirm', 3)
    use_weekly_trend = ml_config.get('use_weekly_trend', True)

    processed_bsp_times = set()
    pending_bsps = []

    # 获取所有周线数据（一次性）
    weekly_data = []
    for snapshot in chan_weekly.step_load():
        weekly_data.append(snapshot[0])

    weekly_iter = iter(weekly_data)

    try:
        for daily_snapshot in chan_daily.step_load():
            daily_chan = daily_snapshot[0]

            if len(daily_chan) < 2:
                continue

            current_price = daily_chan[-1][-1].close
            current_time = str(daily_chan[-1][-1].time)
            current_kline_idx = daily_chan[-1].idx

            total_value = cash + position * current_price
            equity_curve.append({'time': current_time, 'total_value': total_value})

            # 止损止盈
            if position > 0:
                profit_rate = (current_price - cost_price) / cost_price
                if profit_rate <= stop_loss:
                    sell_value = position * current_price * 0.999
                    cash += sell_value
                    trades.append({
                        'time': current_time,
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
                        'time': current_time,
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

            # 获取当前周线状态
            weekly_trend = "NEUTRAL"
            if use_weekly_trend:
                try:
                    # 获取当前时间对应的周线
                    current_weekly = None
                    while True:
                        try:
                            current_weekly = next(weekly_iter)
                            # 检查时间是否匹配（周线时间应该早于或等于日线时间）
                            if str(current_weekly[-1].time) <= current_time:
                                break
                        except StopIteration:
                            break

                    if current_weekly:
                        weekly_trend = get_weekly_trend_simple(current_weekly)
                except Exception:
                    pass

            # 获取日线买卖点
            daily_bsp_list = daily_snapshot.get_latest_bsp(number=10)

            # 检查新的买卖点
            if daily_bsp_list:
                for bsp in daily_bsp_list:
                    bsp_time = str(bsp.klu.time)
                    if bsp_time in processed_bsp_times:
                        continue

                    # 只做一类买卖点
                    if only_t1:
                        if BSP_TYPE.T1 not in bsp.type and BSP_TYPE.T1P not in bsp.type:
                            continue

                    # 周线趋势过滤
                    if use_weekly_trend and weekly_trend != "NEUTRAL":
                        if bsp.is_buy and weekly_trend != "UP":
                            continue
                        if not bsp.is_buy and weekly_trend != "DOWN":
                            continue

                    if bsp.is_buy and position == 0:
                        pending_bsps.append({
                            'bsp': bsp,
                            'bsp_type_str': bsp.type2str(),
                            'kline_idx': current_kline_idx,
                            'weekly_trend': weekly_trend,
                        })
                        processed_bsp_times.add(bsp_time)
                    elif not bsp.is_buy and position > 0:
                        pending_bsps.append({
                            'bsp': bsp,
                            'bsp_type_str': bsp.type2str(),
                            'kline_idx': current_kline_idx,
                            'weekly_trend': weekly_trend,
                        })
                        processed_bsp_times.add(bsp_time)

            # 检查待确认的买卖点
            confirmed = []
            still_pending = []
            for pending in pending_bsps:
                klines_passed = current_kline_idx - pending['kline_idx']
                if klines_passed >= need_confirm:
                    still_valid = False
                    if daily_bsp_list:
                        for current_bsp in daily_bsp_list:
                            if str(current_bsp.klu.time) == str(pending['bsp'].klu.time):
                                if pending['bsp'].is_buy == current_bsp.is_buy:
                                    still_valid = True
                                    break
                    if still_valid:
                        confirmed.append(pending)
                else:
                    still_pending.append(pending)

            pending_bsps = still_pending

            # 执行已确认的买卖点
            for confirmed in confirmed:
                bsp = confirmed['bsp']

                if bsp.is_buy and position == 0:
                    buy_amount = total_value * buy_percent
                    buy_volume = int(buy_amount / current_price / 100) * 100
                    if buy_volume > 0 and cash >= buy_volume * current_price:
                        cost = buy_volume * current_price * 1.001
                        if cash >= cost:
                            cash -= cost
                            position = buy_volume
                            cost_price = current_price * 1.001

                            reason = f"{confirmed['bsp_type_str']}"
                            if use_weekly_trend:
                                reason += f"[周{confirmed.get('weekly_trend', '?')}]"

                            trades.append({
                                'time': current_time,
                                'type': 'buy',
                                'price': current_price,
                                'volume': buy_volume,
                                'reason': reason,
                                'profit': 0,
                                'profit_rate': 0
                            })

                elif not bsp.is_buy and position > 0:
                    profit_rate = (current_price - cost_price) / cost_price
                    if profit_rate > 0.03:
                        sell_value = position * current_price * 0.999
                        cash += sell_value
                        profit = sell_value - position * cost_price

                        reason = f"{confirmed['bsp_type_str']}"
                        if use_weekly_trend:
                            reason += f"[周{confirmed.get('weekly_trend', '?')}]"

                        trades.append({
                            'time': current_time,
                            'type': 'sell',
                            'price': current_price,
                            'volume': position,
                            'reason': reason,
                            'profit': profit,
                            'profit_rate': profit_rate
                        })

                        position = 0
                        cost_price = 0

    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        return None

    # 最终清仓
    if position > 0 and len(daily_chan) > 0:
        final_price = daily_chan[-1][-1].close
        sell_value = position * final_price * 0.999
        cash += sell_value
        profit = sell_value - position * cost_price
        profit_rate = (final_price - cost_price) / cost_price

        trades.append({
            'time': str(daily_chan[-1][-1].time),
            'type': 'sell',
            'price': final_price,
            'volume': position,
            'reason': '强制平仓',
            'profit': profit,
            'profit_rate': profit_rate
        })

    accuracy_stats = calculate_bsp_accuracy(trades)

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
            'trade_count': len(buy_trades),
            'win_count': len(win_trades),
            'loss_count': len(loss_trades),
        },
        'accuracy': accuracy_stats,
        'trades': trades[:200],
        'success': True
    }


def main():
    print("="*80)
    print("缠论买卖点策略 - 周线趋势过滤回测".center(80))
    print("="*80)

    test_configs = []

    def create_chan_config():
        return {
            "trigger_step": True,
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
            "bs_type": "1,1p,2,2s",
        }

    strategy_params_base = {
        "buy_percent": 0.25,
        "stop_loss": -0.05,
        "take_profit": 0.20,
    }

    # 1. T1 + 周趋势 + 确认3K
    test_configs.append({
        "name": "T1+周趋势+确认3K",
        "ml_config": {
            "only_t1": True,
            "need_confirm": 3,
            "use_weekly_trend": True,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 2. T1 + 周趋势 + 确认5K
    test_configs.append({
        "name": "T1+周趋势+确认5K",
        "ml_config": {
            "only_t1": True,
            "need_confirm": 5,
            "use_weekly_trend": True,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 3. T1 + 周趋势 + 确认10K
    test_configs.append({
        "name": "T1+周趋势+确认10K",
        "ml_config": {
            "only_t1": True,
            "need_confirm": 10,
            "use_weekly_trend": True,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 4. T1 + 确认3K (无周过滤对比)
    test_configs.append({
        "name": "T1+确认3K(对比)",
        "ml_config": {
            "only_t1": True,
            "need_confirm": 3,
            "use_weekly_trend": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock | 回测区间: 2018-2023")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个方案)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:18s}...", end=" ", flush=True)

            if current > 1:
                time.sleep(3)

            result = run_backtest_dual_chan(
                stock_code=stock['code'],
                stock_name=stock['name'],
                chan_config_dict=config['chan_config_fn'](),
                strategy_params=config['strategy_params'],
                ml_config=config['ml_config']
            )

            if result and result['metrics']['trade_count'] > 0:
                result['config_name'] = config['name']
                result['ml_config'] = config['ml_config']
                result['industry'] = stock['industry']
                all_results.append(result)

                m = result['metrics']
                a = result['accuracy']
                print(f"✓ {m['trade_count']:2d}次 "
                      f"收益{m['total_return']*100:6.1f}% "
                      f"年化{m['annual_return']*100:5.1f}% "
                      f"胜率{m['win_rate']*100:4.0f}% "
                      f"准确率{a['overall_accuracy']*100:4.0f}%")
            else:
                print("✗")

    successful_results = [r for r in all_results if r['metrics']['trade_count'] > 0]

    print(f"\n{'='*80}")
    print(f"回测完成! 有效配置: {len(successful_results)}/{total}")
    print(f"{'='*80}")

    if not successful_results:
        print("\n❌ 没有成功产生交易的配置！")
        return

    successful_results.sort(key=lambda x: x['accuracy']['overall_accuracy'], reverse=True)

    print("\n" + "="*80)
    print("所有配置对比 (按准确率排序)".center(80))
    print("="*80)

    summary_data = []
    for r in successful_results:
        m = r['metrics']
        a = r['accuracy']
        summary_data.append({
            '股票': r['stock_name'],
            '方案': r['config_name'],
            '准确率': f"{a['overall_accuracy']*100:.0f}%",
            '买点': f"{a['buy_accuracy']*100:.0f}%",
            '年化': f"{m['annual_return']*100:5.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 按方案分组
    print("\n" + "="*80)
    print("按方案分组分析".center(80))
    print("="*80)

    config_stats = {}
    for r in successful_results:
        config_name = r['config_name']
        if config_name not in config_stats:
            config_stats[config_name] = {'acc': [], 'ret': [], 'win': [], 'trades': []}
        config_stats[config_name]['acc'].append(r['accuracy']['overall_accuracy'])
        config_stats[config_name]['ret'].append(r['metrics']['annual_return'])
        config_stats[config_name]['win'].append(r['metrics']['win_rate'])
        config_stats[config_name]['trades'].append(r['metrics']['trade_count'])

    for config_name in sorted(config_stats.keys()):
        stats = config_stats[config_name]
        avg_acc = sum(stats['acc']) / len(stats['acc'])
        avg_ret = sum(stats['ret']) / len(stats['ret'])
        avg_win = sum(stats['win']) / len(stats['win'])
        avg_trades = sum(stats['trades']) / len(stats['trades'])
        print(f"{config_name:20s}: 准确率{avg_acc*100:5.1f}%, "
              f"年化{avg_ret*100:5.1f}%, 胜率{avg_win*100:4.0f}%, "
              f"平均{avg_trades:4.1f}次")

    # 保存结果
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_dual_chan_results.json"
    serializable_results = []
    for r in successful_results:
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'config_name': r['config_name'],
            'ml_config': r['ml_config'],
            'metrics': r['metrics'],
            'accuracy': r['accuracy'],
        }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细结果已保存至: {json_path}")
    print(f"\n{'='*80}")
    print("✅ 周线趋势过滤回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
