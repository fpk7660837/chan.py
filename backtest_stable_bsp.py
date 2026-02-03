"""
缠论买卖点策略 - 稳定买卖点回测

核心思路：
1. 只使用"已确认"笔上的买卖点（is_sure=True）
2. 只在笔完全确定后才交易
3. 使用更严格的笔构造模式
4. 多重过滤确保买卖点稳定性
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


class StableBSPTracker:
    """稳定买卖点跟踪器"""

    def __init__(self, strategy_mode="strict"):
        """
        Args:
            strategy_mode: 策略模式
                - "strict": 只使用is_sure=True的笔上的买卖点
                - "confirm": 确认后仍需额外验证
                - "multi": 需要多个买卖点类型共存
        """
        self.strategy_mode = strategy_mode
        self.pending_bsps = []

    def add_pending_bsp(self, bsp, current_time, current_price, bsp_type_str):
        """添加待确认的买卖点"""
        self.pending_bsps.append({
            'bsp': bsp,
            'detected_time': current_time,
            'detected_price': current_price,
            'bsp_type_str': bsp_type_str,
            'kline_idx': None,
        })

    def check_bi_is_sure(self, bsp, cur_lv_chan):
        """检查买卖点所在的笔是否已确认"""
        try:
            bi = bsp.bi
            if hasattr(bi, '_CBi__is_sure'):
                return bi._CBi__is_sure
            elif hasattr(bi, 'is_sure'):
                return bi.is_sure
            else:
                # 如果没有is_sure属性，检查笔是否已经完全形成
                # 通过检查笔的终点是否是最新K线之前的K线
                if hasattr(cur_lv_chan, 'bi_list') and len(cur_lv_chan.bi_list) > 0:
                    last_bi = cur_lv_chan.bi_list[-1]
                    # 如果这个买卖点的笔不是最后一笔，说明已经确认
                    if hasattr(bi, 'idx'):
                        return bi.idx != last_bi.idx
        except Exception:
            pass
        return False

    def check_multiple_bsp_types(self, bsp):
        """检查是否有多个买卖点类型共存（更可靠）"""
        # 一类+二类，或者一类+三类等
        return len(bsp.type) >= 2

    def check_price_confirms(self, bsp, current_price, current_chan):
        """检查价格是否确认了买卖点"""
        try:
            if bsp.is_buy:
                # 买点：价格应该高于前一个高点（确认反转）
                bi = bsp.bi
                if hasattr(bi, 'pre') and bi.pre:
                    prev_high = bi.pre.end_klc.high
                    return current_price > prev_high
            else:
                # 卖点：价格应该低于前一个低点
                bi = bsp.bi
                if hasattr(bi, 'pre') and bi.pre:
                    prev_low = bi.pre.end_klc.low
                    return current_price < prev_low
        except Exception:
            pass
        return True

    def check_confirmed_bsps(self, current_bsp_list, cur_lv_chan, current_klu, current_kline_idx, current_price):
        """检查待确认的买卖点"""
        confirmed = []
        still_pending = []

        for pending in self.pending_bsps:
            if pending['kline_idx'] is None:
                pending['kline_idx'] = current_kline_idx
                still_pending.append(pending)
                continue

            # 至少等2根K线
            klines_passed = current_kline_idx - pending['kline_idx']
            if klines_passed < 2:
                still_pending.append(pending)
                continue

            original_bsp_time = str(pending['bsp'].klu.time)

            # 1. 检查买卖点是否仍存在
            is_still_valid = False
            matched_bsp = None
            for current_bsp in current_bsp_list:
                if str(current_bsp.klu.time) == original_bsp_time:
                    if (pending['bsp'].is_buy == current_bsp.is_buy and
                        any(t in current_bsp.type for t in pending['bsp'].type)):
                        is_still_valid = True
                        matched_bsp = current_bsp
                        break

            if not is_still_valid:
                continue

            # 2. 根据策略模式进行额外验证
            pass_all_checks = True

            if self.strategy_mode == "strict":
                # 严格模式：必须是基于已确认笔的买卖点
                if not self.check_bi_is_sure(matched_bsp, cur_lv_chan):
                    pass_all_checks = False

            elif self.strategy_mode == "multi":
                # 多重确认模式：需要多个买卖点类型或价格确认
                has_multi_type = self.check_multiple_bsp_types(matched_bsp)
                price_confirms = self.check_price_confirms(matched_bsp, current_price, cur_lv_chan)
                if not (has_multi_type or price_confirms):
                    pass_all_checks = False

            elif self.strategy_mode == "confirm":
                # 确认模式：只需要价格确认
                if not self.check_price_confirms(matched_bsp, current_price, cur_lv_chan):
                    pass_all_checks = False

            if pass_all_checks:
                confirmed.append(pending)
            else:
                still_pending.append(pending)

        self.pending_bsps = still_pending
        return confirmed


def calculate_bsp_accuracy(trades):
    """计算买卖点准确率"""
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
                reason = next_sell.get('reason', '')

                if '止盈' in reason or profit_rate > 0.05:
                    level = 'correct'
                elif profit_rate > 0:
                    level = 'partial'
                elif profit_rate > -0.03:
                    level = 'partial'
                else:
                    level = 'wrong'

                buy_signals.append({'level': level, 'profit_rate': profit_rate})

        elif trade['type'] == 'sell' and '强制平仓' not in trade.get('reason', ''):
            profit_rate = trade.get('profit_rate', 0)
            reason = trade.get('reason', '')

            if '止盈' in reason or profit_rate > 0.05:
                level = 'correct'
            elif profit_rate > 0:
                level = 'partial'
            elif '止损' in reason and profit_rate > -0.03:
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


def run_backtest_stable(stock_code, stock_name, chan_config_dict, strategy_params, strategy_mode="strict"):
    """运行稳定买卖点回测"""

    begin_time = "2018-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig(chan_config_dict.copy() if isinstance(chan_config_dict, dict) else chan_config_dict)

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
    take_profit = strategy_params.get('take_profit', 0.20)
    target_bsp_types = strategy_params.get('bsp_types', [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S])

    bsp_tracker = StableBSPTracker(strategy_mode=strategy_mode)
    processed_bsp_times = set()

    try:
        for chan_snapshot in chan.step_load():
            cur_lv_chan = chan_snapshot[0]

            if len(cur_lv_chan) < 2:
                continue

            current_price = cur_lv_chan[-1][-1].close
            current_time = str(cur_lv_chan[-1][-1].time)
            current_klu = cur_lv_chan[-1][-1]
            current_kline_idx = cur_lv_chan[-1].idx

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

            # 获取买卖点
            bsp_list = chan_snapshot.get_latest_bsp(number=10)

            # 检查新的买卖点
            if bsp_list:
                for bsp in bsp_list:
                    bsp_time = str(bsp.klu.time)
                    if bsp_time in processed_bsp_times:
                        continue

                    if not any(t in bsp.type for t in target_bsp_types):
                        continue

                    if bsp.is_buy and position == 0:
                        bsp_tracker.add_pending_bsp(bsp, current_time, current_price, bsp.type2str())
                        processed_bsp_times.add(bsp_time)
                    elif not bsp.is_buy and position > 0:
                        bsp_tracker.add_pending_bsp(bsp, current_time, current_price, bsp.type2str())
                        processed_bsp_times.add(bsp_time)

            # 检查待确认的买卖点
            confirmed_bsps = bsp_tracker.check_confirmed_bsps(
                bsp_list if bsp_list else [],
                cur_lv_chan,
                current_klu,
                current_kline_idx,
                current_price
            )

            # 执行已确认的交易
            for confirmed in confirmed_bsps:
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

                            trades.append({
                                'time': current_time,
                                'type': 'buy',
                                'price': current_price,
                                'volume': buy_volume,
                                'reason': f"{confirmed['bsp_type_str']}",
                                'profit': 0,
                                'profit_rate': 0
                            })

                elif not bsp.is_buy and position > 0:
                    profit_rate = (current_price - cost_price) / cost_price
                    if profit_rate > 0.03:
                        sell_value = position * current_price * 0.999
                        cash += sell_value
                        profit = sell_value - position * cost_price

                        trades.append({
                            'time': current_time,
                            'type': 'sell',
                            'price': current_price,
                            'volume': position,
                            'reason': f"{confirmed['bsp_type_str']}",
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
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 稳定买卖点回测".center(80))
    print("="*80)

    test_configs = []

    def create_chan_config_strict():
        return {
            "trigger_step": True,
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
            "bi_strict": True,  # 严格笔模式
        }

    def create_chan_config_loose():
        return {
            "trigger_step": True,
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
            "bi_strict": False,  # 宽松笔模式
        }

    strategy_params_base = {
        "buy_percent": 0.25,
        "stop_loss": -0.05,
        "take_profit": 0.20,
        "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
    }

    # 1. 严格模式 + 严格笔
    test_configs.append({
        "name": "严格模式+严格笔",
        "strategy_mode": "strict",
        "chan_config_fn": create_chan_config_strict,
        "strategy_params": strategy_params_base.copy(),
    })

    # 2. 严格模式 + 宽松笔
    test_configs.append({
        "name": "严格模式+宽松笔",
        "strategy_mode": "strict",
        "chan_config_fn": create_chan_config_loose,
        "strategy_params": strategy_params_base.copy(),
    })

    # 3. 多重确认模式
    test_configs.append({
        "name": "多重确认模式",
        "strategy_mode": "multi",
        "chan_config_fn": create_chan_config_strict,
        "strategy_params": strategy_params_base.copy(),
    })

    # 4. 价格确认模式
    test_configs.append({
        "name": "价格确认模式",
        "strategy_mode": "confirm",
        "chan_config_fn": create_chan_config_strict,
        "strategy_params": strategy_params_base.copy(),
    })

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock | 时间周期: 日线 | 回测区间: 2018-2023")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个方案)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:15s}...", end=" ", flush=True)

            if current > 1:
                time.sleep(1.5)

            result = run_backtest_stable(
                stock_code=stock['code'],
                stock_name=stock['name'],
                chan_config_dict=config['chan_config_fn'](),
                strategy_params=config['strategy_params'],
                strategy_mode=config['strategy_mode']
            )

            if result and result['metrics']['trade_count'] > 0:
                result['config_name'] = config['name']
                result['strategy_mode'] = config['strategy_mode']
                result['industry'] = stock['industry']
                result['chan_config'] = config['chan_config_fn']()
                result['strategy_params'] = config['strategy_params']
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

    # 按准确率排序
    successful_results.sort(key=lambda x: x['accuracy']['overall_accuracy'], reverse=True)

    print("\n" + "="*80)
    print("所有配置对比 (按买卖点准确率排序)".center(80))
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
            '卖点': f"{a['sell_accuracy']*100:.0f}%",
            '年化': f"{m['annual_return']*100:5.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 按方案分组分析
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
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_stable_results.json"
    serializable_results = []
    for r in successful_results:
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'industry': r['industry'],
            'config_name': r['config_name'],
            'strategy_mode': r['strategy_mode'],
            'metrics': r['metrics'],
            'accuracy': r['accuracy'],
        }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细结果已保存至: {json_path}")
    print(f"\n{'='*80}")
    print("✅ 稳定买卖点回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
