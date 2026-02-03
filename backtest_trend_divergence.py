"""
缠论买卖点策略 - 趋势过滤+背驰买点回测

核心思路:
1. 大级别趋势过滤：只在周线趋势向上时做日线买点
2. 只做背驰买点：检查买卖点的背驰特征
3. 只在一类买点：最确定的买卖点
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, KL_TYPE, TREND_TYPE
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


class TrendDivergenceTracker:
    """趋势过滤+背驰买点跟踪器"""

    def __init__(self, use_trend_filter=True, use_divergence_only=True, t1_only=True):
        """
        Args:
            use_trend_filter: 是否使用大级别趋势过滤
            use_divergence_only: 是否只做背驰买点
            t1_only: 是否只做一类买卖点
        """
        self.use_trend_filter = use_trend_filter
        self.use_divergence_only = use_divergence_only
        self.t1_only = t1_only
        self.pending_bsps = []

    def add_pending_bsp(self, bsp, current_time, current_price, bsp_type_str, weekly_trend=None):
        """添加待确认的买卖点"""
        self.pending_bsps.append({
            'bsp': bsp,
            'detected_time': current_time,
            'detected_price': current_price,
            'bsp_type_str': bsp_type_str,
            'kline_idx': None,
            'weekly_trend': weekly_trend,
        })

    def check_has_divergence(self, bsp):
        """检查买卖点是否有背驰特征 - 改进版"""
        try:
            # 一类和准一类买卖点本身就是背驰买卖点
            if BSP_TYPE.T1 in bsp.type or BSP_TYPE.T1P in bsp.type:
                return True

            # 检查买卖点的特征
            if hasattr(bsp, 'features'):
                # 检查features中是否有背驰相关字段
                if hasattr(bsp.features, '__dict__'):
                    feature_dict = bsp.features.__dict__
                    # 检查各种可能的背驰字段名
                    if any(v for k, v in feature_dict.items()
                           if 'div' in k.lower() and v):
                        return True

        except Exception:
            pass
        return False

    def check_weekly_trend(self, weekly_chan):
        """检查周线趋势方向 - 改进版"""
        try:
            if not weekly_chan or len(weekly_chan) < 5:
                return "NEUTRAL"

            # 使用简单的MA判断趋势
            # 获取最后几根K线的收盘价
            closes = []
            for i in range(max(0, len(weekly_chan) - 10), len(weekly_chan)):
                if hasattr(weekly_chan[i], 'close'):
                    closes.append(weekly_chan[i].close)

            if len(closes) >= 5:
                # 简单判断：最近价格高于5个单位前 = 上升趋势
                recent_avg = sum(closes[-3:]) / 3
                prev_avg = sum(closes[-6:-3]) / 3 if len(closes) >= 6 else sum(closes[:-3]) / (len(closes) - 3)

                if recent_avg > prev_avg * 1.02:
                    return "UP"
                elif recent_avg < prev_avg * 0.98:
                    return "DOWN"

        except Exception as e:
            pass
        return "NEUTRAL"

    def check_trend_filter(self, bsp, weekly_trend):
        """检查是否符合大级别趋势过滤"""
        if not self.use_trend_filter:
            return True

        if not weekly_trend or weekly_trend == "NEUTRAL":
            return True  # 无法判断趋势时，允许交易

        # 买点：周线趋势向上
        if bsp.is_buy and weekly_trend == "UP":
            return True

        # 卖点：周线趋势向下
        if not bsp.is_buy and weekly_trend == "DOWN":
            return True

        return False

    def check_confirmed_bsps(self, current_bsp_list, cur_lv_chan, current_klu,
                            current_kline_idx, weekly_chan=None):
        """检查待确认的买卖点"""
        confirmed = []
        still_pending = []

        # 获取当前周线趋势
        current_weekly_trend = self.check_weekly_trend(weekly_chan)

        for pending in self.pending_bsps:
            if pending['kline_idx'] is None:
                pending['kline_idx'] = current_kline_idx
                still_pending.append(pending)
                continue

            # 至少等3根K线确认
            klines_passed = current_kline_idx - pending['kline_idx']
            if klines_passed < 3:
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

            # 2. 检查是否是一类买卖点（如果配置了）
            if self.t1_only:
                has_t1 = BSP_TYPE.T1 in matched_bsp.type or BSP_TYPE.T1P in matched_bsp.type
                if not has_t1:
                    still_pending.append(pending)
                    continue

            # 3. 检查是否有背驰（如果配置了）
            if self.use_divergence_only:
                has_div = self.check_has_divergence(matched_bsp)
                if not has_div:
                    still_pending.append(pending)
                    continue

            # 4. 检查大级别趋势过滤（如果配置了）
            if self.use_trend_filter:
                trend_ok = self.check_trend_filter(matched_bsp, current_weekly_trend)
                if not trend_ok:
                    still_pending.append(pending)
                    continue

            # 所有检查通过
            confirmed.append(pending)

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


def run_backtest_trend_divergence(stock_code, stock_name, chan_config_dict,
                                  strategy_params, tracker_config):
    """运行趋势过滤+背驰买点回测"""

    begin_time = "2018-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK

    # 同时加载日线和周线 - 注意顺序必须从大到小
    lv_list = [KL_TYPE.K_WEEK, KL_TYPE.K_DAY]

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

    bsp_tracker = TrendDivergenceTracker(
        use_trend_filter=tracker_config.get('use_trend_filter', True),
        use_divergence_only=tracker_config.get('use_divergence_only', True),
        t1_only=tracker_config.get('t1_only', True),
    )

    processed_bsp_times = set()

    try:
        for chan_snapshot in chan.step_load():
            # chan_snapshot是tuple，每个元素是一个CChan对象
            # [0]是周线，[1]是日线
            weekly_chan = chan_snapshot[0]
            cur_lv_chan = chan_snapshot[1]

            if len(cur_lv_chan) < 2:
                continue

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

            # 获取周线趋势
            weekly_trend = bsp_tracker.check_weekly_trend(weekly_chan)

            # 获取买卖点 - 从日线级别获取
            bsp_list = []
            if hasattr(cur_lv_chan, 'bsp_list') and cur_lv_chan.bsp_list:
                # 获取最近的买卖点
                bsp_list = cur_lv_chan.bsp_list[-10:] if len(cur_lv_chan.bsp_list) >= 10 else cur_lv_chan.bsp_list

            # 检查新的买卖点
            if bsp_list:
                for bsp in bsp_list:
                    bsp_time = str(bsp.klu.time)
                    if bsp_time in processed_bsp_times:
                        continue

                    if not any(t in bsp.type for t in target_bsp_types):
                        continue

                    # 先检查大级别趋势过滤
                    if tracker_config.get('use_trend_filter', True):
                        if not bsp_tracker.check_trend_filter(bsp, weekly_trend):
                            continue

                    if bsp.is_buy and position == 0:
                        bsp_tracker.add_pending_bsp(bsp, current_time, current_price,
                                                    bsp.type2str(), weekly_trend)
                        processed_bsp_times.add(bsp_time)
                    elif not bsp.is_buy and position > 0:
                        bsp_tracker.add_pending_bsp(bsp, current_time, current_price,
                                                    bsp.type2str(), weekly_trend)
                        processed_bsp_times.add(bsp_time)

            # 检查待确认的买卖点
            confirmed_bsps = bsp_tracker.check_confirmed_bsps(
                bsp_list if bsp_list else [],
                cur_lv_chan,
                current_klu,
                current_kline_idx,
                weekly_chan
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

                            reason = f"{confirmed['bsp_type_str']}"
                            if tracker_config.get('use_trend_filter'):
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
    print("缠论买卖点策略 - 趋势过滤+背驰买点回测".center(80))
    print("="*80)

    test_configs = []

    def create_chan_config():
        return {
            "trigger_step": True,
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
            "mean_metrics": [5, 10, 20],  # 添加均线用于趋势判断
        }

    strategy_params_base = {
        "buy_percent": 0.25,
        "stop_loss": -0.05,
        "take_profit": 0.20,
        "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
    }

    # 1. 完整策略：趋势过滤 + 只做背驰 + 只做一类
    test_configs.append({
        "name": "趋势+背驰+T1",
        "tracker_config": {
            "use_trend_filter": True,
            "use_divergence_only": True,
            "t1_only": True,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 2. 趋势过滤 + 只做背驰
    test_configs.append({
        "name": "趋势+背驰",
        "tracker_config": {
            "use_trend_filter": True,
            "use_divergence_only": True,
            "t1_only": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 3. 只做趋势过滤
    test_configs.append({
        "name": "仅趋势过滤",
        "tracker_config": {
            "use_trend_filter": True,
            "use_divergence_only": False,
            "t1_only": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 4. 只做背驰买点
    test_configs.append({
        "name": "仅背驰",
        "tracker_config": {
            "use_trend_filter": False,
            "use_divergence_only": True,
            "t1_only": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 5. 只做一类买卖点
    test_configs.append({
        "name": "仅T1",
        "tracker_config": {
            "use_trend_filter": False,
            "use_divergence_only": False,
            "t1_only": True,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock | 时间周期: 日线+周线 | 回测区间: 2018-2023")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个方案)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:12s}...", end=" ", flush=True)

            if current > 1:
                time.sleep(2)

            result = run_backtest_trend_divergence(
                stock_code=stock['code'],
                stock_name=stock['name'],
                chan_config_dict=config['chan_config_fn'](),
                strategy_params=config['strategy_params'],
                tracker_config=config['tracker_config']
            )

            if result and result['metrics']['trade_count'] > 0:
                result['config_name'] = config['name']
                result['tracker_config'] = config['tracker_config']
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
        print(f"{config_name:15s}: 准确率{avg_acc*100:5.1f}%, "
              f"年化{avg_ret*100:5.1f}%, 胜率{avg_win*100:4.0f}%, "
              f"平均{avg_trades:4.1f}次")

    # 保存结果
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_trend_divergence_results.json"
    serializable_results = []
    for r in successful_results:
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'industry': r['industry'],
            'config_name': r['config_name'],
            'tracker_config': r['tracker_config'],
            'metrics': r['metrics'],
            'accuracy': r['accuracy'],
        }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细结果已保存至: {json_path}")
    print(f"\n{'='*80}")
    print("✅ 趋势过滤+背驰买点回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
