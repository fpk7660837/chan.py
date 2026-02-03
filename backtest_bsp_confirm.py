"""
缠论买卖点策略 - 买卖点确认机制回测

核心改进:
1. 买卖点延迟确认：识别出买卖点后，等待N根K线确认
2. 只有在确认后买卖点仍然存在时才执行交易
3. 避免基于后续被取消的买卖点进行交易
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, KL_TYPE
from datetime import datetime
import json
import pandas as pd
from collections import deque
import time


STOCK_LIST = [
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000333", "name": "美的集团", "industry": "家电"},
    {"code": "sh.600030", "name": "中信证券", "industry": "证券"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
]

# 为了避免BaoStock登录限制，每次运行只测试一只股票
import os
if os.getenv('TEST_ALL_STOCKS'):
    pass  # 测试所有股票
else:
    # 只测试第一只股票
    STOCK_LIST = STOCK_LIST[:1]


class BSPConfirmTracker:
    """买卖点确认跟踪器"""

    def __init__(self, confirm_klines=3):
        """
        Args:
            confirm_klines: 需要等待的K线数来确认买卖点
        """
        self.confirm_klines = confirm_klines
        self.pending_bsps = []  # 待确认的买卖点列表

    def add_pending_bsp(self, bsp, current_time, current_price, bsp_type_str):
        """添加待确认的买卖点"""
        self.pending_bsps.append({
            'bsp': bsp,
            'detected_time': current_time,
            'detected_price': current_price,
            'bsp_type_str': bsp_type_str,
            'kline_idx': None,  # 将在调用时设置
        })

    def check_confirmed_bsps(self, current_bsp_list, current_kline_idx):
        """
        检查待确认的买卖点是否被确认

        Args:
            current_bsp_list: 当前时刻的买卖点列表
            current_kline_idx: 当前K线索引

        Returns:
            list: 被确认的买卖点列表
        """
        confirmed = []
        still_pending = []

        for pending in self.pending_bsps:
            # 计算已经过了多少根K线
            if pending['kline_idx'] is None:
                pending['kline_idx'] = current_kline_idx
                still_pending.append(pending)
                continue

            klines_passed = current_kline_idx - pending['kline_idx']

            if klines_passed >= self.confirm_klines:
                # 已经过了确认期，检查买卖点是否仍存在
                original_bsp_time = str(pending['bsp'].klu.time)

                # 检查当前买卖点列表中是否还有这个点
                is_still_valid = False
                for current_bsp in current_bsp_list:
                    if str(current_bsp.klu.time) == original_bsp_time:
                        # 同一个时间点的买卖点
                        if (pending['bsp'].is_buy == current_bsp.is_buy and
                            any(t in current_bsp.type for t in pending['bsp'].type)):
                            is_still_valid = True
                            break

                if is_still_valid:
                    confirmed.append(pending)
                # 否则，买卖点已被取消，不添加到confirmed
            else:
                still_pending.append(pending)

        self.pending_bsps = still_pending
        return confirmed

    def clear(self):
        """清空待确认列表"""
        self.pending_bsps = []


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

                buy_signals.append({
                    'level': level,
                    'profit_rate': profit_rate,
                })

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

            sell_signals.append({
                'level': level,
                'profit_rate': profit_rate,
            })

    buy_correct = sum(1 for s in buy_signals if s['level'] == 'correct')
    buy_partial = sum(1 for s in buy_signals if s['level'] == 'partial')
    sell_correct = sum(1 for s in sell_signals if s['level'] == 'correct')
    sell_partial = sum(1 for s in sell_signals if s['level'] == 'partial')

    # 有效准确率 = 正确 + 0.5*部分正确
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
        'buy_partial': buy_partial,
        'sell_correct': sell_correct,
        'sell_partial': sell_partial,
    }


def run_backtest_with_confirm(stock_code, stock_name, chan_config_dict, strategy_params, confirm_klines=3):
    """运行带确认机制的回测"""

    begin_time = "2018-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    # 每次都创建新的配置对象，避免被复用
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
    target_bsp_types = strategy_params.get('bsp_types')

    # 创建确认跟踪器
    bsp_tracker = BSPConfirmTracker(confirm_klines=confirm_klines)

    # 记录已处理的买卖点时间
    processed_bsp_times = set()

    try:
        for chan_snapshot in chan.step_load():
            cur_lv_chan = chan_snapshot[0]

            if len(cur_lv_chan) < 2:
                continue

            current_price = cur_lv_chan[-1][-1].close
            current_time = str(cur_lv_chan[-1][-1].time)
            current_kline_idx = cur_lv_chan[-1].idx

            total_value = cash + position * current_price
            equity_curve.append({
                'time': current_time,
                'total_value': total_value,
            })

            # 止损止盈检查
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

            # 获取当前买卖点
            bsp_list = chan_snapshot.get_latest_bsp(number=10)
            if not bsp_list:
                # 即使没有买卖点，也要检查待确认的买卖点
                confirmed_bsps = bsp_tracker.check_confirmed_bsps([], current_kline_idx)
                # 执行已确认的买卖点交易
                for confirmed in confirmed_bsps:
                    if confirmed['bsp'].is_buy and position == 0:
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
                                    'reason': f"{confirmed['bsp_type_str']} (已确认)",
                                    'profit': 0,
                                    'profit_rate': 0
                                })
                continue

            # 检查新的买卖点并添加到待确认列表
            for bsp in bsp_list:
                bsp_time = str(bsp.klu.time)

                # 跳过已处理过的买卖点
                if bsp_time in processed_bsp_times:
                    continue

                # 检查买卖点类型
                if not any(t in bsp.type for t in target_bsp_types):
                    continue

                # 只处理符合条件的买卖点
                if bsp.is_buy and position == 0:
                    bsp_tracker.add_pending_bsp(bsp, current_time, current_price, bsp.type2str())
                    processed_bsp_times.add(bsp_time)
                elif not bsp.is_buy and position > 0:
                    bsp_tracker.add_pending_bsp(bsp, current_time, current_price, bsp.type2str())
                    processed_bsp_times.add(bsp_time)

            # 检查待确认的买卖点
            confirmed_bsps = bsp_tracker.check_confirmed_bsps(bsp_list, current_kline_idx)

            # 执行已确认的买卖点交易
            for confirmed in confirmed_bsps:
                bsp = confirmed['bsp']

                # 买入信号
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
                                'reason': f"{confirmed['bsp_type_str']} (已确认)",
                                'profit': 0,
                                'profit_rate': 0
                            })

                # 卖出信号
                elif not bsp.is_buy and position > 0:
                    profit_rate = (current_price - cost_price) / cost_price

                    # 只有盈利>3%才按卖点卖出
                    if profit_rate > 0.03:
                        sell_value = position * current_price * 0.999
                        cash += sell_value
                        profit = sell_value - position * cost_price

                        trades.append({
                            'time': current_time,
                            'type': 'sell',
                            'price': current_price,
                            'volume': position,
                            'reason': f"{confirmed['bsp_type_str']} (已确认)",
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
        'trades': trades[:200],
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略 - 确认机制回测".center(80))
    print("="*80)

    # 测试不同的确认K线数
    test_configs = []

    def create_chan_config():
        """每次返回新的配置字典，避免被ConfigWithCheck修改"""
        return {
            "trigger_step": True,
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
        }

    for confirm_klines in [1, 2, 3, 5]:
        test_configs.append({
            "name": f"确认{confirm_klines}根K线",
            "confirm_klines": confirm_klines,
            "chan_config_fn": create_chan_config,  # 使用函数而不是字典
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.05,
                "take_profit": 0.20,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        })

    all_results = []
    total = len(STOCK_LIST) * len(test_configs)
    current = 0

    print(f"\n数据源: BaoStock")
    print(f"时间周期: 日线 (K_DAY)")
    print(f"回测区间: 2018-01-01 至 2023-12-31 (6年)")
    print(f"总测试: {total} 个配置 ({len(STOCK_LIST)}只股票 × {len(test_configs)}个确认配置)\n")

    for stock in STOCK_LIST:
        for config in test_configs:
            current += 1
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:12s}...", end=" ", flush=True)

            # 添加延迟避免BaoStock登录限制
            if current > 1:
                time.sleep(2)

            result = run_backtest_with_confirm(
                stock_code=stock['code'],
                stock_name=stock['name'],
                chan_config_dict=config['chan_config_fn'](),  # 调用函数获取新配置
                strategy_params=config['strategy_params'],
                confirm_klines=config['confirm_klines']
            )

            if result and result['metrics']['trade_count'] > 0:
                result['config_name'] = config['name']
                result['confirm_klines'] = config['confirm_klines']
                result['industry'] = stock['industry']
                result['chan_config'] = config['chan_config_fn']()  # 获取配置的快照
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
            '确认K线': r['confirm_klines'],
            '准确率': f"{a['overall_accuracy']*100:.0f}%",
            '买点准确': f"{a['buy_accuracy']*100:.0f}%",
            '卖点准确': f"{a['sell_accuracy']*100:.0f}%",
            '年化收益': f"{m['annual_return']*100:6.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
            '买点信号': a['buy_signal_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 按确认K线数分组分析
    print("\n" + "="*80)
    print("按确认K线数分组分析".center(80))
    print("="*80)

    for confirm_klines in [1, 2, 3, 5]:
        group = [r for r in successful_results if r['confirm_klines'] == confirm_klines]
        if group:
            avg_acc = sum(r['accuracy']['overall_accuracy'] for r in group) / len(group)
            avg_return = sum(r['metrics']['annual_return'] for r in group) / len(group)
            avg_winrate = sum(r['metrics']['win_rate'] for r in group) / len(group)
            avg_trades = sum(r['metrics']['trade_count'] for r in group) / len(group)
            print(f"确认{confirm_klines}根K线: 平均准确率{avg_acc*100:.1f}%, "
                  f"年化收益{avg_return*100:5.1f}%, 胜率{avg_winrate*100:4.0f}%, "
                  f"平均交易{avg_trades:.1f}次")

    # 保存结果
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_confirm_results.json"
    serializable_results = []
    for r in successful_results:
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'industry': r['industry'],
            'config_name': r['config_name'],
            'confirm_klines': r['confirm_klines'],
            'chan_config': r['chan_config'],
            'strategy_params': {
                k: str(v) if isinstance(v, list) else v
                for k, v in r['strategy_params'].items()
            },
            'metrics': r['metrics'],
            'accuracy': r['accuracy'],
        }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细结果已保存至: {json_path}")
    print(f"\n{'='*80}")
    print("✅ 确认机制回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
