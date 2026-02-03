"""
缠论买卖点策略 - 高效率版本

目标：提高交易频率，通过多条件组合过滤保持准确率
替代方案：3K确认 + 其他过滤条件
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


# 股票列表
STOCK_LIST = [
    {"code": "sz.000001", "name": "平安银行", "industry": "银行"},
    {"code": "sh.600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "sz.000333", "name": "美的集团", "industry": "家电"},
    {"code": "sh.600030", "name": "中信证券", "industry": "证券"},
    {"code": "sh.601318", "name": "中国平安", "industry": "保险"},
    {"code": "sz.000596", "name": "古井贡酒", "industry": "白酒"},
    {"code": "sz.000661", "name": "长春高新", "industry": "医药"},
    {"code": "sh.601899", "name": "紫金矿业", "industry": "有色"},
    {"code": "sh.601186", "name": "中国铁建", "industry": "建筑"},
    {"code": "sh.600089", "name": "特变电工", "industry": "电力设备"},
]


def create_chan_config():
    """缠论配置"""
    return {
        "trigger_step": True,
        "divergence_rate": 0.9,
        "min_zs_cnt": 1,
        "bs_type": "1,1p,2,2s",
    }


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


class TrendFilter:
    """趋势过滤器"""

    @staticmethod
    def is_uptrend(chan, lookback=20):
        """
        判断是否处于上升趋势
        简化版：比较最近N根K线的价格
        """
        cur_lv_chan = chan[0]
        if len(cur_lv_chan) < lookback:
            return True  # 数据不足，默认通过

        try:
            # 获取最近lookback根K线的收盘价
            recent_prices = []
            for seg in cur_lv_chan[-5:]:  # 取最近5个线段
                for bi in seg:
                    for klu in bi:
                        recent_prices.append(klu.close)
                        if len(recent_prices) >= lookback:
                            break
                    if len(recent_prices) >= lookback:
                        break
                if len(recent_prices) >= lookback:
                    break

            if len(recent_prices) < lookback:
                return True

            # 简单判断：最近价格 > 20周期前价格
            current_price = recent_prices[-1]
            past_price = recent_prices[0]
            return current_price > past_price
        except:
            return True


class BSPTypeFilter:
    """买卖点类型过滤器"""

    @staticmethod
    def should_skip_bsp(bsp, filter_config):
        """
        根据配置判断是否跳过某个买卖点
        """
        bsp_type_str = bsp.type2str()

        # 只做一类买点
        if filter_config.get('only_t1', True):
            if BSP_TYPE.T1 not in bsp.type and BSP_TYPE.T1P not in bsp.type:
                return True

        # 跳过一类买点延伸（1p）
        if filter_config.get('skip_1p', False):
            if BSP_TYPE.T1P in bsp.type:
                return True

        return False


def run_backtest_efficient(stock_code, stock_name, chan_config_dict, strategy_params, bsp_filter_config):
    """运行高效率回测"""

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

    buy_percent = strategy_params.get('buy_percent', 0.25)
    stop_loss = strategy_params.get('stop_loss', -0.05)
    take_profit = strategy_params.get('take_profit', 0.20)

    # 过滤配置
    only_t1 = bsp_filter_config.get('only_t1', True)
    need_confirm = bsp_filter_config.get('need_confirm', 3)
    skip_1p = bsp_filter_config.get('skip_1p', False)
    use_trend_filter = bsp_filter_config.get('use_trend_filter', False)

    processed_bsp_times = set()
    pending_bsps = []

    trend_filter = TrendFilter()
    bsp_type_filter = BSPTypeFilter()

    try:
        for chan_snapshot in chan.step_load():
            cur_lv_chan = chan_snapshot[0]

            if len(cur_lv_chan) < 2:
                continue

            current_price = cur_lv_chan[-1][-1].close
            current_time = str(cur_lv_chan[-1][-1].time)
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

                    # 买卖点类型过滤
                    filter_cfg = {
                        'only_t1': only_t1,
                        'skip_1p': skip_1p,
                    }
                    if bsp_type_filter.should_skip_bsp(bsp, filter_cfg):
                        continue

                    if bsp.is_buy and position == 0:
                        # 趋势过滤
                        if use_trend_filter:
                            if not trend_filter.is_uptrend(chan):
                                processed_bsp_times.add(bsp_time)
                                continue

                        pending_bsps.append({
                            'bsp': bsp,
                            'bsp_type_str': bsp.type2str(),
                            'kline_idx': current_kline_idx,
                        })
                        processed_bsp_times.add(bsp_time)
                    elif not bsp.is_buy and position > 0:
                        pending_bsps.append({
                            'bsp': bsp,
                            'bsp_type_str': bsp.type2str(),
                            'kline_idx': current_kline_idx,
                        })
                        processed_bsp_times.add(bsp_time)

            # 检查待确认的买卖点
            confirmed = []
            still_pending = []
            for pending in pending_bsps:
                klines_passed = current_kline_idx - pending['kline_idx']
                if klines_passed >= need_confirm:
                    # 检查买卖点是否仍存在
                    still_valid = False
                    if bsp_list:
                        for current_bsp in bsp_list:
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
            for confirmed_bsp in confirmed:
                if confirmed_bsp['bsp'].is_buy and position == 0:
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
                                'reason': f"{confirmed_bsp['bsp_type_str']}",
                                'profit': 0,
                                'profit_rate': 0
                            })
                elif not confirmed_bsp['bsp'].is_buy and position > 0:
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
                            'reason': f"{confirmed_bsp['bsp_type_str']}",
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
    print("="*80)
    print("缠论买卖点策略 - 高效率版本回测".center(80))
    print("="*80)

    # 基础策略参数
    strategy_params_base = {
        "buy_percent": 0.25,
        "stop_loss": -0.05,
        "take_profit": 0.20,
    }

    # 测试配置：高效率组合
    test_configs = []

    # 配置1：T1 + 3K确认 + 跳过1p + 趋势过滤
    test_configs.append({
        "name": "T1+3K+跳过1p+趋势",
        "bsp_filter_config": {
            "only_t1": True,
            "need_confirm": 3,
            "skip_1p": True,
            "use_trend_filter": True,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 配置2：T1 + 3K确认 + 跳过1p
    test_configs.append({
        "name": "T1+3K+跳过1p",
        "bsp_filter_config": {
            "only_t1": True,
            "need_confirm": 3,
            "skip_1p": True,
            "use_trend_filter": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 配置3：T1 + 5K确认 + 跳过1p
    test_configs.append({
        "name": "T1+5K+跳过1p",
        "bsp_filter_config": {
            "only_t1": True,
            "need_confirm": 5,
            "skip_1p": True,
            "use_trend_filter": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 配置4：T1 + 1K确认（最快）
    test_configs.append({
        "name": "T1+1K",
        "bsp_filter_config": {
            "only_t1": True,
            "need_confirm": 1,
            "skip_1p": False,
            "use_trend_filter": False,
        },
        "chan_config_fn": create_chan_config,
        "strategy_params": strategy_params_base.copy(),
    })

    # 配置5：对照组（T1+10K）
    test_configs.append({
        "name": "对照组T1+10K",
        "bsp_filter_config": {
            "only_t1": True,
            "need_confirm": 10,
            "skip_1p": False,
            "use_trend_filter": False,
        },
        "chan_config_fn": create_chan_config,
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
            print(f"[{current}/{total}] {stock['name']:8s} - {config['name']:16s}...", end=" ", flush=True)

            if current > 1:
                time.sleep(2)

            result = run_backtest_efficient(
                stock_code=stock['code'],
                stock_name=stock['name'],
                chan_config_dict=config['chan_config_fn'](),
                strategy_params=config['strategy_params'],
                bsp_filter_config=config['bsp_filter_config']
            )

            if result and result['metrics']['trade_count'] > 0:
                result['config_name'] = config['name']
                result['bsp_filter_config'] = config['bsp_filter_config']
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

    # 按方案分组统计
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

    # 按准确率排序
    sorted_configs = sorted(config_stats.items(), key=lambda x: sum(x[1]['acc'])/len(x[1]['acc']), reverse=True)

    print(f"{'方案':20s} | 准确率 | 年化 | 胜率 | 平均交易次数")
    print("-"*80)
    for config_name, stats in sorted_configs:
        avg_acc = sum(stats['acc']) / len(stats['acc'])
        avg_ret = sum(stats['ret']) / len(stats['ret'])
        avg_win = sum(stats['win']) / len(stats['win'])
        avg_trades = sum(stats['trades']) / len(stats['trades'])
        print(f"{config_name:20s} | {avg_acc*100:5.1f}%  | {avg_ret*100:4.1f}% | {avg_win*100:4.0f}% | {avg_trades:4.1f}次")

    # 效率对比
    print("\n" + "="*80)
    print("效率对比分析".center(80))
    print("="*80)
    print(f"{'方案':20s} | 交易次数 | 准确率 | 效率评分*")
    print("-"*80)

    for config_name, stats in sorted_configs:
        avg_trades = sum(stats['trades']) / len(stats['trades'])
        avg_acc = sum(stats['acc']) / len(stats['acc'])
        # 效率评分 = 交易次数 * 准确率（交易越多且准确率越高，效率越高）
        efficiency_score = avg_trades * avg_acc
        print(f"{config_name:20s} | {avg_trades:5.1f}次   | {avg_acc*100:5.1f}%  | {efficiency_score:5.2f}")

    print("\n*效率评分 = 平均交易次数 × 准确率（越高越好）")

    # 保存结果
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_efficient_results.json"
    serializable_results = []
    for r in successful_results:
        r_copy = {
            'stock_code': r['stock_code'],
            'stock_name': r['stock_name'],
            'config_name': r['config_name'],
            'bsp_filter_config': r['bsp_filter_config'],
            'metrics': r['metrics'],
            'accuracy': r['accuracy'],
        }
        serializable_results.append(r_copy)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细结果已保存至: {json_path}")
    print(f"\n{'='*80}")
    print("✅ 高效率版回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
