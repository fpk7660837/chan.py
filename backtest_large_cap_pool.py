"""
缠论买卖点策略 - 大中市值股票池回测

测试所有市值在100亿-500亿人民币之间的A股股票，使用最优的缠论策略配置：
- T1-only BSP过滤
- 10根K线确认
- 止损-5%，止盈+20%
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
import re
import os
import akshare as ak


def parse_market_cap(value_str):
    """解析市值字符串，转换为亿元单位"""
    if pd.isna(value_str):
        return 0
    match = re.search(r'([\d.]+)', str(value_str))
    if match:
        return float(match.group(1))
    return 0


def convert_code_to_baostock(code):
    """将akshare代码转换为BaoStock格式"""
    if code.startswith('6'):
        return f'sh.{code}'
    else:
        return f'sz.{code}'


def get_stocks_by_market_cap(min_cap=100, max_cap=500):
    """
    获取指定市值范围的股票列表

    过滤条件:
        1. 排除ST股票
        2. 排除科创板(688开头)
        3. 排除北交所(8、43开头)
        4. 排除B股(200、900开头)
        5. 排除CDR(920开头)
        6. 排除停牌股票
        7. 市值过滤

    Args:
        min_cap: 最小市值（亿元）
        max_cap: 最大市值（亿元）

    Returns:
        pd.DataFrame: 包含股票代码、名称、市值的列表
    """
    print("正在获取A股实时行情...")
    try:
        df = ak.stock_zh_a_spot_em()
        print(f"获取到 {len(df)} 只股票")

        # 1. 排除ST股票
        before_count = len(df)
        df = df[~df['名称'].str.contains('ST', case=False, na=False)]
        print(f"排除ST股票后: {len(df)} (剔除 {before_count - len(df)})")

        # 2. 排除科创板(688开头)
        before_count = len(df)
        df = df[~df['代码'].str.startswith('688')]
        print(f"排除科创板后: {len(df)} (剔除 {before_count - len(df)})")

        # 3. 排除北交所(8、43开头)
        before_count = len(df)
        df = df[~df['代码'].str.startswith('8')]
        df = df[~df['代码'].str.startswith('43')]
        print(f"排除北交所后: {len(df)} (剔除 {before_count - len(df)})")

        # 4. 排除B股(200、900开头)
        before_count = len(df)
        df = df[~df['代码'].str.startswith('200')]
        df = df[~df['代码'].str.startswith('900')]
        print(f"排除B股后: {len(df)} (剔除 {before_count - len(df)})")

        # 5. 排除CDR(920开头)
        before_count = len(df)
        df = df[~df['代码'].str.startswith('920')]
        print(f"排除CDR后: {len(df)} (剔除 {before_count - len(df)})")

        # 6. 排除停牌股票(成交量为0)
        before_count = len(df)
        df = df[df['成交量'] > 0]
        print(f"排除停牌股票后: {len(df)} (剔除 {before_count - len(df)})")

        # 7. 市值过滤（总市值在min_cap-max_cap亿之间）
        before_count = len(df)
        df['market_cap_num'] = df['总市值'].apply(parse_market_cap)
        df = df[(df['market_cap_num'] >= min_cap) & (df['market_cap_num'] <= max_cap)]
        print(f"市值{min_cap}-{max_cap}亿筛选后: {len(df)} (剔除 {before_count - len(df)})")

        # 转换为BaoStock格式
        df['baostock_code'] = df['代码'].apply(convert_code_to_baostock)

        return df[['baostock_code', '代码', '名称', 'market_cap_num']].reset_index(drop=True)
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return pd.DataFrame()


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


def run_backtest_simple(stock_code, stock_name, chan_config_dict, strategy_params, bsp_filter_config):
    """运行简化回测（复用backtest_simple_stable.py的逻辑）"""

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

    only_t1 = bsp_filter_config.get('only_t1', True)
    need_confirm = bsp_filter_config.get('need_confirm', 3)

    processed_bsp_times = set()
    pending_bsps = []
    total_bsp_seen = 0

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
                    total_bsp_seen += 1
                    bsp_time = str(bsp.klu.time)
                    if bsp_time in processed_bsp_times:
                        continue

                    # 只做一类买卖点
                    if only_t1:
                        if BSP_TYPE.T1 not in bsp.type and BSP_TYPE.T1P not in bsp.type:
                            continue

                    if bsp.is_buy and position == 0:
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


def create_chan_config():
    """创建缠论配置"""
    return {
        "trigger_step": True,
        "divergence_rate": 0.9,
        "min_zs_cnt": 1,
        "bs_type": "1,1p,2,2s",
    }


def load_progress(results_path):
    """加载已保存的进度"""
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                completed_codes = set(data.get('completed_codes', []))
                results = data.get('results', [])
                return completed_codes, results
        except Exception as e:
            print(f"加载进度失败: {e}")
    return set(), []


def save_progress(results_path, completed_codes, results):
    """保存进度"""
    try:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'completed_codes': list(completed_codes),
                'results': results,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存进度失败: {e}")


def run_batch_backtest(stock_list, config_name, chan_config_dict, strategy_params, bsp_filter_config, results_path):
    """批量回测，支持错误恢复和进度保存"""

    print(f"\n{'='*80}")
    print(f"批量回测 - {config_name}".center(80))
    print(f"{'='*80}")
    print(f"股票池: {len(stock_list)} 只")
    print(f"回测区间: 2018-2023 (日线)")
    print(f"策略: T1-only BSP + {bsp_filter_config['need_confirm']}K确认")
    print(f"止损: {strategy_params['stop_loss']*100:.0f}% | 止盈: {strategy_params['take_profit']*100:.0f}%")
    print(f"{'='*80}\n")

    # 加载进度
    completed_codes, results = load_progress(results_path)
    print(f"已完成的股票: {len(completed_codes)}")

    # 过滤已完成的股票
    pending_stocks = stock_list[~stock_list['baostock_code'].isin(completed_codes)]
    print(f"待处理股票: {len(pending_stocks)}\n")

    total = len(pending_stocks)
    success_count = 0
    fail_count = 0

    for idx, row in pending_stocks.iterrows():
        stock_code = row['baostock_code']
        stock_name = row['名称']
        market_cap = row['market_cap_num']

        print(f"[{idx+1}/{total}] {stock_code} {stock_name} ({market_cap:.0f}亿)...", end=" ", flush=True)

        # BaoStock限流：每个请求间隔3秒
        if idx > 0:
            time.sleep(3)

        result = run_backtest_simple(
            stock_code=stock_code,
            stock_name=stock_name,
            chan_config_dict=chan_config_dict,
            strategy_params=strategy_params,
            bsp_filter_config=bsp_filter_config
        )

        if result and result['metrics']['trade_count'] > 0:
            result['config_name'] = config_name
            result['bsp_filter_config'] = bsp_filter_config
            result['market_cap'] = market_cap
            results.append(result)
            completed_codes.add(stock_code)
            success_count += 1

            m = result['metrics']
            a = result['accuracy']
            print(f"✓ {m['trade_count']:2d}次 "
                  f"收益{m['total_return']*100:6.1f}% "
                  f"年化{m['annual_return']*100:5.1f}% "
                  f"胜率{m['win_rate']*100:4.0f}% "
                  f"准确率{a['overall_accuracy']*100:4.0f}%")
        else:
            fail_count += 1
            print("✗")

        # 每处理10只股票保存一次进度
        if (idx + 1) % 10 == 0:
            save_progress(results_path, completed_codes, results)
            print(f"  [进度已保存]")

    # 最终保存
    save_progress(results_path, completed_codes, results)

    print(f"\n{'='*80}")
    print(f"回测完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"{'='*80}")

    return results


def print_summary(results):
    """打印统计摘要"""
    if not results:
        print("没有可用的结果")
        return

    print(f"\n{'='*80}")
    print("统计摘要".center(80))
    print(f"{'='*80}")

    # 整体统计
    avg_accuracy = sum(r['accuracy']['overall_accuracy'] for r in results) / len(results)
    avg_return = sum(r['metrics']['annual_return'] for r in results) / len(results)
    avg_win_rate = sum(r['metrics']['win_rate'] for r in results) / len(results)
    avg_trades = sum(r['metrics']['trade_count'] for r in results) / len(results)

    print(f"股票数量: {len(results)}")
    print(f"平均准确率: {avg_accuracy*100:.1f}%")
    print(f"平均年化收益: {avg_return*100:.1f}%")
    print(f"平均胜率: {avg_win_rate*100:.1f}%")
    print(f"平均交易次数: {avg_trades:.1f}")

    # 按市值分组统计
    print(f"\n{'='*80}")
    print("按市值分组统计".center(80))
    print(f"{'='*80}")

    cap_groups = {
        '100-200亿': [],
        '200-300亿': [],
        '300-400亿': [],
        '400-500亿': [],
    }

    for r in results:
        cap = r['market_cap']
        if 100 <= cap < 200:
            cap_groups['100-200亿'].append(r)
        elif 200 <= cap < 300:
            cap_groups['200-300亿'].append(r)
        elif 300 <= cap < 400:
            cap_groups['300-400亿'].append(r)
        elif 400 <= cap <= 500:
            cap_groups['400-500亿'].append(r)

    for group_name, group_results in cap_groups.items():
        if group_results:
            avg_acc = sum(r['accuracy']['overall_accuracy'] for r in group_results) / len(group_results)
            avg_ret = sum(r['metrics']['annual_return'] for r in group_results) / len(group_results)
            avg_win = sum(r['metrics']['win_rate'] for r in group_results) / len(group_results)
            print(f"{group_name:12s}: {len(group_results):3d}只 | "
                  f"准确率{avg_acc*100:5.1f}% | "
                  f"年化{avg_ret*100:5.1f}% | "
                  f"胜率{avg_win*100:4.0f}%")

    # 表现最好的10只股票
    print(f"\n{'='*80}")
    print("表现最好的10只股票 (按准确率排序)".center(80))
    print(f"{'='*80}")

    sorted_results = sorted(results, key=lambda x: x['accuracy']['overall_accuracy'], reverse=True)
    top_10 = sorted_results[:10]

    summary_data = []
    for r in top_10:
        m = r['metrics']
        a = r['accuracy']
        summary_data.append({
            '代码': r['stock_code'],
            '名称': r['stock_name'],
            '市值': f"{r['market_cap']:.0f}亿",
            '准确率': f"{a['overall_accuracy']*100:.0f}%",
            '年化': f"{m['annual_return']*100:5.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # 表现最差的10只股票
    print(f"\n{'='*80}")
    print("表现最差的10只股票 (按准确率排序)".center(80))
    print(f"{'='*80}")

    bottom_10 = sorted_results[-10:]
    summary_data = []
    for r in bottom_10:
        m = r['metrics']
        a = r['accuracy']
        summary_data.append({
            '代码': r['stock_code'],
            '名称': r['stock_name'],
            '市值': f"{r['market_cap']:.0f}亿",
            '准确率': f"{a['overall_accuracy']*100:.0f}%",
            '年化': f"{m['annual_return']*100:5.1f}%",
            '胜率': f"{m['win_rate']*100:4.0f}%",
            '交易': m['trade_count'],
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))


def main():
    print("="*80)
    print("缠论买卖点策略 - 大中市值股票池回测".center(80))
    print("="*80)

    # 策略配置
    strategy_params = {
        "buy_percent": 0.25,   # 每次买入25%仓位
        "stop_loss": -0.05,    # 止损-5%
        "take_profit": 0.20,   # 止盈+20%
    }

    bsp_filter_config = {
        "only_t1": True,       # 只做一类买卖点
        "need_confirm": 10,    # 10根K线确认
    }

    config_name = "T1+确认10K"

    # 结果文件路径
    results_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_large_cap_pool_results.json"

    # 获取股票池
    stock_list = get_stocks_by_market_cap(min_cap=100, max_cap=500)

    if stock_list.empty:
        print("\n❌ 没有获取到股票列表！")
        return

    # 排序：按市值从小到大
    stock_list = stock_list.sort_values('market_cap_num').reset_index(drop=True)

    # 批量回测
    results = run_batch_backtest(
        stock_list=stock_list,
        config_name=config_name,
        chan_config_dict=create_chan_config(),
        strategy_params=strategy_params,
        bsp_filter_config=bsp_filter_config,
        results_path=results_path
    )

    # 打印统计摘要
    print_summary(results)

    print(f"\n✅ 详细结果已保存至: {results_path}")
    print(f"\n{'='*80}")
    print("✅ 大中市值股票池回测完成！".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
