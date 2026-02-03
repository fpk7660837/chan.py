"""
缠论买卖点策略回测 - 最终版本

直接使用缠论的买卖点进行交易，测试不同配置的效果
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from datetime import datetime
import json


def backtest_chan_strategy(config_name, chan_config_dict, strategy_params):
    """
    回测单个配置

    Args:
        config_name: 配置名称
        chan_config_dict: 缠论配置字典
        strategy_params: 策略参数（止损、止盈等）
    """

    print(f"\n{'='*80}")
    print(f"测试配置: {config_name}")
    print(f"{'='*80}")

    code = "sz.000001"
    begin_time = "2018-01-01"
    end_time = "2023-12-31"
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig(chan_config_dict)

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    # 策略状态
    initial_capital = 100000.0
    cash = initial_capital
    position = 0  # 持仓数量
    cost_price = 0  # 成本价

    trades = []
    equity_curve = []

    buy_percent = strategy_params.get('buy_percent', 0.3)
    stop_loss = strategy_params.get('stop_loss', -0.08)
    take_profit = strategy_params.get('take_profit', 0.25)
    target_bsp_types = strategy_params.get('bsp_types', [BSP_TYPE.T1, BSP_TYPE.T1P])

    for chan_snapshot in chan.step_load():
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
                sell_value = position * current_price * 0.999  # 扣除手续费和滑点
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
            buy_volume = int(buy_amount / current_price / 100) * 100  # 取整到100股

            if buy_volume > 0 and cash >= buy_volume * current_price:
                cost = buy_volume * current_price * 1.001  # 加上手续费和滑点
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

    # 最终清仓
    if position > 0:
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

    # 计算年化收益率（简化）
    years = 5.0  # 2018-2023
    annual_return = (1 + total_return) ** (1 / years) - 1

    # 计算胜率
    win_trades = [t for t in trades if t['type'] == 'sell' and t['profit'] > 0]
    loss_trades = [t for t in trades if t['type'] == 'sell' and t['profit'] < 0]
    win_rate = len(win_trades) / len([t for t in trades if t['type'] == 'sell']) if len([t for t in trades if t['type'] == 'sell']) > 0 else 0

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

    print(f"\n【回测结果】")
    print(f"  交易次数: {len([t for t in trades if t['type'] == 'buy'])}")
    print(f"  最终资产: {final_value:,.2f}")
    print(f"  累计收益率: {total_return*100:.2f}%")
    print(f"  年化收益率: {annual_return*100:.2f}%")
    print(f"  最大回撤: {max_drawdown*100:.2f}%")
    print(f"  胜率: {win_rate*100:.2f}%")
    print(f"  盈亏比: {profit_loss_ratio:.2f}")

    if len(trades) > 0:
        print(f"\n【最近5笔交易】")
        for trade in trades[-5:]:
            print(f"  {trade['time']}: {trade['type']} {trade['volume']}股 @ {trade['price']:.2f} - {trade['reason']}")

    return {
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
            'trade_count': len([t for t in trades if t['type'] == 'buy']),
            'win_count': len(win_trades),
            'loss_count': len(loss_trades),
        },
        'trades': trades,
        'success': True
    }


def main():
    """主函数"""

    print("="*80)
    print("缠论买卖点策略回测系统 - 最终版".center(80))
    print("="*80)

    # 测试配置
    test_configs = [
        {
            "name": "配置1: 标准参数-一类买卖点",
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
            "name": "配置2: 宽松参数-一二类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 1.1,
                "min_zs_cnt": 1,
                "bs_type": "1,1p,2,2s",
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.10,
                "take_profit": 0.30,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S],
            }
        },
        {
            "name": "配置3: 严格参数-仅一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.7,
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
        {
            "name": "配置4: 高背驰-一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.6,
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
            "name": "配置5: 平衡参数-一类买卖点",
            "chan_config": {
                "trigger_step": True,
                "divergence_rate": 0.8,
                "min_zs_cnt": 1,
                "bs_type": "1,1p",
            },
            "strategy_params": {
                "buy_percent": 0.25,
                "stop_loss": -0.07,
                "take_profit": 0.22,
                "bsp_types": [BSP_TYPE.T1, BSP_TYPE.T1P],
            }
        },
    ]

    all_results = []
    for config in test_configs:
        try:
            result = backtest_chan_strategy(
                config_name=config['name'],
                chan_config_dict=config['chan_config'],
                strategy_params=config['strategy_params']
            )
            all_results.append(result)
        except Exception as e:
            print(f"配置失败: {e}")
            import traceback
            traceback.print_exc()

    # 筛选有效结果
    successful_results = [r for r in all_results if r.get('success') and r['metrics']['trade_count'] > 0]

    if not successful_results:
        print("\n❌ 所有配置都未产生交易！")
        return

    # 按年化收益率排序
    successful_results.sort(key=lambda x: x['metrics']['annual_return'], reverse=True)

    best_result = successful_results[0]

    # 生成报告
    report_content = f"""# 缠论买卖点策略回测最佳配置报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 回测说明
- **回测标的**: 平安银行 (sz.000001)
- **回测区间**: 2018-01-01 至 2023-12-31
- **时间周期**: 日线
- **初始资金**: 100,000 元
- **数据源**: BaoStock

## 最佳配置

### 配置名称
**{best_result['config_name']}**

### 缠论参数
```json
{json.dumps(best_result['chan_config'], indent=2, ensure_ascii=False)}
```

### 策略参数
- 单次买入仓位: {best_result['strategy_params']['buy_percent']*100:.0f}%
- 止损比例: {best_result['strategy_params']['stop_loss']*100:.0f}%
- 止盈比例: {best_result['strategy_params']['take_profit']*100:.0f}%
- 关注买卖点类型: {', '.join([str(t.value) for t in best_result['strategy_params']['bsp_types']])}

## 回测绩效

### 收益指标
- **最终资产**: {best_result['metrics']['final_value']:,.2f} 元
- **累计收益率**: {best_result['metrics']['total_return']*100:.2f}%
- **年化收益率**: {best_result['metrics']['annual_return']*100:.2f}%

### 风险指标
- **最大回撤**: {best_result['metrics']['max_drawdown']*100:.2f}%

### 交易统计
- **交易次数**: {best_result['metrics']['trade_count']}
- **盈利次数**: {best_result['metrics']['win_count']}
- **亏损次数**: {best_result['metrics']['loss_count']}
- **胜率**: {best_result['metrics']['win_rate']*100:.2f}%
- **盈亏比**: {best_result['metrics']['profit_loss_ratio']:.2f}

## 所有配置对比

| 排名 | 配置名称 | 年化收益率 | 累计收益率 | 最大回撤 | 胜率 | 交易次数 |
|------|---------|-----------|-----------|---------|------|---------|
"""

    for i, r in enumerate(successful_results, 1):
        m = r['metrics']
        report_content += f"| {i} | {r['config_name']} | {m['annual_return']*100:.2f}% | {m['total_return']*100:.2f}% | {m['max_drawdown']*100:.2f}% | {m['win_rate']*100:.2f}% | {m['trade_count']} |\n"

    report_content += f"""
## 使用建议

1. **参数理解**:
   - `divergence_rate`: 背驰率，值越小要求背驰越明显（更严格）
   - `min_zs_cnt`: 最小中枢数量，影响买卖点的识别
   - `bs_type`: 买卖点类型，1=一类, 1p=准一类, 2=二类, 2s=准二类

2. **实盘应用**:
   - 建议先用小资金验证策略有效性
   - 可根据市场环境调整止盈止损比例
   - 注意控制单次仓位和总仓位

3. **风险提示**:
   - 历史回测不代表未来表现
   - 实盘交易存在滑点和流动性风险
   - 建议结合基本面分析使用
"""

    # 保存报告
    report_path = "/Users/fupengkai/Documents/quant/chan.py/缠论买卖点策略最佳配置报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✅ 报告已保存至: {report_path}")

    # 保存JSON结果
    json_path = "/Users/fupengkai/Documents/quant/chan.py/backtest_results.json"
    serializable_results = []
    for r in all_results:
        r_copy = {
            'config_name': r['config_name'],
            'chan_config': r['chan_config'],
            'strategy_params': {
                k: str(v) if isinstance(v, list) else v
                for k, v in r['strategy_params'].items()
            },
            'metrics': r['metrics'],
            'trade_sample': r['trades'][-10:] if len(r['trades']) > 0 else []
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
