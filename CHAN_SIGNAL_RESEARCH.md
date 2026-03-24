# 缠论买点有效性验证系统

这套模块的目标不是直接推荐股票，而是先回答一个更基础的问题：

`你的缠论买点，在历史上是否具备统计价值。`

## 系统组成

1. `Research/SignalSnapshot.py`
   - 把缠论买卖点标准化成统一事件对象。

2. `Research/SignalSnapshotCollector.py`
   - 从 `CChan` 批量收集事件。

3. `Research/SignalEvaluator.py`
   - 统计未来 `5/10/20` 根 K 线的收益、MFE、MAE、止损命中和反向信号。

4. `Research/Baselines.py`
   - 提供随机入场和动量突破两个对照组。

5. `Research/ChanSignalBacktest.py`
   - 做纯事件回测，不引入 ML。

6. `Research/Visualization/SignalVisualizer.py`
   - 输出单个信号的结构图和统计说明。

7. `App/run_chan_signal_research.py`
   - 一键跑完整研究流程，生成 CSV、JSON、图片和 Markdown 报告。

## 使用方式

```bash
python3.11 App/run_chan_signal_research.py \
  --codes 600519,000333,600036 \
  --begin 2023-01-01 \
  --end 2024-12-31 \
  --level day \
  --data-src akshare \
  --direction buy \
  --holding-period 20 \
  --stop-loss 0.05 \
  --output-dir ./outputs/chan_signal_research
```

## 输出内容

- `signal_snapshots.csv`
  - 标准化信号事件表

- `signal_evaluations.csv`
  - 每个事件的前瞻收益和风险统计

- `backtest_summary.json`
  - 纯缠论、随机基准、动量基准的事件回测结果

- `cases/*.png`
  - 最优/最差案例结构图

- `research_report.md`
  - 汇总报告

## 解释原则

建议优先看这几项：

1. `chan_bsp` 是否明显优于 `baseline_random`
2. 某些 `bsp_type` 是否优于整体平均
3. `sure` 与 `unsure` 的差异是否显著
4. 哪些信号虽然收益高，但 `MAE` 很大
5. ML 未来应该优先过滤掉哪类高失败信号
