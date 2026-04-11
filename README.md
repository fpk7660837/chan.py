# chan.py

这个分支不是纯缠论静态计算示例，而是一套可运行的研究仓库，包含：

- 缠论结构计算与自定义数据源接入
- 买卖点信号研究
- 机器学习打分、训练与回测
- `AutoResearch` 训练/选股/训练 sweep
- 本地 SQLite 日线库，减少训练和选股对在线 provider 的依赖

## 当前建议的阅读顺序

1. [docs/usage.md](./docs/usage.md)
   这份是当前分支的主使用文档。
2. [docs/autoresearch.md](./docs/autoresearch.md)
   看实验 spec、训练、选股、sweep 和模型发布。
3. [quick_guide.md](./quick_guide.md)
   看底层 `CChan`、缠论元素提取、自定义数据源接入。
4. [CHAN_SIGNAL_RESEARCH.md](./CHAN_SIGNAL_RESEARCH.md)
   看不带 ML 的信号统计研究链路。

## 快速开始

### 1. 环境

- Python `3.11`
- 常用依赖：

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm akshare baostock
```

- 可选依赖：

```bash
pip install xgboost
```

### 2. 准备本地日线库

当前训练和选股会优先使用本地 SQLite 行情库。默认路径：

```text
./data/market_data.sqlite3
```

也可以通过环境变量覆盖：

```bash
export CHAN_LOCAL_DB_PATH=/your/path/market_data.sqlite3
```

同步 A 股前复权日线：

```bash
python3.11 App/sync_a_share_daily_to_sqlite.py --universe hs300 --begin 2018-01-01
```

或者：

```bash
python3.11 App/sync_a_share_daily_to_sqlite.py --codes-file ./hs300_codes.txt --begin 2018-01-01
```

### 3. 跑一次选股

直接脚本：

```bash
python3.11 App/generate_stock_recommendations.py --as-of 2026-04-10 --universe hs300
```

或者跑 `AutoResearch` spec：

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/hs300_daily_selection.json
```

### 4. 跑一次训练

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training.json --publish-model
```

## 主要入口

- `main.py`
  缠论基础能力示例。
- `App/generate_stock_recommendations.py`
  直接按日期出推荐名单。
- `App/run_autoresearch_pipeline.py`
  跑 `selection / training / training_sweep`。
- `App/run_chan_signal_research.py`
  跑不带 ML 的信号有效性研究。
- `App/sync_a_share_daily_to_sqlite.py`
  下载并更新本地 SQLite 日线库。

## 主要目录

- `AutoResearch/`
  实验 spec 解释、执行、存储、leaderboard。
- `ML/`
  特征、训练、预测、评估、回测。
- `Research/`
  信号研究与报告输出。
- `DataAPI/`
  行情数据源，包括 `AkShare`、`BaoStock`、`CSV` 和本地 SQLite。
- `experiments/autoresearch/`
  可直接运行的实验 spec。
- `models/`
  全局模型发布目录。
- `outputs/`
  非 AutoResearch 输出目录。

## 结果输出

- `AutoResearch/results/`
  AutoResearch 运行结果、summary、manifest、leaderboard。
- `models/`
  通过 `--publish-model` 发布后的模型。
- `outputs/`
  直接脚本输出的推荐或研究结果。

## 说明

- 如果本地 SQLite 库存在，训练和选股会优先读本地库。
- 如果本地库不存在，会回退到在线 provider；但在线 provider 更慢，也更容易受网络影响。
- `docs/superpowers/` 下是开发过程中的 spec 和计划历史，不是最终用户入口文档。
