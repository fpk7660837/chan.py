# 使用文档

这份文档面向当前分支的实际能力，优先说明怎么把仓库跑起来，而不是解释所有内部实现。

## 1. 你应该先看什么

- [README.md](../README.md)
  仓库总入口和常用命令。
- [docs/autoresearch.md](./autoresearch.md)
  AutoResearch 的 spec、训练、选股、sweep。
- [quick_guide.md](../quick_guide.md)
  `CChan`、缠论元素、自定义数据源接入。
- [CHAN_SIGNAL_RESEARCH.md](../CHAN_SIGNAL_RESEARCH.md)
  纯信号研究，不依赖 ML 排名。

## 2. 环境要求

- Python `3.11`
- 建议先安装：

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm akshare baostock
```

- 可选：

```bash
pip install xgboost
```

## 3. 本地 SQLite 行情库

### 3.1 为什么推荐先做本地库

训练和选股如果直接依赖在线 provider，容易遇到：

- 网络超时
- provider 不稳定
- 训练重复拉历史数据，速度慢

当前分支已经支持本地 SQLite 日线库。只要数据库存在，训练和选股会优先使用本地库。

### 3.2 默认路径

默认数据库路径：

```text
./data/market_data.sqlite3
```

也可以通过环境变量覆盖：

```bash
export CHAN_LOCAL_DB_PATH=/your/path/market_data.sqlite3
```

### 3.3 下载数据

同步 HS300：

```bash
python3.11 App/sync_a_share_daily_to_sqlite.py --universe hs300 --begin 2018-01-01
```

同步指定股票：

```bash
python3.11 App/sync_a_share_daily_to_sqlite.py --codes 600519,000333 --begin 2020-01-01 --end 2026-04-11
```

同步代码文件：

```bash
python3.11 App/sync_a_share_daily_to_sqlite.py --codes-file ./hs300_codes.txt --begin 2018-01-01
```

当前同步脚本范围是：

- A 股日线
- 前复权 `QFQ`
- SQLite 表 `daily_bars`

## 4. 常用工作流

### 4.1 体验基础缠论计算

```bash
python3.11 main.py
```

如果你主要想看底层缠论对象、`CChanConfig`、自定义数据源，实现细节在 [quick_guide.md](../quick_guide.md)。

### 4.2 直接按日期出推荐名单

```bash
python3.11 App/generate_stock_recommendations.py --as-of 2026-04-10 --universe hs300
```

也可以传：

- `--codes`
- `--codes-file`
- `--top-k`
- `--min-score`

输出默认写到 `./outputs/`。

### 4.3 跑 AutoResearch 选股

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/hs300_daily_selection.json
```

输出写到：

```text
AutoResearch/results/experiments/<experiment>/runs/<run_id>/
```

### 4.4 跑 AutoResearch 训练

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training.json
```

如果你希望把最终模型发布到全局 `./models`：

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training.json --publish-model
```

### 4.5 跑训练 sweep

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training_sweep.json
```

### 4.6 生成下一轮 sweep

```bash
python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep
```

或者直接生成并执行：

```bash
python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep --execute-generated-sweep
```

### 4.7 跑纯信号研究

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

## 5. AutoResearch 的当前行为

- `selection`
  用现有模型对股票池排序，输出推荐名单。
- `training`
  训练模型，保存 metadata，并可选做 downstream benchmark selection。
- `training_sweep`
  按 grid 扩展多个训练实验。

当前训练还带了两个实用机制：

- 样本不足保护
  样本太少会拒绝训练，避免明显过拟合。
- 自动扩容重试
  会按 `原始范围 -> 更早 begin_time -> HS300` 自动放大训练范围。

详细说明见 [docs/autoresearch.md](./autoresearch.md)。

## 6. 输出目录

### 6.1 AutoResearch

```text
AutoResearch/results/
```

常见文件：

- `spec.json`
- `summary.json`
- `manifest.json`
- `recommendations.csv`
- `recommendations.json`
- `models/model_<version>.pkl`
- `models/metadata_<version>.json`

### 6.2 直接脚本输出

```text
outputs/
```

### 6.3 全局模型发布

```text
models/
```

## 7. 推荐用法

如果你主要是做自动训练和选股，建议按这个顺序：

1. 先把行情同步到本地 SQLite。
2. 用 `AutoResearch` 训练模型。
3. 再用 `AutoResearch` 或直接脚本做日度选股。
4. 纯信号统计分析单独走 `run_chan_signal_research.py`。

这样最稳定，也最容易复现。
