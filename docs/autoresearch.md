# AutoResearch

`AutoResearch/` 是当前仓库里统一的实验层，用来把训练、选股、sweep 和结果落盘组织成可复现流程。

它不负责重新实现底层模型或特征，而是复用现有模块：

- `ML/Training/Trainer.py`
- `ML/Prediction/Predictor.py`
- `ML/Utils/ModelIO.py`
- `Research/SignalReport.py`

## 1. 入口

统一入口脚本：

```bash
python3.11 App/run_autoresearch_pipeline.py --spec <json-spec>
```

实验 spec 默认放在：

```text
experiments/autoresearch/
```

## 2. 支持的模式

### 2.1 `selection`

用途：

- 加载现有模型
- 对指定股票池做横截面排序
- 输出某个 `as_of` 日期下的推荐名单

示例：

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/hs300_daily_selection.json
```

### 2.2 `training`

用途：

- 加载训练 spec
- 组装训练股票池
- 训练模型
- 写出 `summary / manifest / model / metadata`
- 可选执行 downstream benchmark selection

示例：

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training.json
```

发布到全局模型目录：

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/baseline_model_training.json \
  --publish-model
```

### 2.3 `training_sweep`

用途：

- 用 grid 展开多个训练实验
- 排名各个变体
- 写 sweep summary 和 leaderboard

示例：

```bash
python3.11 App/run_autoresearch_pipeline.py --spec experiments/autoresearch/baseline_model_training_sweep.json
```

## 3. 数据源行为

当前分支已经接入本地 SQLite 日线库。

规则是：

1. 如果本地 SQLite 数据库存在，训练和选股优先读本地库
2. 如果本地库不存在，再回退到在线 provider

默认数据库路径：

```text
./data/market_data.sqlite3
```

也可以通过环境变量覆盖：

```bash
export CHAN_LOCAL_DB_PATH=/your/path/market_data.sqlite3
```

同步脚本：

```bash
python3.11 App/sync_a_share_daily_to_sqlite.py --universe hs300 --begin 2018-01-01
```

## 4. 训练自动扩容

训练时如果样本数量不足，当前实现不会直接失败，而是按固定梯度自动重试。

当前 phase-1 扩容顺序：

1. 原始训练 spec
2. `begin_time - 2 years`
3. `begin_time - 4 years`
4. `hs300 + 原始 begin_time`
5. `hs300 + begin_time - 2 years`

训练 summary 会记录：

- `training_attempts`
- `original_training_spec`
- `effective_training_spec`
- `auto_expansion`
- `dataset_profile`
- `classification_metrics`
- `overfit_risk`

## 5. 常用 spec

当前仓库里直接可用的几个 spec：

- `experiments/autoresearch/hs300_daily_selection.json`
- `experiments/autoresearch/baseline_model_training.json`
- `experiments/autoresearch/baseline_model_training_sweep.json`

生成的新 sweep 会写到：

```text
experiments/autoresearch/generated/
```

## 6. 常用命令

跑默认目录下所有 spec：

```bash
python3.11 App/run_autoresearch_pipeline.py
```

只跑一个 spec：

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/hs300_daily_selection.json
```

生成下一轮 sweep：

```bash
python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep
```

生成并立刻执行：

```bash
python3.11 App/run_autoresearch_pipeline.py --generate-next-sweep --execute-generated-sweep
```

改结果目录：

```bash
python3.11 App/run_autoresearch_pipeline.py \
  --spec experiments/autoresearch/hs300_daily_selection.json \
  --results-root ./tmp/autoresearch
```

## 7. 输出结构

单次实验结果默认写到：

```text
AutoResearch/results/experiments/<experiment>/runs/<run_id>/
```

常见文件：

- `spec.json`
- `summary.json`
- `manifest.json`
- `recommendations.csv`
- `recommendations.json`
- `models/model_<version>.pkl`
- `models/metadata_<version>.json`

sweep 结果写到：

```text
AutoResearch/results/sweeps/<sweep>/runs/<run_id>/
```

## 8. 建议

如果你希望训练和选股稳定可复现，建议固定成这个流程：

1. 先同步本地 SQLite 行情库
2. 再跑训练 spec
3. 最后跑选股 spec

这样不会把研究链路绑死在在线 provider 上。
