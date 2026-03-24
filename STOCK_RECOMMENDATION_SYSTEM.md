# 股票推荐系统文档

本文档说明当前仓库内“股票推荐系统”的设计原理、模块结构、运行流程和使用方式。

## 1. 系统目标

这套系统不是用来“猜下一只暴涨股”，而是做两件事：

1. 对单个买卖点做质量评分。
2. 在股票池内做横截面排序，输出某个日期下更值得关注的股票名单。

核心思想是：

- 先用 `chan.py` 识别结构性买卖点。
- 再用机器学习模型判断这些买卖点的历史质量。
- 最后把股票池中“最近有效且分数更高”的股票排出来，形成推荐名单。

## 2. 系统原理

### 2.1 事件驱动，而不是直接预测整只股票

系统的最小样本不是“某只股票某天会不会涨”，而是：

- 某只股票
- 某个缠论买卖点
- 该买卖点当时的结构、指标、市场环境特征
- 之后固定持有期的未来收益

这比直接做整票涨跌预测更贴近当前仓库的能力，因为 `chan.py` 本身先产出的是买卖点对象。

### 2.2 可交易标签，而不是乐观标签

当前标签采用更接近真实交易的定义：

- 信号产生后，不在“当前 K 线收盘瞬间”穿越成交。
- 默认使用下一根 K 线开盘价入场。
- 持有固定窗口后，按配置的价格类型出场。
- 用这段可交易收益决定标签和训练目标。

默认配置见：

- `Config/MLConfig.py`
- `ML/Training/LabelBuilder.py`

这解决了原来“用未来窗口内最高价/最低价打标签”的乐观偏差。

### 2.3 训练集按时间排序

训练样本会按全市场事件时间排序，再做时间序列切分，而不是按股票逐只拼接后随机切分。

这样可以降低以下问题：

- 时间泄漏
- 同一阶段市场环境同时出现在训练集和测试集
- 结果看起来很好但实盘复现不了

实现位置：

- `ML/Training/Trainer.py`

### 2.4 推荐逻辑是“股票池排序”

对某个 `as_of` 日期，系统会：

1. 为股票池内每只股票计算所有买点分数。
2. 只保留在 `as_of` 之前出现的信号。
3. 只保留最近若干根 K 线内仍有效的信号。
4. 每只股票只取最近一个有效信号。
5. 按分数从高到低排序，输出前 `K` 只股票。

这一步不是回测，而是实际出名单逻辑。

实现位置：

- `ML/Prediction/Predictor.py`
- `App/generate_stock_recommendations.py`

### 2.5 回测分两层

#### 单信号回测

用途：

- 验证模型筛选出来的买点，单独看是否比原始信号更有效。

特点：

- 单个信号入场
- 固定持有期
- 可禁止同一股票重叠持仓

实现位置：

- `ML/Backtest/MLBacktest.py`

#### 横截面组合回测

用途：

- 验证“每期从股票池里选出前 N 只股票”是否能形成更好的组合收益。

特点：

- 周期调仓
- 股票池内排序
- 等权持仓
- 组合收益评估

实现位置：

- `ML/Backtest/CrossSectionBacktest.py`

## 3. 主要模块

### 3.1 配置

文件：

- `Config/MLConfig.py`

负责统一管理：

- 特征配置
- 标签配置
- 训练配置
- 单信号回测配置
- 横截面组合回测配置

### 3.2 特征工程

文件：

- `ML/FeatureEngine/BSPFeatureExtractor.py`

当前特征包括：

- 买卖点基础特征
- 笔特征
- 线段特征
- 中枢特征
- K 线指标特征
- 市场环境特征

### 3.3 标签构建

文件：

- `ML/Training/LabelBuilder.py`

默认标签策略：

- `forward_return`

含义：

- 默认下一根 K 线开盘入场
- 持有 `lookforward_bars`
- 用固定窗口后的收益率做标签

### 3.4 训练器

文件：

- `ML/Training/Trainer.py`

负责：

1. 从 `CChan` 中提取买卖点样本
2. 提取特征
3. 构建标签
4. 做时间序列切分
5. 训练模型

### 3.5 预测器

文件：

- `ML/Prediction/Predictor.py`

负责：

- 单个买卖点打分
- 批量打分
- 某只股票内的买卖点排序
- 股票池横截面排序

### 3.6 推荐名单脚本

文件：

- `App/generate_stock_recommendations.py`

作用：

- 指定日期
- 指定股票池或自动获取 A 股可交易股票
- 输出推荐名单到 `csv` 或 `json`

## 4. 数据流

完整链路如下：

1. `CChan` 读取历史行情并计算结构。
2. `bs_point_lst` 生成买卖点事件。
3. `BSPFeatureExtractor` 提取每个买卖点的特征。
4. `LabelBuilder` 生成未来收益标签。
5. `Trainer` 训练模型。
6. `Predictor` 给新买卖点打分。
7. `generate_stock_recommendations.py` 按日期汇总股票池推荐名单。

## 5. 环境要求

最低要求：

- Python 3.11

常用依赖：

```bash
pip install akshare pandas numpy scikit-learn lightgbm
```

可选依赖：

```bash
pip install xgboost
```

说明：

- 仓库本身依赖 Python 3.11。
- 如果没有安装 `scikit-learn`，训练器无法运行。
- 如果没有安装 `akshare` 和 `pandas`，自动股票池扫描无法运行。

## 6. 使用方式

### 6.1 训练模型

脚本：

- `Examples/train_model_demo.py`

命令：

```bash
python3.11 Examples/train_model_demo.py
```

默认流程：

1. 加载训练集样本
2. 训练 LightGBM 模型
3. 保存模型到 `./models`
4. 在测试集上输出基础评估结果

### 6.2 查看单只股票买卖点评分

脚本：

- `Examples/predict_demo.py`

命令：

```bash
python3.11 Examples/predict_demo.py
```

适合用途：

- 看单只股票最近出现了哪些高质量买卖点

### 6.3 运行单信号回测

脚本：

- `Examples/backtest_demo.py`

命令：

```bash
python3.11 Examples/backtest_demo.py
```

适合用途：

- 验证模型对买卖点质量过滤是否有效

### 6.4 运行股票池排序 + 组合回测示例

脚本：

- `Examples/stock_selection_demo.py`

命令：

```bash
python3.11 Examples/stock_selection_demo.py
```

适合用途：

- 看股票池内最新排序结果
- 看横截面组合回测结果

### 6.5 按日期输出推荐名单

脚本：

- `App/generate_stock_recommendations.py`

#### 最简单用法

```bash
python3.11 App/generate_stock_recommendations.py --as-of 2024-12-31
```

作用：

- 自动获取当前可交易 A 股股票池
- 加载最新模型
- 生成 `2024-12-31` 这个截面下的推荐股票名单

#### 指定股票池

```bash
python3.11 App/generate_stock_recommendations.py \
  --as-of 2024-12-31 \
  --codes 600519,000333,600036
```

#### 使用股票池文件

```bash
python3.11 App/generate_stock_recommendations.py \
  --as-of 2024-12-31 \
  --codes-file ./codes.txt
```

支持：

- `txt`
- `csv`
- `json`

#### 指定输出文件

```bash
python3.11 App/generate_stock_recommendations.py \
  --as-of 2024-12-31 \
  --output ./outputs/recommendations_2024-12-31.json
```

#### 常用参数

- `--as-of`：推荐日期，格式 `YYYY-MM-DD`
- `--model-version`：指定模型版本，不传则默认最新版本
- `--top-k`：输出前 K 只股票
- `--min-score`：最低分数阈值
- `--signal-lookback-bars`：信号最多向前保留多少根 K 线
- `--history-days`：为了构建结构和特征，回看多少天历史数据
- `--stale-days`：若最后一根 K 线离目标日期太久，则跳过该股票
- `--codes`：直接传股票代码列表
- `--codes-file`：从文件读取股票池
- `--output`：输出路径

## 7. 输出格式

推荐名单包含字段：

- `rank`
- `as_of`
- `code`
- `name`
- `score`
- `signal_time`
- `signal_type`
- `signal_price`
- `signal_idx`
- `model_version`

示例：

```json
[
  {
    "rank": 1,
    "as_of": "2024-12-31",
    "code": "600519",
    "name": "贵州茅台",
    "score": 0.912345,
    "signal_time": "2024/12/27",
    "signal_type": "1,2",
    "signal_price": 1720.5,
    "signal_idx": 456,
    "model_version": "v1.0"
  }
]
```

## 8. 推荐名单的生成规则

对于某个 `as_of` 日期：

1. 只使用 `as_of` 当天及之前可见的数据。
2. 每只股票只取最近一个有效买点。
3. 只保留最近 `signal_lookback_bars` 根 K 线内出现的信号。
4. 只输出分数高于阈值的股票。
5. 最终按分数降序输出。

这意味着推荐名单更像“当前可执行候选池”，而不是“未来收益确定性预测”。

## 9. 当前适用场景

适合：

- 研究缠论结构信号的统计有效性
- 对股票池做相对排序
- 做中短期候选股筛选
- 构建自己的半自动研究流程

不适合直接当成：

- 全自动实盘交易系统
- 完整的风控系统
- 基本面主导的长线投资系统

## 10. 当前限制

当前系统仍有这些边界：

- 主要依赖技术结构信号，基本面因子还没有系统接入
- 当前股票池过滤较基础，还没有行业中性、风格暴露控制
- 组合回测默认等权，未加入更复杂的资金分配逻辑
- 交易成本只做了基础手续费和滑点近似
- 真实实盘还需要补交易接口、风控、日志、异常恢复

## 11. 建议的使用顺序

建议按下面顺序使用：

1. 先训练模型。
2. 用单信号回测看模型是否比原始信号更稳。
3. 用横截面组合回测看排序逻辑是否有组合价值。
4. 最后再用按日期出名单脚本做研究或半自动跟踪。

## 12. 相关文件索引

- `Config/MLConfig.py`
- `ML/FeatureEngine/BSPFeatureExtractor.py`
- `ML/Training/LabelBuilder.py`
- `ML/Training/Trainer.py`
- `ML/Prediction/Predictor.py`
- `ML/Backtest/MLBacktest.py`
- `ML/Backtest/CrossSectionBacktest.py`
- `Examples/train_model_demo.py`
- `Examples/predict_demo.py`
- `Examples/backtest_demo.py`
- `Examples/stock_selection_demo.py`
- `App/generate_stock_recommendations.py`
