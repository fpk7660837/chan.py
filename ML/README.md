# 缠论框架机器学习模块

## 概述

本模块为缠论框架添加了完整的机器学习能力，实现买卖点质量预测与排序，支持 LightGBM、XGBoost、RandomForest 等多种算法可切换。

## 核心功能

- ✅ **完整的特征工程**：提取笔、线段、中枢、K线技术指标等多层次特征
- ✅ **多种模型支持**：LightGBM、XGBoost、RandomForest 可切换
- ✅ **灵活的标签策略**：基于未来收益率的标签构建
- ✅ **预测与排序**：为买卖点打分并排序，筛选高质量信号
- ✅ **完整的回测系统**：评估模型在实际交易中的表现
- ✅ **丰富的评估指标**：AUC、精确率、夏普比率、卡玛比率、胜率、盈亏比等

## 目录结构

```
ML/                              # 机器学习核心模块
├── FeatureEngine/               # 特征工程
│   ├── BSPFeatureExtractor.py  # 买卖点特征提取器（核心）
│   └── MultiLevelExtractor.py  # 多级别特征提取器
├── Models/                      # 模型管理
│   ├── BaseModel.py            # 模型抽象接口
│   ├── LGBMModel.py            # LightGBM实现
│   ├── XGBModel.py             # XGBoost实现
│   ├── RFModel.py              # RandomForest实现
│   └── ModelFactory.py         # 模型工厂（算法切换）
├── Training/                    # 训练模块
│   ├── LabelBuilder.py         # 标签构建器
│   └── Trainer.py              # 训练器
├── Prediction/                  # 预测模块
│   └── Predictor.py            # 预测器（买卖点打分）
├── Evaluation/                  # 评估模块
│   └── Metrics.py              # 评估指标
├── Backtest/                    # 回测模块
│   └── MLBacktest.py           # ML回测引擎
└── Utils/                       # 工具模块
    └── ModelIO.py              # 模型保存/加载

Config/                          # 全局配置
└── MLConfig.py                 # ML配置管理

Examples/                        # 示例代码
├── train_model_demo.py         # 训练示例
├── predict_demo.py             # 预测示例
└── backtest_demo.py            # 回测示例
```

## 快速开始

### 1. 安装依赖

```bash
pip install lightgbm scikit-learn numpy
# 可选：如果使用XGBoost
pip install xgboost
```

### 2. 训练模型

```python
from Chan import CChan
from Common.CEnum import KL_TYPE, DATA_SRC
from Config.MLConfig import MLConfig
from ML.Training.Trainer import Trainer
from ML.Utils.ModelIO import ModelIO

# 准备训练数据
train_chan_list = []
for code in ['600000', '600016', '600036']:
    chan = CChan(
        code=code,
        begin_time='2020-01-01',
        end_time='2022-12-31',
        data_src=DATA_SRC.AKSHARE,
        lv_list=[KL_TYPE.K_DAY]
    )
    train_chan_list.append(chan)

# 配置参数
config = MLConfig()

# 训练模型
trainer = Trainer(config.to_dict())
model = trainer.train(train_chan_list, model_type='lightgbm')

# 保存模型
model_io = ModelIO('./models')
model_io.save(model, version='v1.0', metadata={'config': config.to_dict()})
```

### 3. 预测买卖点质量

```python
from ML.Utils.ModelIO import ModelIO
from ML.Prediction.Predictor import Predictor
from ML.FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor

# 加载模型
model_io = ModelIO('./models')
model = model_io.load()
metadata = model_io.load_metadata()

# 创建预测器
feature_extractor = BSPFeatureExtractor(metadata['config']['feature_config'])
predictor = Predictor(model, feature_extractor)

# 准备新数据
new_chan = CChan(code='600519', lv_list=[KL_TYPE.K_DAY])

# 获取高质量买卖点
top_buy_points = predictor.rank_bsp(new_chan, top_k=5, direction='buy')

for bsp, score in top_buy_points:
    print(f"时间: {bsp.klu.time}, 分数: {score:.3f}")
```

### 4. 运行回测

```python
from ML.Backtest.MLBacktest import MLBacktest

# 准备测试数据
test_chan_list = [...]

# 创建回测引擎
backtest_config = {
    'score_threshold': 0.7,
    'holding_period': 20,
    'initial_capital': 100000,
}
backtest = MLBacktest(predictor, backtest_config)

# 运行回测
metrics = backtest.run(test_chan_list)

print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
print(f"胜率: {metrics['win_rate']:.3f}")
print(f"盈亏比: {metrics['profit_loss_ratio']:.3f}")
```

## 特征说明

### 买卖点基础特征
- `bsp_type`: 买卖点类型编码（1类/1p/2/2s/3a/3b）
- `bsp_direction`: 方向（买点=1，卖点=-1）
- `is_seg_bsp`: 是否为线段买卖点

### 笔级别特征
- `bi_amp`: 振幅
- `bi_macd_area`: MACD面积
- `bi_macd_peak`: MACD峰值
- `bi_macd_slope`: MACD斜率
- `bi_volume`: 成交量统计
- `bi_klu_cnt`: K线数量
- `bi_dir`: 方向
- `bi_rsi`: RSI值

### 线段特征
- `seg_amp`: 线段振幅
- `seg_dir`: 线段方向
- `seg_bi_cnt`: 包含笔的数量
- `seg_slope`: 线段斜率

### 中枢特征
- `zs_high`: 中枢上沿
- `zs_low`: 中枢下沿
- `zs_amp`: 中枢振幅
- `zs_cnt`: 中枢数量
- `price_to_zs`: 价格与中枢的相对位置

### K线技术指标
- `macd_dif`, `macd_dea`, `macd_macd`: MACD指标
- `kdj_k`, `kdj_d`, `kdj_j`: KDJ指标
- `rsi`: RSI指标
- `volume`: 成交量

### 市场环境特征
- `volatility`: 波动率
- `trend_strength`: 趋势强度

## 配置说明

### 标签构建配置
```python
label_config = {
    'label_strategy': 'future_return',  # 标签策略
    'lookforward_bars': 20,             # 未来N根K线
    'threshold_pct': 0.05,              # 收益率阈值（5%）
    'use_highest_for_buy': True,        # 买点使用最高价
    'use_lowest_for_sell': True,        # 卖点使用最低价
}
```

### 模型配置
```python
model_config = {
    'model_type': 'lightgbm',  # 可切换为 xgboost, randomforest
    'lightgbm_params': {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'n_estimators': 100,
    }
}
```

### 回测配置
```python
backtest_config = {
    'score_threshold': 0.7,      # 交易信号阈值
    'holding_period': 20,        # 持仓周期
    'initial_capital': 100000,   # 初始资金
    'commission_rate': 0.0003,   # 手续费率
    'slippage': 0.001,           # 滑点
}
```

## 评估指标

### 分类指标
- **AUC**: ROC曲线下面积
- **精确率**: 预测为正的样本中真正为正的比例
- **召回率**: 真正为正的样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

### 交易指标
- **夏普比率**: 风险调整后收益
- **卡玛比率**: 收益与最大回撤的比值
- **胜率**: 盈利交易占总交易的比例
- **盈亏比**: 平均盈利与平均亏损的比值
- **最大回撤**: 最大回撤幅度

## 示例代码

完整的示例代码位于 `Examples/` 目录：

1. **train_model_demo.py**: 完整的训练流程示例
2. **predict_demo.py**: 预测和排序买卖点示例
3. **backtest_demo.py**: 回测和参数优化示例

运行示例：
```bash
# 训练模型
python Examples/train_model_demo.py

# 预测
python Examples/predict_demo.py

# 回测
python Examples/backtest_demo.py
```

## 使用建议

### 数据准备
- 训练数据：建议使用多只股票、足够长的时间周期（至少2-3年）
- 测试数据：使用不同时间段的数据，避免过拟合
- 数据质量：确保K线数据完整，技术指标计算正确

### 特征工程
- 可以根据需要启用/禁用特定特征类型
- 支持多级别特征（日线+周线）提升预测能力
- 注意特征缺失值的处理

### 模型选择
- **LightGBM**: 默认推荐，速度快、效果好
- **XGBoost**: 经典算法，适合特征较多的场景
- **RandomForest**: 简单稳定，适合快速验证

### 标签策略
- `lookforward_bars`: 根据交易周期调整（短线10-20，中线30-60）
- `threshold_pct`: 根据市场波动调整（牛市降低，熊市提高）

### 回测优化
- 调整 `score_threshold` 找到最佳信号阈值
- 考虑手续费和滑点，贴近实际交易
- 分析不同市场环境下的表现

## 注意事项

1. **数据泄漏**: 严格按时间划分训练集/测试集，避免使用未来数据
2. **过拟合**: 使用交叉验证，关注训练集和测试集的性能差异
3. **标签不平衡**: 牛熊市标签分布可能不均，考虑使用加权或采样
4. **回测偏差**: 模拟回测与实盘有差异，需考虑滑点、手续费等
5. **多级别同步**: 确保不同级别K线数据时间对齐

## 后续扩展

- [ ] 多级别特征完全对齐支持
- [ ] 在线学习和模型增量更新
- [ ] 深度学习模型（LSTM、Transformer）
- [ ] 强化学习交易策略
- [ ] 自动超参数优化
- [ ] 特征自动选择
- [ ] 模型集成（Stacking/Blending）

## License

本模块遵循与缠论框架相同的开源协议。
