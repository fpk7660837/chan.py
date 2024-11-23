# 缠论回测系统

基于backtrader框架实现的缠论交易策略回测系统。

## 目录结构

```
BackTest/
├── data/               # 数据目录
│   └── data_loader.py  # 数据加载器
├── strategies/         # 策略目录
│   └── chan_strategy.py# 缠论策略实现
├── results/           # 回测结果目录
└── run_backtest.py    # 回测运行器
```

## 使用方法

1. 安装依赖：
```bash
pip install backtrader==1.9.76.123 baostock==0.8.8 pandas matplotlib numpy
```

2. 运行回测：
```bash
python run_backtest.py
```

## 参数说明

- `code`: 股票代码（如：sz.000001）
- `start_date`: 回测开始日期
- `end_date`: 回测结束日期
- `cash`: 初始资金
- `commission`: 手续费率
- `position_size`: 仓位大小(0-1)
- `buy_type`: 买入信号类型（如：3a表示三买）
- `sell_type`: 卖出信号类型（如：3b表示三卖）

## 回测结果

回测结果将保存在results目录下，包括：
- 交易记录CSV文件
- 回测性能指标（收益率、夏普比率、最大回撤等）
- K线图表（包含买卖点标记）
