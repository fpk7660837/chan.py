# 缠论回测与监控通知系统 - 使用指南

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [回测系统](#回测系统)
- [监控系统](#监控系统)
- [通知系统](#通知系统)
- [示例代码](#示例代码)

---

## 项目概述

基于 chan.py 缠论框架实现的回测与监控通知系统，提供：

**回测功能**：
- ✅ 完整的回测引擎，支持逐步回测
- ✅ 策略基类，方便自定义策略
- ✅ 持仓管理和交易成本计算
- ✅ 绩效分析（收益率、最大回撤、夏普比率等）
- ✅ 可视化（权益曲线、回撤曲线）

**监控功能**：
- ✅ 股票池扫描
- ✅ 买卖点实时检测
- ✅ 价格突破检测
- ✅ 持仓监控（止损止盈）

**通知功能**：
- ✅ 钉钉机器人
- ✅ 企业微信机器人
- ✅ 飞书机器人

---

## 快速开始

### 安装依赖

```bash
pip install requests numpy matplotlib akshare
```

### 运行回测示例

```bash
python Examples/backtest_example.py
```

### 运行监控示例

```bash
python Examples/monitor_example.py
```

---

## 回测系统

### 基本用法

```python
from Backtest.BacktestEngine import CBacktestEngine
from Backtest.BacktestConfig import CBacktestConfig
from Backtest.Strategy import CBSPStrategy
from Common.CEnum import DATA_SRC, KL_TYPE

# 1. 配置回测参数
config = CBacktestConfig(
    initial_capital=100000.0,
    commission_rate=0.0003,
    slippage_rate=0.001,
    stamp_tax_rate=0.001,
    begin_time="2023-01-01",
    end_time="2024-12-31",
    data_src=DATA_SRC.AKSHARE,
    lv_list=[KL_TYPE.K_DAY],
)

# 2. 创建策略
strategy = CBSPStrategy(buy_percent=0.2)

# 3. 运行回测
engine = CBacktestEngine(config)
result = engine.run(strategy, ["000001", "600519"])

# 4. 查看结果
print(f"总收益率: {result.metrics['total_return']*100:.2f}%")
print(f"最大回撤: {result.metrics['max_drawdown']*100:.2f}%")
```

### 自定义策略

继承 `CStrategy` 类并实现 `on_bar` 方法：

```python
from Backtest.Strategy import CStrategy, CSignal

class MyStrategy(CStrategy):
    def on_bar(self, chan_dict, positions, timestamp):
        signals = []

        for code, chan in chan_dict.items():
            # 获取买卖点
            bsp_list = chan.get_latest_bsp(number=1)

            if bsp_list and bsp_list[0].is_buy:
                # 生成买入信号
                signals.append(CSignal(
                    code=code,
                    direction="buy",
                    percent=0.3,
                    reason="买点信号"
                ))

        return signals
```

### 绩效指标说明

| 指标 | 说明 |
|------|------|
| total_return | 累计收益率 |
| annual_return | 年化收益率 |
| max_drawdown | 最大回撤 |
| sharpe_ratio | 夏普比率 |
| win_rate | 胜率 |
| profit_loss_ratio | 盈亏比 |
| trade_count | 交易次数 |
| avg_hold_days | 平均持仓天数 |

---

## 监控系统

### 基本用法

```python
from Monitor.MonitorEngine import CMonitorEngine
from Monitor.MonitorConfig import CMonitorConfig
from Monitor.EventDetector import CBSPDetector
from Notification.NotificationService import CNotificationService

# 1. 配置监控
config = CMonitorConfig(
    scan_interval=60,
    stock_pool=["000001", "600519", "000858"],
    notification_config={
        "dingtalk": {
            "webhook_url": "YOUR_WEBHOOK_URL",
            "secret": "YOUR_SECRET"
        }
    }
)

# 2. 创建监控引擎
engine = CMonitorEngine(config)

# 3. 添加检测器
detector = CBSPDetector(time_window_days=3)
engine.add_detector(detector)

# 4. 设置通知服务
notification = CNotificationService(config.notification_config)
engine.set_notification_service(notification)

# 5. 启动监控
engine.start()
```

### 事件检测器

**买卖点检测器**：
```python
from Monitor.EventDetector import CBSPDetector
from Common.CEnum import BSP_TYPE

detector = CBSPDetector(
    bsp_types=[BSP_TYPE.T1, BSP_TYPE.T2],  # 关注的买卖点类型
    time_window_days=3,  # 检测最近3天的买卖点
)
```

**价格突破检测器**：
```python
from Monitor.EventDetector import CPriceBreakDetector

detector = CPriceBreakDetector(
    break_type="both",  # "high"/"low"/"both"
    lookback_days=20,   # 20日新高/新低
)
```

**持仓监控检测器**：
```python
from Monitor.EventDetector import CPositionMonitorDetector

detector = CPositionMonitorDetector(
    position_storage=storage,
    stop_loss=-0.05,    # 止损-5%
    take_profit=0.20,   # 止盈+20%
)
```

---

## 通知系统

### 钉钉机器人

1. 创建钉钉群机器人，获取 webhook 地址
2. 如需加签，获取 secret 密钥
3. 配置：

```python
notification_config = {
    "dingtalk": {
        "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=xxx",
        "secret": "SECxxx"  # 可选
    }
}
```

### 企业微信机器人

```python
notification_config = {
    "wechat": {
        "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
    }
}
```

### 飞书机器人

```python
notification_config = {
    "feishu": {
        "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
    }
}
```

### 测试通知

```python
from Notification.NotificationService import CNotificationService

service = CNotificationService(notification_config)
service.test_all()  # 测试所有通知渠道
```

---

## 示例代码

### 示例1：简单回测

```python
from Backtest import CBacktestEngine, CBacktestConfig, CBSPStrategy
from Common.CEnum import DATA_SRC, KL_TYPE

config = CBacktestConfig(
    initial_capital=100000,
    begin_time="2023-01-01",
    end_time="2024-12-31",
    data_src=DATA_SRC.AKSHARE,
)

strategy = CBSPStrategy(buy_percent=0.2)
engine = CBacktestEngine(config)
result = engine.run(strategy, ["000001"])

# 绘制图表
from Backtest.Performance import CPerformance
performance = CPerformance(result, config)
performance.plot_equity_curve("equity.png")
```

### 示例2：实时监控

```python
from Monitor import CMonitorEngine, CMonitorConfig, CBSPDetector
from Notification import CNotificationService

config = CMonitorConfig(
    scan_interval=60,
    stock_pool=["000001", "600519"],
    notification_config={"dingtalk": {"webhook_url": "xxx"}}
)

engine = CMonitorEngine(config)
engine.add_detector(CBSPDetector())
engine.set_notification_service(CNotificationService(config.notification_config))
engine.start()
```

---

## 配置参数详解

### 回测配置 (CBacktestConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| initial_capital | float | 100000.0 | 初始资金 |
| commission_rate | float | 0.0003 | 手续费率（万三） |
| slippage_rate | float | 0.001 | 滑点率 |
| stamp_tax_rate | float | 0.001 | 印花税率（仅卖出） |
| max_position_per_stock | float | 0.3 | 单只股票最大仓位 |
| max_total_position | float | 0.95 | 总仓位上限 |
| begin_time | str | "2020-01-01" | 回测开始时间 |
| end_time | str | None | 回测结束时间 |
| match_mode | str | "next_open" | 成交方式 |

### 监控配置 (CMonitorConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| scan_interval | int | 60 | 扫描间隔（秒） |
| stock_pool | List[str] | [] | 股票池 |
| lookback_days | int | 365 | 历史数据天数 |
| work_hours | tuple | ((9,30),(15,0)) | 交易时间 |

---

## 常见问题

**Q: 如何添加更多股票到监控池？**

A: 修改 `stock_pool` 参数：
```python
config = CMonitorConfig(
    stock_pool=["000001", "600519", "000858", "000333"]
)
```

**Q: 如何修改扫描频率？**

A: 修改 `scan_interval` 参数（单位：秒）：
```python
config = CMonitorConfig(
    scan_interval=300  # 每5分钟扫描一次
)
```

**Q: 如何只监控买点或只监控卖点？**

A: 在检测器中过滤：
```python
def detect(self, chan_dict):
    events = []
    for code, chan in chan_dict.items():
        bsp_list = chan.get_latest_bsp(number=1)
        if bsp_list and bsp_list[0].is_buy:  # 只关注买点
            events.append(...)
    return events
```

---

## 注意事项

1. **数据源限制**：akshare 可能有访问频率限制，建议扫描间隔不要太短
2. **交易时间**：监控系统会自动检查是否在交易时间，非交易时间不会扫描
3. **通知频率**：同一事件不会重复通知，避免刷屏
4. **资金管理**：回测时严格控制仓位，避免过度杠杆

---

## 更新日志

### v1.0.0 (2024-01-14)

- ✅ 实现回测引擎核心功能
- ✅ 实现监控引擎和事件检测
- ✅ 支持钉钉/企业微信/飞书通知
- ✅ 提供策略示例和完整文档
