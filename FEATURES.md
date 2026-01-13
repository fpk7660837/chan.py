# 回测与监控系统功能特性

## 新增功能概览

本次更新为 chan.py 缠论框架新增了完整的**回测系统**和**监控通知系统**，使其从单纯的技术分析工具升级为完整的量化交易解决方案。

---

## 一、回测系统 (Backtest/)

### 核心特性

✅ **完整的回测引擎**
- 基于 `trigger_step` 机制的逐步回测
- 支持单只股票和多只股票组合回测
- 时间对齐和多级别支持

✅ **策略开发框架**
- 抽象基类 `CStrategy`，方便自定义策略
- 内置示例策略：买卖点策略、均线策略
- 策略生命周期回调：`on_bar()`, `on_trade()`, `on_backtest_end()`

✅ **完善的成本计算**
- 手续费：万三（最低5元）
- 滑点：可配置比例
- 印花税：千一（仅卖出）

✅ **持仓管理**
- T+1 交易制度
- 仓位控制（单只/总仓位）
- 成本价、市值、浮动盈亏实时计算

✅ **绩效分析**
- 收益指标：累计收益率、年化收益率
- 风险指标：最大回撤、夏普比率、波动率
- 交易统计：胜率、盈亏比、平均持仓天数
- 可视化：权益曲线、回撤曲线

### 快速开始

```python
from Backtest import CBacktestEngine, CBacktestConfig, CBSPStrategy
from Common.CEnum import DATA_SRC, KL_TYPE

# 配置
config = CBacktestConfig(
    initial_capital=100000,
    begin_time="2023-01-01",
    end_time="2024-12-31",
)

# 策略
strategy = CBSPStrategy(buy_percent=0.2)

# 回测
engine = CBacktestEngine(config)
result = engine.run(strategy, ["000001", "600519"])

# 查看结果
print(f"收益率: {result.metrics['total_return']*100:.2f}%")
print(f"最大回撤: {result.metrics['max_drawdown']*100:.2f}%")
```

---

## 二、监控系统 (Monitor/)

### 核心特性

✅ **股票池扫描**
- 定期扫描配置的股票池
- CChan 对象缓存优化
- 交易时间检测

✅ **事件检测**
- **买卖点检测** (`CBSPDetector`)：实时检测缠论买卖点
- **价格突破检测** (`CPriceBreakDetector`)：突破新高/新低
- **持仓监控** (`CPositionMonitorDetector`)：止损止盈提醒
- 支持自定义检测器

✅ **后台运行**
- 独立线程运行，不阻塞主程序
- 异常自动恢复
- 事件去重和过滤

### 快速开始

```python
from Monitor import CMonitorEngine, CMonitorConfig, CBSPDetector
from Notification import CNotificationService

# 配置
config = CMonitorConfig(
    scan_interval=60,
    stock_pool=["000001", "600519"],
    notification_config={
        "dingtalk": {"webhook_url": "YOUR_WEBHOOK"}
    }
)

# 启动监控
engine = CMonitorEngine(config)
engine.add_detector(CBSPDetector())
engine.set_notification_service(CNotificationService(config.notification_config))
engine.start()
```

---

## 三、通知系统 (Notification/)

### 核心特性

✅ **多渠道支持**
- 钉钉机器人（支持加签）
- 企业微信机器人
- 飞书机器人

✅ **消息格式化**
- Markdown 格式
- 纯文本格式
- HTML 格式
- 级别标识（🔴高/🟡中/🟢低）

✅ **统一接口**
- 一次配置，多渠道发送
- 测试功能确保配置正确

### 快速开始

```python
from Notification import CNotificationService

config = {
    "dingtalk": {
        "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=xxx",
        "secret": "SECxxx"
    }
}

service = CNotificationService(config)
service.test_all()  # 测试所有通知渠道
```

---

## 四、存储系统 (Storage/)

### 核心特性

✅ **回测结果持久化**
- 回测运行记录
- 交易明细
- 权益曲线
- SQLite 数据库存储

✅ **持仓状态管理**
- 持仓信息存储
- 批量更新价格
- 持仓查询

### 快速开始

```python
from Storage import CSQLiteDatabase, CBacktestStorage

db = CSQLiteDatabase("backtest.db")
storage = CBacktestStorage(db)

# 保存回测结果
run_id = storage.save_backtest(result)

# 查看历史回测
history = storage.list_backtests(limit=10)
```

---

## 五、完整使用流程

### 流程1：策略回测

```
1. 定义策略 (继承 CStrategy)
   ↓
2. 配置回测参数 (CBacktestConfig)
   ↓
3. 运行回测 (CBacktestEngine.run())
   ↓
4. 查看绩效 (result.metrics)
   ↓
5. 保存结果 (CBacktestStorage.save_backtest())
   ↓
6. 绘制图表 (performance.plot_equity_curve())
```

### 流程2：实时监控

```
1. 配置股票池和通知渠道 (CMonitorConfig)
   ↓
2. 创建监控引擎 (CMonitorEngine)
   ↓
3. 添加检测器 (add_detector)
   ↓
4. 启动监控 (start)
   ↓
5. 接收通知 (钉钉/微信/飞书)
```

---

## 六、代码结构

```
chan.py/
├── Backtest/              # 回测模块
│   ├── BacktestEngine.py  # 回测引擎
│   ├── Strategy.py        # 策略基类
│   ├── Position.py        # 持仓管理
│   ├── Performance.py     # 绩效分析
│   └── ...
├── Monitor/               # 监控模块
│   ├── MonitorEngine.py   # 监控引擎
│   ├── EventDetector.py   # 事件检测
│   ├── Scanner.py         # 扫描器
│   └── ...
├── Notification/          # 通知模块
│   ├── DingTalkNotifier.py
│   ├── WeChatNotifier.py
│   ├── FeishuNotifier.py
│   └── ...
├── Storage/               # 存储模块
│   ├── Database.py
│   ├── BacktestStorage.py
│   └── PositionStorage.py
└── Examples/              # 示例代码
    ├── backtest_example.py
    ├── monitor_example.py
    ├── README.md
    └── strategies/
        ├── bsp_strategy.py
        └── macd_strategy.py
```

---

## 七、技术亮点

### 1. 完美集成现有框架
- 充分利用 CChan 的 `trigger_step` 和 `step_load()` 机制
- 复用 `get_latest_bsp()` 买卖点识别
- 兼容所有数据源（AkShare、BaoStock等）

### 2. 模块化设计
- 回测、监控、通知三模块独立
- 策略基类支持自定义扩展
- 检测器基类灵活可扩展

### 3. 实用性强
- 交易成本完整计算
- T+1 交易制度
- 止损止盈监控
- 多级别支持

---

## 八、配置示例

### 回测配置

```python
config = CBacktestConfig(
    initial_capital=100000.0,      # 初始10万
    commission_rate=0.0003,        # 万三手续费
    slippage_rate=0.001,           # 0.1%滑点
    stamp_tax_rate=0.001,          # 千一印花税
    max_position_per_stock=0.3,    # 单只最大30%
    max_total_position=0.95,       # 总仓位95%
    begin_time="2023-01-01",
    end_time="2024-12-31",
)
```

### 监控配置

```python
config = CMonitorConfig(
    scan_interval=60,                          # 每60秒扫描
    stock_pool=["000001", "600519", "000858"], # 股票池
    notification_config={
        "dingtalk": {
            "webhook_url": "YOUR_WEBHOOK",
            "secret": "YOUR_SECRET"
        }
    },
    work_hours=((9, 30), (15, 0))             # 交易时间
)
```

---

## 九、常见问题

**Q: 如何自定义策略？**

继承 `CStrategy` 并实现 `on_bar` 方法：

```python
class MyStrategy(CStrategy):
    def on_bar(self, chan_dict, positions, timestamp):
        signals = []
        # 你的策略逻辑
        return signals
```

**Q: 如何添加新的检测器？**

继承 `CEventDetector` 并实现 `detect` 方法：

```python
class MyDetector(CEventDetector):
    def detect(self, chan_dict):
        events = []
        # 你的检测逻辑
        return events
```

**Q: 支持哪些数据源？**

支持所有 chan.py 原有数据源：
- AkShare（推荐）
- BaoStock
- CCXT（加密货币）
- CSV 文件

---

## 十、性能优化建议

1. **回测优化**
   - 使用日线级别回测速度最快
   - 减少不必要的指标计算
   - 合理设置股票池大小

2. **监控优化**
   - 扫描间隔建议 ≥60秒
   - 使用缓存减少重复加载
   - 只监控重点股票

3. **存储优化**
   - 定期清理历史数据
   - 使用索引加速查询
   - 批量插入提高性能

---

## 十一、未来规划

### 短期（P1）
- [ ] 更多策略示例
- [ ] 实时数据增量更新
- [ ] 更多技术指标检测器
- [ ] Web 可视化界面

### 中期（P2）
- [ ] 策略参数优化（GridSearch）
- [ ] 多进程加速
- [ ] 更多绩效指标
- [ ] 移动端推送

### 长期（P3）
- [ ] 机器学习集成
- [ ] 实盘交易对接
- [ ] 云端部署
- [ ] 社区策略市场

---

## 十二、贡献指南

欢迎贡献代码和建议！

- 提交 Issue：报告 bug 或提出功能建议
- Pull Request：贡献代码或文档
- 讨论组：Telegram 群组讨论

---

## 十三、许可证

本项目遵循原 chan.py 项目的许可证。

---

## 十四、致谢

感谢 chan.py 原作者提供的优秀缠论计算框架！

本回测与监控系统在原框架基础上扩展，充分利用了其强大的缠论计算能力。
