# Chan.py Web 功能概览

本文件概述 `web/` 目录内可独立交付的缠论在线分析系统，并标注当前仓库已经完成的能力，便于后续开发、联调和需求沟通。

## 总体目标
- ✅ **历史缠论分析**：`/api/analysis/calculate` 已实现完整入参解析与缠论计算（`web/backend/api/analysis.py:72`）。
- ✅ **交互式图表展示**：`ChartService.generate_plotly_chart` 已生成 Plotly 图配置（`web/backend/services/chart_service.py:16`），前端 `index.html` 嵌入渲染逻辑。
- ✅ **配置管理**：默认配置与预设接口 `/api/config/default`、`/api/config/presets` 已提供（`web/backend/api/config.py:25`）。
- ⚠️ **实时行情与告警**：REST 与 WebSocket 接口 (`web/backend/api/alerts.py:14`) 及后端策略引擎已就绪，但需要外部实时数据源喂入；前端尚未完成对 WebSocket 的 UI 绑定。
- 🛠️ **扩展能力**：`/api/chart/export` 仅保留 TODO，占位待实现；可扩展策略/数据源已在代码中预留接口。

## 目录结构
```
web/
├── backend/           # FastAPI 后端
│   ├── main.py        # 入口，注册路由与实时引擎
│   ├── api/           # REST/WebSocket 接口层
│   └── services/      # 业务逻辑、状态、策略等
├── frontend/          # React 单页应用（Babel 即时编译）
├── requirements.txt   # 后端依赖
├── start_uv.sh        # 便捷启动脚本
└── WEB_ARCHITECTURE.md
```

## 后端模块职责
- `web/backend/main.py`：初始化 FastAPI，挂载静态资源，注册 API 路由并调用 `setup_realtime` 创建实时管道。
- `web/backend/api/analysis.py`：`POST /api/analysis/calculate`（已实现并调用 ChanService）。
- `web/backend/api/chart.py`：`POST /api/chart/generate`（已实现）；`/export` 当前返回占位消息，待开发。
- `web/backend/api/config.py`：默认配置及预设已实现。
- `web/backend/api/alerts.py`：REST + WebSocket 已就绪，仅缺与前端/数据源的集成。
- `web/backend/services/chan_service.py`：衔接仓库根目录的 Chan 计算，已支持 MACD/MA/BOLL/KDJ/RSI 生成与截断。
- `web/backend/services/chart_service.py`：Plotly 图表生成逻辑已完备。
- `web/backend/runtime/realtime.py` + `services/chan_trigger.py`：实现 trigger_load 驱动的增量会话、状态缓存及策略调度。
- `web/backend/strategies/`：策略基类、注册中心、示例策略 `SimpleBreakoutStrategy`，用于实时价格突破告警。
- `web/backend/services/state_cache.py`、`alert_dispatcher.py`：管理实时缓存、告警历史与广播。

## 前端页面要点
- `web/frontend/index.html` 使用 React 18 + Ant Design 5（CDN 版）与本地打包的 KLineCharts。
- 页面布局含左右可拖拽的侧边栏（参数配置、指标控制、告警面板）及主图区域；当前页面已实现表单/界面框架，但部分事件处理仍需补充。
- 左侧自选面板已支持：按代码或名称搜索添加、快速添加常用指数、自选列表高亮当前标的并一键切换。
- 主图默认展示原始 K 线，同时以虚线矩形并附带轻量底色圈出由多根原始 K 线合并而成的缠论 K 线，可在指标面板切换显示。
- 核心接口调用顺序：
  1. `GET /api/config/default` 与 `/api/config/presets` 初始化初始配置。
  2. `POST /api/analysis/calculate` 获取缠论分析结果。
  3. （可选）`POST /api/chart/generate` 获取 Plotly JSON；当前前端仍需补充调用与渲染逻辑。
  4. 实时能力通过 `/api/alerts/*` REST 接口与 `ws://.../api/alerts/stream` 建立 WebSocket 订阅（前端尚未接入）。

## 关键数据流
1. **历史分析**：前端表单提交参数 → `/api/analysis/calculate` → ChanService 调用根目录 Chan 模块 → 返回结构化数据 → 前端绘制。
2. **实时行情**：外部实时 tick 推送到 `/api/alerts/feed` → ChanTriggerSession 增量计算 → StrategyRegistry 评估策略 → AlertDispatcher 推送至 WebSocket → 前端告警面板展示。
3. **配置与策略**：`/api/config` 提供默认值和模板（已实现），`/api/alerts/strategies` 返回策略列表（已实现，但前端未消费）。

## 扩展建议
- 图表导出：补全 `/api/chart/export`，支持 PNG/SVG/PDF/HTML 下载（当前仅返回占位文案）。
- 多策略支持：扩展 `strategies/` 目录，利用 `StrategyRegistry` 动态注册（框架已具备）。
- 数据源扩展：在 ChanService 中添加更多 `DATA_SRC_MAP` 映射与前端选项（目前仅 BAO_STOCK/CSV）。
- 状态持久化：将 `StateCache` 替换为 Redis/数据库，以跨进程维持状态和历史。
- 用户配置持久化：结合 FastAPI 路由与数据库保存用户偏好，或使用浏览器 `localStorage`（前端需实现）。

## 运行与依赖
```bash
cd web
./start_uv.sh              # 使用 uv 创建虚拟环境并安装依赖
# 或手动：
uv venv --python python3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r ../Script/requirements.txt
cd backend && python main.py
```

访问：
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

至此，仅修改 `web/` 目录即可实现浏览器端缠论分析与实时告警平台。
