# Chan.py Web 功能概览

本文件概述 `web/` 目录内可独立交付的缠论在线分析系统，便于后续开发、联调和需求沟通。

## 总体目标
- **历史缠论分析**：输入股票代码、时间范围与周期，返回 K 线、笔、线段、中枢、买卖点及多种指标数据。
- **交互式图表展示**：前端渲染多周期蜡烛图，叠加各类缠论结果，并可对配置、指标开启/关闭。
- **配置管理**：提供默认参数、预设模板及本地化保存能力。
- **实时行情与告警**：支持外部 push 实时 K 线，触发策略运算并通过 WebSocket 下发告警。
- **扩展能力**：预留图表导出、更多策略/指标、外部数据源等扩展点。

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
- `web/backend/api/analysis.py`：`POST /api/analysis/calculate`，封装 ChanService，返回缠论各项指标及元数据。
- `web/backend/api/chart.py`：`POST /api/chart/generate`，调用 ChartService 生成 Plotly JSON；`/export` 预留图表导出。
- `web/backend/api/config.py`：提供默认配置和预设方案，驱动前端配置面板。
- `web/backend/api/alerts.py`：REST + WebSocket，负责实时 tick 进站、状态查询、历史告警与订阅推送。
- `web/backend/services/chan_service.py`：衔接仓库根目录的 Chan 计算，负责参数整理、指标生成、结果裁剪。
- `web/backend/services/chart_service.py`：组装 Plotly 图表（K 线、成交量、笔/段/中枢/买卖点）。
- `web/backend/runtime/realtime.py` + `services/chan_trigger.py`：实现 trigger_load 驱动的增量会话、状态缓存及策略调度。
- `web/backend/strategies/`：策略基类、注册中心、示例策略 `SimpleBreakoutStrategy`，用于实时价格突破告警。
- `web/backend/services/state_cache.py`、`alert_dispatcher.py`：管理实时缓存、告警历史与广播。

## 前端页面要点
- `web/frontend/index.html` 使用 React 18 + Ant Design 5（CDN 版）与本地打包的 KLineCharts。
- 页面布局含左右可拖拽的侧边栏（参数配置、指标控制、告警面板）及主图区域。
- 核心接口调用顺序：
  1. `GET /api/config/default` 与 `/api/config/presets` 初始化初始配置。
  2. `POST /api/analysis/calculate` 获取缠论分析结果。
  3. （可选）`POST /api/chart/generate` 获取 Plotly JSON；前端也可自行处理结果渲染。
  4. 实时能力通过 `/api/alerts/*` REST 接口与 `ws://.../api/alerts/stream` 建立 WebSocket 订阅。

## 关键数据流
1. **历史分析**：前端表单提交参数 → `/api/analysis/calculate` → ChanService 调用根目录 Chan 模块 → 返回结构化数据 → 前端绘制。
2. **实时行情**：外部实时 tick 推送到 `/api/alerts/feed` → ChanTriggerSession 增量计算 → StrategyRegistry 评估策略 → AlertDispatcher 推送至 WebSocket → 前端告警面板展示。
3. **配置与策略**：`/api/config` 提供默认值和模板，`/api/alerts/strategies` 返回策略列表以供前端选择或展示说明。

## 扩展建议
- 图表导出：补全 `/api/chart/export`，支持 PNG/SVG/PDF/HTML 下载。
- 多策略支持：扩展 `strategies/` 目录，利用 `StrategyRegistry` 动态注册。
- 数据源扩展：在 ChanService 中添加更多 `DATA_SRC_MAP` 映射与前端选项。
- 状态持久化：将 `StateCache` 替换为 Redis/数据库，以跨进程维持状态和历史。
- 用户配置持久化：结合 FastAPI 路由与数据库保存用户偏好，或使用浏览器 `localStorage`。

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
