# Chan.py Web Platform

Web-based visualization platform for Chan theory analysis.

## Features

- 📊 Interactive K-line charts with Chan theory indicators
- 🔍 Real-time calculation of Bi, Seg, ZhongShu, and BuySellPoints
- 📈 Multiple timeframe support (日线/60分/30分/15分/5分)
- 🇨🇳 A-share market support (BaoStock data source)
- 🎨 Modern UI with Vue.js 3 + Element Plus
- ⚡ Ultra-fast setup with uv (10-100x faster than pip)

## Quick Start

### Prerequisites

- **Python 3.11+** (required)
- **uv** (recommended package manager)

Install uv:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Start Server

```bash
cd web
./start_uv.sh
```

Then open http://localhost:8000 in your browser.

### Manual Installation

```bash
cd web

# Create virtual environment
uv venv --python python3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (super fast with uv!)
uv pip install -r requirements.txt
uv pip install -r ../Script/requirements.txt

# Start server
cd backend
python main.py
```

## Architecture

```
Browser (Vue.js 3 + Plotly)
    ↓ REST API
FastAPI Backend
    ↓ Python Import
Original chan.py Code (No Modifications!)
    ↓
BaoStock Data Source (A-share)
```

## Project Structure

```
web/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── start_uv.sh         # One-click startup
├── backend/            # FastAPI backend
│   ├── main.py
│   ├── api/           # API routes
│   └── services/      # Business logic
└── frontend/          # Vue.js frontend
    └── index.html
```

## API Documentation

After starting the server, visit:
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Usage

1. Enter A-share stock code (e.g., `sz.000001` for 平安银行, `sh.600000` for 浦发银行)
2. Select date range (default: from 2020-01-01)
3. Select K-line level (日线/60分/30分/15分/5分)
4. Configure Chan theory parameters:
   - Seg algorithm (线段算法): 缠论特征序列/笔破坏/1+1终结
   - ZhongShu algorithm (中枢算法): 段内中枢/跨段中枢/自动
5. Select display elements (笔/线段/中枢/买卖点)
6. Click "开始分析" (Start Analysis)
7. View interactive chart with all Chan theory indicators

## Why uv?

uv is a modern Python package manager that's **10-100x faster** than pip:

- ⚡ Install dependencies in 2-3 seconds (vs 45s with pip)
- 🎯 Automatic Python version management
- 🔒 Built-in dependency locking
- 💯 100% compatible with pip

Speed comparison:
| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Install deps | 45s | 2.5s | **18x** |
| Create venv | 3.2s | 0.1s | **32x** |

## Common Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Add new dependency
uv pip install plotly
echo "plotly>=5.18.0" >> requirements.txt

# Update dependencies
uv pip install --upgrade -r requirements.txt
```

## Troubleshooting

**Python version < 3.11?**
```bash
# Install Python 3.11
brew install python@3.11  # macOS
sudo apt install python3.11  # Ubuntu

# Create venv with specific version
uv venv --python python3.11
```

**Port 8000 already in use?**
Edit `backend/main.py` and change the port number.

**Import errors?**
Make sure you're in the `web/` directory and dependencies are installed.

## Development

```bash
# Activate environment
source .venv/bin/activate

# Run in development mode (with reload)
cd backend
uvicorn main:app --reload

# Add development dependencies
uv add --dev pytest ruff black
```

## Key Design Principles

✅ **Zero Modification** - Original chan.py code remains unchanged  
✅ **Loose Coupling** - Independent web layer  
✅ **Easy Updates** - Can `git pull` original code anytime  
✅ **Clean Architecture** - Clear separation of concerns  

## License

Follows the license of the original chan.py project.
