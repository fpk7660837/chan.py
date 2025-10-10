#!/bin/bash

# Chan.py Web Platform Startup Script (using uv)

echo "🚀 Starting Chan.py Web Platform..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install it with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv --python python3.11 2>/dev/null || uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (super fast with uv!)
echo "📥 Installing dependencies..."
uv pip install -r requirements.txt -q
[ -f "../Script/requirements.txt" ] && uv pip install -r ../Script/requirements.txt -q

echo ""
echo "════════════════════════════════════════════════"
echo "  ✅ Server starting..."
echo "════════════════════════════════════════════════"
echo ""
echo "  🌐 Web:  http://localhost:8000"
echo "  📖 Docs: http://localhost:8000/docs"
echo ""

# Run backend server
cd backend
python main.py

