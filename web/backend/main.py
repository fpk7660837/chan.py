"""
FastAPI backend server for chan.py web visualization platform
"""
import sys
from pathlib import Path

# Add parent directory to sys.path to import original chan.py modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add backend directory to sys.path for local imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from api import analysis, chart, config, alerts
from runtime.realtime import setup_realtime

app = FastAPI(
    title="Chan.py Web API",
    description="Web-based visualization platform for Chan theory analysis",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(chart.router, prefix="/api/chart", tags=["Chart"])
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])

# Mount static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

# Initialize real-time orchestrator
setup_realtime(app)

@app.get("/")
async def root():
    """Serve frontend index.html"""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Chan.py Web API is running. Visit /docs for API documentation."}

@app.get("/index.klinechart.html")
async def klinechart_version():
    """Serve KLineChart version"""
    kline_file = frontend_path / "index.klinechart.html"
    if kline_file.exists():
        return FileResponse(kline_file)
    return {"error": "KLineChart version not found"}

@app.get("/simple")
async def simple_chart():
    """Serve simplified chart page"""
    simple_file = frontend_path / "simple_chart.html"
    if simple_file.exists():
        return FileResponse(simple_file)
    return {"error": "Simple chart page not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "chan.py-web",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print("🚀 Starting Chan.py Web Server...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
