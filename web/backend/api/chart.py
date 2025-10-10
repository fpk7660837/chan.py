"""
Chart API endpoints
Generate interactive charts using Plotly
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from services.chart_service import ChartService

router = APIRouter()


class ChartRequest(BaseModel):
    """Request model for chart generation"""
    code: str
    kline_data: List[dict]
    bi_list: Optional[List[dict]] = None
    seg_list: Optional[List[dict]] = None
    zs_list: Optional[List[dict]] = None
    bsp_list: Optional[List[dict]] = None
    
    # Chart config
    plot_bi: bool = True
    plot_seg: bool = True
    plot_zs: bool = True
    plot_bsp: bool = True
    plot_macd: bool = False
    width: int = 1200
    height: int = 800


@router.post("/generate")
async def generate_chart(request: ChartRequest):
    """
    Generate interactive Plotly chart
    
    Returns:
        Plotly JSON configuration for frontend rendering
    """
    try:
        service = ChartService()
        chart_json = service.generate_plotly_chart(request.dict())
        return {"chart": chart_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@router.post("/export")
async def export_chart(request: ChartRequest, format: str = "png"):
    """
    Export chart as image file
    
    Args:
        format: Output format (png/svg/pdf/html)
        
    Returns:
        File download response
    """
    # TODO: Implement chart export
    return {"message": "Export functionality coming soon"}

