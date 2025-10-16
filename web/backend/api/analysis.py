"""
Analysis API endpoints
Calculate Chan theory indicators: Bi, Seg, ZhongShu, BuySellPoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

from services.chan_service import ChanService

router = APIRouter()


class DataSourceEnum(str, Enum):
    """Data source options (A-share only)"""
    BAO_STOCK = "BAO_STOCK"
    CSV = "CSV"


class KLineTypeEnum(str, Enum):
    """K-line type options"""
    K_1M = "1m"
    K_5M = "5m"
    K_15M = "15m"
    K_30M = "30m"
    K_60M = "60m"
    K_DAY = "day"
    K_WEEK = "week"
    K_MON = "mon"


class AnalysisRequest(BaseModel):
    """Request model for Chan analysis"""
    code: str = Field(..., description="Stock code, e.g., sz.000001, HK.00700")
    begin_time: Optional[str] = Field("2020-01-01", description="Start date, format: YYYY-MM-DD")
    end_time: Optional[str] = Field(None, description="End date, format: YYYY-MM-DD")
    data_src: DataSourceEnum = Field(DataSourceEnum.BAO_STOCK, description="Data source")
    lv_list: List[KLineTypeEnum] = Field([KLineTypeEnum.K_DAY], description="K-line levels")
    
    # Chan config parameters
    bi_strict: bool = Field(True, description="Use strict Bi rules")
    seg_algo: str = Field("chan", description="Seg algorithm: chan/break/1+1")
    zs_algo: str = Field("normal", description="ZhongShu algorithm: normal/over_seg/auto")
    plot_bi: bool = Field(True, description="Include Bi in results")
    plot_seg: bool = Field(True, description="Include Seg in results") 
    plot_zs: bool = Field(True, description="Include ZhongShu in results")
    plot_bsp: bool = Field(True, description="Include BuySellPoints in results")
    plot_macd: bool = Field(True, description="Include MACD indicator in results")
    plot_ma: bool = Field(False, description="Include MA (Moving Average) indicator in results")
    plot_boll: bool = Field(False, description="Include BOLL (Bollinger Bands) indicator in results")
    plot_kdj: bool = Field(False, description="Include KDJ indicator in results")
    plot_rsi: bool = Field(False, description="Include RSI indicator in results")
    ma_params: Optional[List[int]] = Field([5, 10, 20, 60], description="MA periods")
    kdj_period: int = Field(9, description="KDJ calculation period")
    rsi_period: int = Field(14, description="RSI calculation period")


class AnalysisResponse(BaseModel):
    """Response model for Chan analysis"""
    code: str
    data_source: str
    kline_data: List[dict]
    bi_list: Optional[List[dict]] = None
    seg_list: Optional[List[dict]] = None
    zs_list: Optional[List[dict]] = None
    bsp_list: Optional[List[dict]] = None
    macd_data: Optional[List[dict]] = None
    ma_data: Optional[dict] = None
    boll_data: Optional[List[dict]] = None
    kdj_data: Optional[List[dict]] = None
    rsi_data: Optional[List[dict]] = None
    meta: dict


@router.post("/calculate", response_model=AnalysisResponse)
async def calculate_chan_analysis(request: AnalysisRequest):
    """
    Calculate Chan theory analysis for given stock
    
    Returns:
    - K-line data
    - Bi (笔) list
    - Seg (线段) list  
    - ZhongShu (中枢) list
    - BuySellPoint (买卖点) list
    """
    try:
        print(f"📊 Analyzing: {request.code}, Period: {request.begin_time} to {request.end_time}")
        print(f"📈 Levels: {request.lv_list}, Data source: {request.data_src}")
        
        service = ChanService()
        result = service.calculate(request.dict())
        
        print(f"✅ Analysis complete: {result['meta'].get('total_klines', 0)} K-lines")
        return result
    except Exception as e:
        import traceback
        error_detail = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/search")
async def search_stock(keyword: str):
    """
    Search stock by keyword
    
    Args:
        keyword: Stock code or name keyword
        
    Returns:
        List of matching stocks
    """
    # TODO: Implement stock search functionality
    return {
        "results": [
            {"code": "sz.000001", "name": "平安银行"},
            {"code": "sh.600000", "name": "浦发银行"}
        ]
    }

