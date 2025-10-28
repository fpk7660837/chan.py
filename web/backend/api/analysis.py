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


class MacdConfigModel(BaseModel):
    fast: int = Field(12, description="MACD fast period")
    slow: int = Field(26, description="MACD slow period")
    signal: int = Field(9, description="MACD signal period")


class DemarkConfigModel(BaseModel):
    demark_len: int = Field(9, description="DeMark length")
    setup_bias: int = Field(4, description="Setup bias")
    countdown_bias: int = Field(2, description="Countdown bias")
    max_countdown: int = Field(13, description="Max countdown")
    tiaokong_st: bool = Field(True, description="Count gap as setup")
    setup_cmp2close: bool = Field(True, description="Compare setup to close price")
    countdown_cmp2close: bool = Field(True, description="Compare countdown to close price")

    class Config:
        extra = "ignore"


class ChanConfigModel(BaseModel):
    bi_algo: Optional[str] = Field("normal", description="Bi algorithm")
    bi_strict: bool = Field(True, description="Strict bi rules")
    bi_fx_check: str = Field("strict", description="Bi fractal check mode")
    gap_as_kl: bool = Field(False, description="Treat gap as K line")
    bi_end_is_peak: bool = Field(True, description="Bi end must be peak")
    bi_allow_sub_peak: bool = Field(True, description="Allow sub peak")
    seg_algo: str = Field("chan", description="Segment algorithm")
    left_seg_method: str = Field("peak", description="Left segment method")
    zs_combine: bool = Field(True, description="Enable ZS combine")
    zs_combine_mode: str = Field("zs", description="ZS combine mode")
    one_bi_zs: bool = Field(False, description="Allow one-bi ZS")
    zs_algo: str = Field("normal", description="ZS algorithm")
    trigger_step: bool = Field(False, description="Enable trigger step")
    skip_step: int = Field(0, ge=0, description="Skip step")
    kl_data_check: bool = Field(True, description="Check K-line consistency")
    max_kl_misalgin_cnt: int = Field(2, ge=0, description="Max misalign count")
    max_kl_inconsistent_cnt: int = Field(5, ge=0, description="Max inconsistent count")
    auto_skip_illegal_sub_lv: bool = Field(False, description="Auto skip illegal sub level")
    print_warning: bool = Field(True, description="Print warnings")
    print_err_time: bool = Field(True, description="Print error timestamps")
    mean_metrics: List[int] = Field(default_factory=list, description="Mean metric windows")
    trend_metrics: List[int] = Field(default_factory=list, description="Trend metric windows")
    macd: MacdConfigModel = Field(default_factory=MacdConfigModel, description="MACD config")
    boll_n: int = Field(20, ge=1, description="BOLL window")
    cal_rsi: bool = Field(False, description="Calculate RSI")
    rsi_cycle: int = Field(14, ge=1, description="RSI period")
    cal_kdj: bool = Field(False, description="Calculate KDJ")
    kdj_cycle: int = Field(9, ge=1, description="KDJ period")
    cal_demark: bool = Field(False, description="Calculate DeMark")
    demark: DemarkConfigModel = Field(default_factory=DemarkConfigModel, description="Demark config")

    class Config:
        extra = "ignore"


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
    chan_config: ChanConfigModel = Field(default_factory=ChanConfigModel, description="Core Chan config")

    class Config:
        extra = "ignore"


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
