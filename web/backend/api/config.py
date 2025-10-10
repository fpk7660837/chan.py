"""
Configuration API endpoints
Manage Chan calculation and chart display configs
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()


class ConfigResponse(BaseModel):
    """Configuration response model"""
    chan_config: Dict[str, Any]
    plot_config: Dict[str, Any]
    data_sources: list


@router.get("/default", response_model=ConfigResponse)
async def get_default_config():
    """
    Get default configuration
    
    Returns:
        Default Chan config and plot config
    """
    return {
        "chan_config": {
            "bi_strict": True,
            "seg_algo": "chan",
            "zs_algo": "normal",
            "zs_combine": True,
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
            "bs_type": "1,2,3a,3b"
        },
        "plot_config": {
            "plot_kline": True,
            "plot_kline_combine": True,
            "plot_bi": True,
            "plot_seg": True,
            "plot_zs": True,
            "plot_bsp": True,
            "plot_macd": False
        },
        "data_sources": [
            {"value": "BAO_STOCK", "label": "BaoStock (A股)"},
            {"value": "CSV", "label": "本地CSV文件"}
        ]
    }


@router.get("/presets")
async def get_config_presets():
    """
    Get configuration presets
    
    Returns:
        List of preset configurations
    """
    return {
        "presets": [
            {
                "name": "Default",
                "description": "Standard Chan theory configuration",
                "config": {"bi_strict": True, "seg_algo": "chan"}
            },
            {
                "name": "Aggressive", 
                "description": "More sensitive to trading signals",
                "config": {"bi_strict": False, "divergence_rate": 0.8}
            },
            {
                "name": "Conservative",
                "description": "Fewer but more reliable signals", 
                "config": {"bi_strict": True, "min_zs_cnt": 2}
            }
        ]
    }

