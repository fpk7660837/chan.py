"""
MonitorConfig - 监控配置类
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from Common.CEnum import DATA_SRC, KL_TYPE


@dataclass
class CMonitorConfig:
    """监控配置"""

    # ============ 扫描设置 ============
    scan_interval: int = 60                # 扫描间隔（秒）
    stock_pool: List[str] = field(default_factory=list)  # 股票池

    # ============ 数据设置 ============
    data_src: DATA_SRC = DATA_SRC.AKSHARE
    lv_list: List[KL_TYPE] = field(default_factory=lambda: [KL_TYPE.K_DAY])
    lookback_days: int = 365               # 历史数据天数

    # ============ 缠论配置 ============
    chan_config: Dict = field(default_factory=lambda: {
        "bi_strict": True,
        "print_warning": False,
    })

    # ============ 通知配置 ============
    notification_config: Dict = field(default_factory=dict)

    # ============ 工作时间 ============
    work_hours: Tuple[Tuple[int, int], Tuple[int, int]] = ((9, 30), (15, 0))  # 交易时间

    # ============ 其他设置 ============
    max_concurrent_scans: int = 5          # 最大并发扫描数
    enable_cache: bool = True              # 是否启用缓存
    cache_update_interval: int = 300       # 缓存更新间隔（秒）

    def is_trading_time(self) -> bool:
        """检查当前是否在交易时间内"""
        from datetime import datetime

        now = datetime.now()
        current_time = (now.hour, now.minute)

        # 检查是否在工作时间范围内
        start_hour, start_minute = self.work_hours[0]
        end_hour, end_minute = self.work_hours[1]

        start_minutes = start_hour * 60 + start_minute
        end_minutes = end_hour * 60 + end_minute
        current_minutes = current_time[0] * 60 + current_time[1]

        return start_minutes <= current_minutes <= end_minutes

    def is_trading_day(self) -> bool:
        """检查当前是否为交易日（简化版，只检查周末）"""
        from datetime import datetime

        now = datetime.now()
        # 0=周一, 6=周日
        return now.weekday() < 5  # 周一到周五
