"""
Scanner - 股票池扫描器

负责扫描股票池并维护CChan对象缓存
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE

logger = logging.getLogger(__name__)


class CScanner:
    """股票池扫描器"""

    def __init__(self, scan_interval: int = 60):
        """
        Args:
            scan_interval: 扫描间隔（秒）
        """
        self.scan_interval = scan_interval
        self.chan_cache: Dict[str, CChan] = {}  # CChan对象缓存
        self.last_scan_time: Dict[str, datetime] = {}  # 上次扫描时间

    def scan(self,
             stock_pool: List[str],
             data_src: DATA_SRC,
             lv_list: List[KL_TYPE],
             chan_config: Dict,
             lookback_days: int = 365) -> Dict[str, CChan]:
        """
        扫描股票池，返回更新后的CChan字典

        策略：
        - 首次扫描：创建CChan对象，加载历史数据
        - 后续扫描：使用现有CChan对象（暂不支持增量更新，每次重新加载）

        Args:
            stock_pool: 股票池
            data_src: 数据源
            lv_list: K线级别列表
            chan_config: 缠论配置
            lookback_days: 回溯天数

        Returns:
            {code: CChan对象}
        """
        result = {}
        begin_time = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_time = datetime.now().strftime("%Y-%m-%d")

        # 准备配置
        config = CChanConfig(chan_config)

        for code in stock_pool:
            try:
                # 检查是否需要更新
                if code in self.chan_cache:
                    # 使用缓存（简化版：暂不支持增量更新）
                    result[code] = self.chan_cache[code]
                else:
                    # 首次扫描，创建CChan对象
                    logger.info(f"初始化 {code} 的缠论数据...")
                    chan = CChan(
                        code=code,
                        begin_time=begin_time,
                        end_time=end_time,
                        data_src=data_src,
                        lv_list=lv_list,
                        config=config,
                        autype=AUTYPE.QFQ,
                    )
                    self.chan_cache[code] = chan
                    result[code] = chan

                self.last_scan_time[code] = datetime.now()

            except Exception as e:
                logger.error(f"扫描 {code} 失败: {e}")

        return result

    def clear_cache(self):
        """清空缓存"""
        self.chan_cache.clear()
        self.last_scan_time.clear()

    def remove_from_cache(self, code: str):
        """从缓存中移除指定股票"""
        if code in self.chan_cache:
            del self.chan_cache[code]
        if code in self.last_scan_time:
            del self.last_scan_time[code]

    def update_stock(self, code: str, data_src: DATA_SRC, lv_list: List[KL_TYPE],
                     chan_config: Dict, lookback_days: int = 365) -> Optional[CChan]:
        """
        强制更新指定股票的数据

        Args:
            code: 股票代码
            data_src: 数据源
            lv_list: K线级别
            chan_config: 缠论配置
            lookback_days: 回溯天数

        Returns:
            更新后的CChan对象
        """
        # 移除缓存
        self.remove_from_cache(code)

        # 重新扫描
        result = self.scan([code], data_src, lv_list, chan_config, lookback_days)

        return result.get(code)
