"""
MonitorEngine - 监控引擎核心

负责定期扫描股票池、检测事件并发送通知
"""

import threading
import time
import logging
from typing import List, Dict, Optional

from Monitor.MonitorConfig import CMonitorConfig
from Monitor.Scanner import CScanner
from Monitor.EventDetector import CEventDetector, CEvent

logger = logging.getLogger(__name__)


class CMonitorEngine:
    """
    监控引擎核心类

    职责：
    - 定期扫描股票池
    - 运行所有事件检测器
    - 触发通知
    - 记录事件日志
    """

    def __init__(self, config: CMonitorConfig):
        """
        Args:
            config: 监控配置
        """
        self.config = config
        self.scanner = CScanner(config.scan_interval)
        self.detectors: List[CEventDetector] = []
        self.notification_service = None  # 将在添加时设置
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

    def add_detector(self, detector: CEventDetector):
        """添加事件检测器"""
        self.detectors.append(detector)
        logger.info(f"添加检测器: {detector.__class__.__name__}")

    def set_notification_service(self, service):
        """设置通知服务"""
        self.notification_service = service
        logger.info("通知服务已设置")

    def start(self):
        """启动监控（后台线程）"""
        if self.running:
            logger.warning("监控已经在运行")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("监控引擎已启动")

    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("监控引擎已停止")

    def _monitor_loop(self):
        """监控主循环"""
        logger.info("监控循环开始")

        while self.running:
            try:
                # 检查是否在交易时间
                if not self.config.is_trading_day():
                    logger.info("非交易日，跳过扫描")
                    time.sleep(60)
                    continue

                if not self.config.is_trading_time():
                    logger.debug("非交易时间，等待...")
                    time.sleep(60)
                    continue

                # 执行扫描
                logger.info(f"开始扫描股票池（{len(self.config.stock_pool)}只股票）...")
                self._scan_and_detect()

                # 等待下次扫描
                logger.info(f"等待{self.config.scan_interval}秒后进行下次扫描...")
                time.sleep(self.config.scan_interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}", exc_info=True)
                time.sleep(10)  # 出错后等待10秒再继续

    def _scan_and_detect(self):
        """扫描并检测事件"""

        # 1. 扫描股票池
        try:
            chan_dict = self.scanner.scan(
                stock_pool=self.config.stock_pool,
                data_src=self.config.data_src,
                lv_list=self.config.lv_list,
                chan_config=self.config.chan_config,
                lookback_days=self.config.lookback_days
            )
            logger.info(f"成功扫描 {len(chan_dict)} 只股票")

        except Exception as e:
            logger.error(f"扫描失败: {e}")
            return

        # 2. 运行所有检测器
        all_events = []
        for detector in self.detectors:
            try:
                events = detector.detect(chan_dict)
                all_events.extend(events)
                logger.info(f"{detector.__class__.__name__} 检测到 {len(events)} 个事件")
            except Exception as e:
                logger.error(f"{detector.__class__.__name__} 检测失败: {e}", exc_info=True)

        # 3. 过滤和去重
        filtered_events = self._filter_events(all_events)
        logger.info(f"过滤后剩余 {len(filtered_events)} 个事件")

        # 4. 发送通知
        for event in filtered_events:
            try:
                logger.info(f"事件: {event}")

                if self.notification_service:
                    self.notification_service.send(event)
                    logger.info(f"已发送通知: {event.title}")

            except Exception as e:
                logger.error(f"发送通知失败: {e}", exc_info=True)

    def _filter_events(self, events: List[CEvent]) -> List[CEvent]:
        """
        过滤和去重事件

        Args:
            events: 原始事件列表

        Returns:
            过滤后的事件列表
        """
        # 简单去重：同一股票同一类型的事件只保留一个
        seen = set()
        filtered = []

        # 按级别排序（high > medium > low）
        level_priority = {"high": 0, "medium": 1, "low": 2}
        events.sort(key=lambda e: level_priority.get(e.level, 3))

        for event in events:
            key = f"{event.code}_{event.type}"
            if key not in seen:
                filtered.append(event)
                seen.add(key)

        return filtered

    def scan_once(self) -> Dict:
        """
        手动执行一次扫描（用于测试）

        Returns:
            扫描结果和事件
        """
        logger.info("执行单次扫描...")

        # 扫描股票池
        chan_dict = self.scanner.scan(
            stock_pool=self.config.stock_pool,
            data_src=self.config.data_src,
            lv_list=self.config.lv_list,
            chan_config=self.config.chan_config,
            lookback_days=self.config.lookback_days
        )

        # 运行检测器
        all_events = []
        for detector in self.detectors:
            events = detector.detect(chan_dict)
            all_events.extend(events)

        # 过滤事件
        filtered_events = self._filter_events(all_events)

        return {
            "chan_dict": chan_dict,
            "events": filtered_events,
            "event_count": len(filtered_events),
        }
