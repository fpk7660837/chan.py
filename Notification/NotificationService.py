"""
NotificationService - 通知服务核心

管理多个通知渠道，提供统一的发送接口
"""

import logging
from typing import Dict

from Monitor.EventDetector import CEvent
from Notification.Notifier import CNotifier
from Notification.DingTalkNotifier import CDingTalkNotifier
from Notification.WeChatNotifier import CWeChatNotifier
from Notification.FeishuNotifier import CFeishuNotifier

logger = logging.getLogger(__name__)


class CNotificationService:
    """通知服务核心"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 通知配置字典，格式：
                {
                    "dingtalk": {"webhook_url": "...", "secret": "..."},
                    "wechat": {"webhook_url": "..."},
                    "feishu": {"webhook_url": "..."}
                }
        """
        self.notifiers: Dict[str, CNotifier] = {}
        self._init_notifiers(config)

    def _init_notifiers(self, config: Dict):
        """根据配置初始化通知器"""

        if "dingtalk" in config:
            try:
                self.notifiers["dingtalk"] = CDingTalkNotifier(config["dingtalk"])
                logger.info("钉钉通知器已初始化")
            except Exception as e:
                logger.error(f"钉钉通知器初始化失败: {e}")

        if "wechat" in config:
            try:
                self.notifiers["wechat"] = CWeChatNotifier(config["wechat"])
                logger.info("企业微信通知器已初始化")
            except Exception as e:
                logger.error(f"企业微信通知器初始化失败: {e}")

        if "feishu" in config:
            try:
                self.notifiers["feishu"] = CFeishuNotifier(config["feishu"])
                logger.info("飞书通知器已初始化")
            except Exception as e:
                logger.error(f"飞书通知器初始化失败: {e}")

        if not self.notifiers:
            logger.warning("未配置任何通知渠道")

    def send(self, event: CEvent):
        """
        发送事件通知到所有已配置的渠道

        Args:
            event: 事件对象
        """
        if not self.notifiers:
            logger.warning("没有可用的通知渠道")
            return

        for name, notifier in self.notifiers.items():
            try:
                notifier.send(event)
                logger.info(f"通知已发送到 {name}")
            except Exception as e:
                logger.error(f"{name} 通知发送失败: {e}")

    def test_all(self):
        """测试所有通知渠道"""
        results = {}

        for name, notifier in self.notifiers.items():
            try:
                success = notifier.test_send()
                results[name] = success
                if success:
                    print(f"✓ {name} 测试成功")
                else:
                    print(f"✗ {name} 测试失败")
            except Exception as e:
                results[name] = False
                print(f"✗ {name} 测试失败: {e}")

        return results
