"""
FeishuNotifier - 飞书机器人通知器

支持飞书群机器人webhook通知
"""

import requests
from typing import Dict

from Monitor.EventDetector import CEvent
from Notification.Notifier import CNotifier


class CFeishuNotifier(CNotifier):
    """飞书机器人通知器"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典，包含：
                - webhook_url: webhook地址
        """
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")

        if not self.webhook_url:
            raise ValueError("webhook_url is required")

    def send(self, event: CEvent):
        """发送飞书消息"""

        # 格式化消息
        message = self.formatter.format_text(event)

        # 构造payload（飞书使用text类型）
        payload = {
            "msg_type": "text",
            "content": {
                "text": message
            }
        }

        # 发送请求
        response = requests.post(self.webhook_url, json=payload, timeout=10)

        # 检查响应
        if response.status_code != 200:
            raise Exception(f"飞书通知发送失败: HTTP {response.status_code}, {response.text}")

        result = response.json()
        if result.get("code") != 0:
            raise Exception(f"飞书通知发送失败: {result.get('msg')}")
