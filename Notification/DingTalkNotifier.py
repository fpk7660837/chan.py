"""
DingTalkNotifier - 钉钉机器人通知器

支持钉钉群机器人webhook通知
"""

import requests
import time
import hmac
import hashlib
import base64
import urllib.parse
from typing import Dict

from Monitor.EventDetector import CEvent
from Notification.Notifier import CNotifier


class CDingTalkNotifier(CNotifier):
    """钉钉机器人通知器"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典，包含：
                - webhook_url: webhook地址
                - secret: 加签密钥（可选）
        """
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        self.secret = config.get("secret")

        if not self.webhook_url:
            raise ValueError("webhook_url is required")

    def send(self, event: CEvent):
        """发送钉钉消息"""

        # 格式化消息
        message = self.formatter.format_markdown(event)

        # 构造payload
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": event.title,
                "text": message
            }
        }

        # 计算签名（如果配置了secret）
        url = self.webhook_url
        if self.secret:
            timestamp = str(round(time.time() * 1000))
            sign = self._calculate_sign(timestamp, self.secret)
            url = f"{url}&timestamp={timestamp}&sign={sign}"

        # 发送请求
        response = requests.post(url, json=payload, timeout=10)

        # 检查响应
        if response.status_code != 200:
            raise Exception(f"钉钉通知发送失败: HTTP {response.status_code}, {response.text}")

        result = response.json()
        if result.get("errcode") != 0:
            raise Exception(f"钉钉通知发送失败: {result.get('errmsg')}")

    def _calculate_sign(self, timestamp: str, secret: str) -> str:
        """
        计算钉钉加签

        Args:
            timestamp: 时间戳（毫秒）
            secret: 密钥

        Returns:
            签名字符串
        """
        # 拼接时间戳和密钥
        string_to_sign = f"{timestamp}\n{secret}"

        # HMAC-SHA256加密
        hmac_code = hmac.new(
            secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()

        # Base64编码
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

        return sign
