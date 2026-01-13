"""
Notifier - 通知器基类
"""

from abc import ABC, abstractmethod
from typing import Dict
from Monitor.EventDetector import CEvent
from Notification.MessageFormatter import CMessageFormatter


class CNotifier(ABC):
    """通知器基类"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 通知配置
        """
        self.config = config
        self.formatter = CMessageFormatter()

    @abstractmethod
    def send(self, event: CEvent):
        """
        发送通知

        Args:
            event: 事件对象
        """
        pass

    def test_send(self) -> bool:
        """
        测试发送功能

        Returns:
            是否发送成功
        """
        from datetime import datetime

        test_event = CEvent(
            type="test",
            code="000001",
            name="测试股票",
            level="low",
            title="测试通知",
            message="这是一条测试消息，如果您收到此消息，说明通知配置正常。",
            data={"test": True},
            timestamp=datetime.now()
        )

        try:
            self.send(test_event)
            return True
        except Exception as e:
            print(f"测试发送失败: {e}")
            return False
