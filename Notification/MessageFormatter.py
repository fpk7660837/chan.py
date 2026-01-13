"""
MessageFormatter - 消息格式化器

将事件格式化为各种通知渠道支持的格式
"""

from Monitor.EventDetector import CEvent


class CMessageFormatter:
    """消息格式化器"""

    def format_markdown(self, event: CEvent) -> str:
        """
        格式化为Markdown格式

        Args:
            event: 事件对象

        Returns:
            Markdown格式的消息
        """
        # 根据级别选择emoji
        level_emoji = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }

        emoji = level_emoji.get(event.level, "⚪")

        # 基础消息
        lines = [
            f"### {emoji} {event.title}",
            "",
            f"**股票**: {event.code} {event.name}",
            f"**时间**: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**消息**: {event.message}",
        ]

        # 添加详细数据
        if event.data:
            lines.append("")
            lines.append("**详情**:")
            for key, value in event.data.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def format_text(self, event: CEvent) -> str:
        """
        格式化为纯文本格式

        Args:
            event: 事件对象

        Returns:
            纯文本格式的消息
        """
        lines = [
            f"【{event.level.upper()}】{event.title}",
            f"股票: {event.code} {event.name}",
            f"时间: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"消息: {event.message}",
        ]

        if event.data:
            lines.append("详情:")
            for key, value in event.data.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def format_html(self, event: CEvent) -> str:
        """
        格式化为HTML格式

        Args:
            event: 事件对象

        Returns:
            HTML格式的消息
        """
        level_color = {
            "high": "#ff0000",
            "medium": "#ff9900",
            "low": "#00cc00"
        }

        color = level_color.get(event.level, "#666666")

        html = f"""
<div style="border-left: 4px solid {color}; padding-left: 10px;">
    <h3 style="color: {color};">{event.title}</h3>
    <p><strong>股票:</strong> {event.code} {event.name}</p>
    <p><strong>时间:</strong> {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>消息:</strong> {event.message}</p>
"""

        if event.data:
            html += "    <p><strong>详情:</strong></p>\n    <ul>\n"
            for key, value in event.data.items():
                if isinstance(value, float):
                    html += f"        <li>{key}: {value:.4f}</li>\n"
                else:
                    html += f"        <li>{key}: {value}</li>\n"
            html += "    </ul>\n"

        html += "</div>"

        return html
