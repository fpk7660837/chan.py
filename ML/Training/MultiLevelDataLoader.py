"""
多级别训练数据加载边界。

本模块只定义训练路径需要的最小接口，不绑定具体存储实现。
后续可以由 ClickHouse 或其他后端提供真正的数据加载器。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


class MultiLevelDataLoader:
    """多级别训练数据加载边界。"""

    def load_training_contexts(
        self,
        universe: Sequence[Tuple[str, str]],
        *,
        begin_time: str,
        end_time: str,
        levels: Sequence[str],
        universe_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> Tuple[List[Any], List[Tuple[str, str]]]:
        raise NotImplementedError(
            "Multi-level market-data loader is not implemented. "
            "Provide a concrete backend in the external storage task."
        )
