"""
标签构建器

根据不同策略为买卖点构建标签，支持未来收益率策略
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from BuySellPoint.BS_Point import CBS_Point


class LabelBuilder:
    """标签构建器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化标签构建器

        Args:
            config: 标签配置字典，包含：
                - label_strategy: 标签策略（'future_return'）
                - lookforward_bars: 未来N根K线
                - threshold_pct: 收益率阈值
                - use_highest_for_buy: 买点使用最高价
                - use_lowest_for_sell: 卖点使用最低价
        """
        self.config = config or {}
        self.strategy = self.config.get('label_strategy', 'future_return')
        self.lookforward_bars = self.config.get('lookforward_bars', 20)
        self.threshold_pct = self.config.get('threshold_pct', 0.05)
        self.use_highest_for_buy = self.config.get('use_highest_for_buy', True)
        self.use_lowest_for_sell = self.config.get('use_lowest_for_sell', True)

    def build_labels(self, bsp_list: List[CBS_Point]) -> Tuple[np.ndarray, np.ndarray]:
        """
        为买卖点列表构建标签

        Args:
            bsp_list: 买卖点列表

        Returns:
            (labels, returns): 标签数组和收益率数组
        """
        if self.strategy == 'future_return':
            return self._future_return_labels(bsp_list)
        else:
            raise ValueError(f"Unknown label strategy: {self.strategy}")

    def _future_return_labels(self, bsp_list: List[CBS_Point]) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于未来收益率的标签策略

        对每个买卖点：
        1. 获取当前价格
        2. 遍历未来N根K线
        3. 买点：计算未来最高价的收益率
        4. 卖点：计算未来最低价的收益率
        5. 超过阈值则标签=1，否则=0

        Args:
            bsp_list: 买卖点列表

        Returns:
            (labels, returns): 标签数组和收益率数组
        """
        labels = []
        returns = []

        for bsp in bsp_list:
            label, ret = self._calculate_future_return(bsp)
            labels.append(label)
            returns.append(ret)

        return np.array(labels), np.array(returns)

    def _calculate_future_return(self, bsp: CBS_Point) -> Tuple[int, float]:
        """
        计算单个买卖点的未来收益率

        Args:
            bsp: 买卖点对象

        Returns:
            (label, return): 标签和收益率
        """
        if bsp.klu is None:
            return 0, 0.0

        current_price = bsp.klu.close
        future_prices = self._get_future_prices(bsp.klu)

        if len(future_prices) == 0:
            # 没有足够的未来数据，标签为0
            return 0, 0.0

        # 根据买卖点方向计算收益率
        if bsp.is_buy:
            # 买点：计算未来最高价的收益率
            if self.use_highest_for_buy:
                future_price = max(future_prices)
            else:
                future_price = future_prices[-1]  # 使用最后一根K线的收盘价

            ret = (future_price - current_price) / current_price
        else:
            # 卖点：计算未来最低价的收益率（做空）
            if self.use_lowest_for_sell:
                future_price = min(future_prices)
            else:
                future_price = future_prices[-1]

            ret = (current_price - future_price) / current_price

        # 判断是否超过阈值
        label = 1 if ret >= self.threshold_pct else 0

        return label, ret

    def _get_future_prices(self, klu) -> List[float]:
        """
        获取未来N根K线的价格

        Args:
            klu: 当前K线

        Returns:
            未来N根K线的价格列表（收盘价）
        """
        prices = []
        current = klu

        for _ in range(self.lookforward_bars):
            if hasattr(current, 'next') and current.next is not None:
                current = current.next
                prices.append(current.close)
            else:
                break

        return prices

    def get_label_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        获取标签分布统计

        Args:
            labels: 标签数组

        Returns:
            统计信息字典
        """
        total = len(labels)
        positive = np.sum(labels == 1)
        negative = np.sum(labels == 0)

        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_ratio': positive / total if total > 0 else 0.0,
            'negative_ratio': negative / total if total > 0 else 0.0,
        }
