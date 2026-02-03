"""
多级别特征提取器

从多个时间级别提取特征并整合，例如日线+周线联立特征
"""

from typing import Dict, List, Any, Optional
from BuySellPoint.BS_Point import CBS_Point
from .BSPFeatureExtractor import BSPFeatureExtractor


class MultiLevelExtractor:
    """多级别特征提取器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化多级别特征提取器

        Args:
            config: 配置字典，包含：
                - level_list: 级别列表，如['day', 'week']
                - 其他BSPFeatureExtractor的配置
        """
        self.config = config or {}
        self.level_list = self.config.get('level_list', ['day'])

        # 为每个级别创建特征提取器
        self.extractors = {}
        for level in self.level_list:
            self.extractors[level] = BSPFeatureExtractor(self.config)

    def extract(self, bsp: CBS_Point, level_name: str = 'day') -> Dict[str, float]:
        """
        提取单个级别的特征

        Args:
            bsp: 买卖点对象
            level_name: 级别名称

        Returns:
            特征字典，key带级别前缀
        """
        if level_name not in self.extractors:
            return {}

        # 提取特征
        features = self.extractors[level_name].extract(bsp)

        # 添加级别前缀
        prefixed_features = {}
        for key, value in features.items():
            prefixed_features[f'{level_name}_{key}'] = value

        return prefixed_features

    def extract_multi_level(self, bsp_dict: Dict[str, CBS_Point]) -> Dict[str, float]:
        """
        提取多级别特征并整合

        Args:
            bsp_dict: 级别到买卖点的映射，如{'day': bsp_day, 'week': bsp_week}

        Returns:
            整合后的特征字典
        """
        all_features = {}

        for level_name, bsp in bsp_dict.items():
            if bsp is not None and level_name in self.level_list:
                features = self.extract(bsp, level_name)
                all_features.update(features)

        return all_features

    def extract_with_sup_level(self, bsp: CBS_Point) -> Dict[str, float]:
        """
        提取买卖点及其上级级别的特征

        Args:
            bsp: 买卖点对象

        Returns:
            包含当前级别和上级级别的特征字典
        """
        all_features = {}

        # 提取当前级别特征
        current_level = 'day'  # 默认为日线
        current_features = self.extract(bsp, current_level)
        all_features.update(current_features)

        # 尝试提取上级级别特征
        try:
            # 通过K线的sup_kl向上追溯
            if bsp.klu is not None and hasattr(bsp.klu, 'sup_kl') and bsp.klu.sup_kl is not None:
                sup_klu = bsp.klu.sup_kl

                # 需要找到对应的上级买卖点（这需要从上级CChan中查找）
                # 由于无法直接访问上级CChan，这里先留空
                # 实际使用时需要传入完整的bsp_dict
                pass

        except Exception as e:
            pass

        return all_features

    def get_feature_names(self) -> List[str]:
        """获取所有级别的特征名称列表"""
        feature_names = []

        for level_name in self.level_list:
            if level_name in self.extractors:
                base_names = self.extractors[level_name].get_feature_names()
                prefixed_names = [f'{level_name}_{name}' for name in base_names]
                feature_names.extend(prefixed_names)

        return feature_names
