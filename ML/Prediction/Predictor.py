"""
预测器

为买卖点打分和排序
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from Chan import CChan
from BuySellPoint.BS_Point import CBS_Point
from ..FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ..Models.BaseModel import BaseModel


class Predictor:
    """预测器"""

    def __init__(self, model: BaseModel, feature_extractor: BSPFeatureExtractor = None):
        """
        初始化预测器

        Args:
            model: 训练好的模型
            feature_extractor: 特征提取器
        """
        self.model = model
        self.feature_extractor = feature_extractor

        if self.feature_extractor is None:
            self.feature_extractor = BSPFeatureExtractor()

    def predict_single(self, bsp: CBS_Point) -> Dict[str, Any]:
        """
        预测单个买卖点

        Args:
            bsp: 买卖点对象

        Returns:
            预测结果字典，包含：
                - score: 预测概率（0-1之间）
                - label: 预测标签（0或1）
        """
        # 提取特征
        features = self.feature_extractor.extract(bsp)

        # 转换为numpy数组
        feature_names = self.feature_extractor.get_feature_names()
        X = np.array([[features.get(name, 0.0) for name in feature_names]])

        # 预测
        proba = self.model.predict_proba(X)
        score = proba[0, 1]  # 正类概率

        label = self.model.predict(X)[0]

        return {
            'score': score,
            'label': label,
        }

    def predict_batch(self, bsp_list: List[CBS_Point]) -> List[Dict[str, Any]]:
        """
        批量预测买卖点

        Args:
            bsp_list: 买卖点列表

        Returns:
            预测结果列表
        """
        if len(bsp_list) == 0:
            return []

        # 批量提取特征
        features_list = []
        feature_names = self.feature_extractor.get_feature_names()

        for bsp in bsp_list:
            features = self.feature_extractor.extract(bsp)
            features_list.append([features.get(name, 0.0) for name in feature_names])

        X = np.array(features_list)

        # 批量预测
        proba = self.model.predict_proba(X)
        scores = proba[:, 1]  # 正类概率

        labels = self.model.predict(X)

        # 构建结果
        results = []
        for i in range(len(bsp_list)):
            results.append({
                'score': scores[i],
                'label': labels[i],
            })

        return results

    def rank_bsp(self, chan: CChan, top_k: int = 5, direction: str = 'all') -> List[Tuple[CBS_Point, float]]:
        """
        对CChan中所有买卖点打分并排序

        Args:
            chan: CChan实例
            top_k: 返回前K个高分买卖点
            direction: 过滤方向，'buy'只返回买点，'sell'只返回卖点，'all'返回所有

        Returns:
            [(买卖点, 分数), ...] 按分数降序排列
        """
        # 获取所有买卖点
        bsp_list = []
        if hasattr(chan, 'bs_point_lst'):
            for bsp_dict in chan.bs_point_lst.values():
                for bsp in bsp_dict:
                    # 过滤方向
                    if direction == 'buy' and not bsp.is_buy:
                        continue
                    if direction == 'sell' and bsp.is_buy:
                        continue

                    bsp_list.append(bsp)

        if len(bsp_list) == 0:
            return []

        # 批量预测
        results = self.predict_batch(bsp_list)

        # 排序
        bsp_scores = [(bsp, result['score']) for bsp, result in zip(bsp_list, results)]
        bsp_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前K个
        return bsp_scores[:top_k]

    def filter_bsp_by_threshold(self, chan: CChan, threshold: float = 0.7, direction: str = 'all') -> List[Tuple[CBS_Point, float]]:
        """
        过滤出分数超过阈值的买卖点

        Args:
            chan: CChan实例
            threshold: 分数阈值
            direction: 过滤方向

        Returns:
            [(买卖点, 分数), ...] 符合条件的买卖点列表
        """
        # 获取所有买卖点
        bsp_list = []
        if hasattr(chan, 'bs_point_lst'):
            for bsp_dict in chan.bs_point_lst.values():
                for bsp in bsp_dict:
                    # 过滤方向
                    if direction == 'buy' and not bsp.is_buy:
                        continue
                    if direction == 'sell' and bsp.is_buy:
                        continue

                    bsp_list.append(bsp)

        if len(bsp_list) == 0:
            return []

        # 批量预测
        results = self.predict_batch(bsp_list)

        # 过滤
        filtered = [(bsp, result['score']) for bsp, result in zip(bsp_list, results)
                    if result['score'] >= threshold]

        # 排序
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered
