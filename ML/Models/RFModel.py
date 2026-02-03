"""
RandomForest模型实现
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .BaseModel import BaseModel


class RFModel(BaseModel):
    """RandomForest模型"""

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化RandomForest模型

        Args:
            params: RandomForest参数字典
        """
        super().__init__(params)

        # 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
        }

        # 合并参数
        default_params.update(self.params)
        self.params = default_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[List] = None,
            feature_names: Optional[List[str]] = None,
            **kwargs) -> 'RFModel':
        """
        训练RandomForest模型

        Args:
            X: 训练特征矩阵
            y: 训练标签
            eval_set: 验证集（RandomForest不使用验证集）
            feature_names: 特征名称列表
            **kwargs: 其他参数

        Returns:
            self
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("scikit-learn not installed. Please install with: pip install scikit-learn")

        # 保存特征名称
        self.feature_names = feature_names

        # 创建并训练模型
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征矩阵

        Returns:
            预测类别数组（0或1）
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵

        Returns:
            预测概率数组，shape为(n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性

        Returns:
            特征名到重要性分数的字典
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        importance = self.model.feature_importances_

        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}

    def get_top_features(self, top_k: int = 10) -> List[tuple]:
        """
        获取最重要的K个特征

        Args:
            top_k: 返回前K个特征

        Returns:
            [(特征名, 重要性分数), ...] 按重要性降序排列
        """
        importance_dict = self.get_feature_importance()
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]
