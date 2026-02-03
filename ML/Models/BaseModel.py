"""
模型抽象基类

定义统一的模型接口，支持不同算法的可切换性
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class BaseModel(ABC):
    """模型抽象基类"""

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化模型

        Args:
            params: 模型参数字典
        """
        self.params = params or {}
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[List] = None,
            feature_names: Optional[List[str]] = None,
            **kwargs) -> 'BaseModel':
        """
        训练模型

        Args:
            X: 训练特征矩阵
            y: 训练标签
            eval_set: 验证集 [(X_val, y_val)]
            feature_names: 特征名称列表
            **kwargs: 其他参数

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征矩阵

        Returns:
            预测类别数组
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵

        Returns:
            预测概率数组，shape为(n_samples, n_classes)
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性

        Returns:
            特征名到重要性分数的字典
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.params

    def set_params(self, **params):
        """设置模型参数"""
        self.params.update(params)
        return self

    def save(self, filepath: str):
        """
        保存模型（子类可覆盖以实现特定保存逻辑）

        Args:
            filepath: 保存路径
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        加载模型（子类可覆盖以实现特定加载逻辑）

        Args:
            filepath: 模型路径

        Returns:
            加载的模型对象
        """
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
