"""
XGBoost模型实现
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .BaseModel import BaseModel


class XGBModel(BaseModel):
    """XGBoost模型"""

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化XGBoost模型

        Args:
            params: XGBoost参数字典
        """
        super().__init__(params)

        # 默认参数
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }

        # 合并参数
        default_params.update(self.params)
        self.params = default_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[List] = None,
            feature_names: Optional[List[str]] = None,
            **kwargs) -> 'XGBModel':
        """
        训练XGBoost模型

        Args:
            X: 训练特征矩阵
            y: 训练标签
            eval_set: 验证集 [(X_val, y_val)]
            feature_names: 特征名称列表
            **kwargs: 其他参数

        Returns:
            self
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Please install with: pip install xgboost")

        # 保存特征名称
        self.feature_names = feature_names

        # 创建模型
        self.model = xgb.XGBClassifier(**self.params)

        # 训练参数
        fit_params = {
            'verbose': kwargs.get('verbose_eval', 10),
        }

        if eval_set is not None and len(eval_set) > 0:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = kwargs.get('early_stopping_rounds', 50)

        # 训练模型
        self.model.fit(X, y, **fit_params)

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
