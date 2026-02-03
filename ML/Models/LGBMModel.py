"""
LightGBM模型实现
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .BaseModel import BaseModel


class LGBMModel(BaseModel):
    """LightGBM模型"""

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化LightGBM模型

        Args:
            params: LightGBM参数字典
        """
        super().__init__(params)

        # 默认参数
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
        }

        # 合并参数
        default_params.update(self.params)
        self.params = default_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[List] = None,
            feature_names: Optional[List[str]] = None,
            **kwargs) -> 'LGBMModel':
        """
        训练LightGBM模型

        Args:
            X: 训练特征矩阵
            y: 训练标签
            eval_set: 验证集 [(X_val, y_val)]
            feature_names: 特征名称列表
            **kwargs: 其他参数（如early_stopping_rounds, verbose_eval）

        Returns:
            self
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Please install with: pip install lightgbm")

        # 保存特征名称
        self.feature_names = feature_names

        # 创建数据集
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        valid_sets = [train_data]
        valid_names = ['train']

        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # 训练参数
        train_params = {
            'early_stopping_rounds': kwargs.get('early_stopping_rounds', 50),
            'verbose_eval': kwargs.get('verbose_eval', 10),
        }

        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=train_params['early_stopping_rounds'], verbose=False),
                lgb.log_evaluation(period=train_params['verbose_eval']),
            ]
        )

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

        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

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

        # LightGBM返回正类概率
        proba_pos = self.model.predict(X, num_iteration=self.model.best_iteration)

        # 转换为两列格式 [负类概率, 正类概率]
        proba = np.vstack([1 - proba_pos, proba_pos]).T
        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性

        Returns:
            特征名到重要性分数的字典
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        importance = self.model.feature_importance(importance_type='gain')

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
