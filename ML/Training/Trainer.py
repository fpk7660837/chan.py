"""
训练器

编排完整的模型训练流程：特征提取→标签构建→数据集分割→模型训练
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from Chan import CChan
from ..FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ..Models.ModelFactory import ModelFactory
from ..Models.BaseModel import BaseModel
from .LabelBuilder import LabelBuilder


class Trainer:
    """训练器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化训练器

        Args:
            config: 配置字典，包含：
                - feature_config: 特征配置
                - label_config: 标签配置
                - model_config: 模型配置
                - training_config: 训练配置
        """
        self.config = config or {}
        self.feature_config = self.config.get('feature_config', {})
        self.label_config = self.config.get('label_config', {})
        self.model_config = self.config.get('model_config', {})
        self.training_config = self.config.get('training_config', {})

        # 初始化组件
        self.feature_extractor = BSPFeatureExtractor(self.feature_config)
        self.label_builder = LabelBuilder(self.label_config)

    def train(self, chan_list: List[CChan], model_type: str = None) -> BaseModel:
        """
        训练模型

        Args:
            chan_list: CChan实例列表（训练数据）
            model_type: 模型类型，如果为None则使用配置中的类型

        Returns:
            训练好的模型
        """
        print("Step 1: Extracting buy/sell points from CChan instances...")
        bsp_list = self._extract_bsp_from_chan_list(chan_list)
        print(f"Found {len(bsp_list)} buy/sell points")

        print("\nStep 2: Extracting features...")
        X, feature_names = self._extract_features(bsp_list)
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")

        print("\nStep 3: Building labels...")
        y, returns = self.label_builder.build_labels(bsp_list)
        label_dist = self.label_builder.get_label_distribution(y)
        print(f"Label distribution: {label_dist['positive']} positive ({label_dist['positive_ratio']:.2%}), "
              f"{label_dist['negative']} negative ({label_dist['negative_ratio']:.2%})")

        print("\nStep 4: Splitting train/test sets...")
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

        print("\nStep 5: Training model...")
        model = self._train_model(X_train, y_train, X_test, y_test, feature_names, model_type)
        print("Model training completed!")

        # 显示特征重要性
        print("\nTop 10 important features:")
        top_features = model.get_top_features(top_k=10)
        for i, (feature_name, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature_name}: {importance:.4f}")

        return model

    def _extract_bsp_from_chan_list(self, chan_list: List[CChan]) -> List:
        """从CChan列表中提取所有买卖点"""
        bsp_list = []

        for chan in chan_list:
            # 获取所有买卖点
            if hasattr(chan, 'bs_point_lst'):
                for bsp_dict in chan.bs_point_lst.values():
                    for bsp in bsp_dict:
                        bsp_list.append(bsp)

        return bsp_list

    def _extract_features(self, bsp_list: List) -> Tuple[np.ndarray, List[str]]:
        """提取特征"""
        features_list = []
        feature_names = None

        for bsp in bsp_list:
            features = self.feature_extractor.extract(bsp)
            features_list.append(features)

            if feature_names is None:
                feature_names = list(features.keys())

        # 转换为numpy数组
        X = np.array([[feat_dict[name] for name in feature_names] for feat_dict in features_list])

        return X, feature_names

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分割训练集和测试集"""
        test_size = self.training_config.get('test_size', 0.2)
        random_state = self.model_config.get('lightgbm_params', {}).get('random_state', 42)

        use_time_series_split = self.training_config.get('use_time_series_split', True)

        if use_time_series_split:
            # 时间序列分割（不打乱顺序）
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # 随机分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_names: List[str], model_type: str = None) -> BaseModel:
        """训练模型"""
        # 确定模型类型
        if model_type is None:
            model_type = self.model_config.get('model_type', 'lightgbm')

        # 获取模型参数
        param_key = f'{model_type}_params'
        model_params = self.model_config.get(param_key, {})

        # 创建模型
        model = ModelFactory.create_model(model_type, model_params)

        # 训练参数
        train_kwargs = {
            'early_stopping_rounds': self.training_config.get('early_stopping_rounds', 50),
            'verbose_eval': self.training_config.get('verbose_eval', 10),
        }

        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            feature_names=feature_names,
            **train_kwargs
        )

        return model
