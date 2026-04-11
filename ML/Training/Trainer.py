"""
训练器

编排完整的模型训练流程：样本收集→特征提取→标签构建→时间序列切分→模型训练。
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from Chan import CChan
from ..Evaluation.Metrics import Metrics
from ..FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ..Models.BaseModel import BaseModel
from ..Models.ModelFactory import ModelFactory
from .LabelBuilder import LabelBuilder


class Trainer:
    """训练器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_config = self.config.get('feature_config', {})
        self.label_config = self.config.get('label_config', {})
        self.model_config = self.config.get('model_config', {})
        self.training_config = self.config.get('training_config', {})

        self.feature_extractor = BSPFeatureExtractor(self.feature_config)
        self.label_builder = LabelBuilder(self.label_config)
        self.last_split_info: Dict[str, Any] = {}
        self.last_dataset_profile: Dict[str, Any] = {}
        self.last_classification_metrics: Dict[str, Any] = {}
        self.last_overfit_risk: Dict[str, Any] = {}

    def train(self, chan_list: List[CChan], model_type: str = None) -> BaseModel:
        self.last_split_info = {}
        self.last_dataset_profile = {}
        self.last_classification_metrics = {}
        self.last_overfit_risk = {}

        print("Step 1: Collecting buy/sell points...")
        bsp_list = self._extract_bsp_from_chan_list(chan_list)
        if not bsp_list:
            raise ValueError("No eligible buy/sell points found for training.")
        print(f"Collected {len(bsp_list)} samples")

        print("\nStep 2: Extracting features...")
        X, feature_names = self._extract_features(bsp_list)
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")

        print("\nStep 3: Building labels...")
        y, returns = self.label_builder.build_labels(bsp_list)
        label_dist = self.label_builder.get_label_distribution(y)
        print(
            f"Label distribution: {label_dist['positive']} positive ({label_dist['positive_ratio']:.2%}), "
            f"{label_dist['negative']} negative ({label_dist['negative_ratio']:.2%})"
        )
        self.last_dataset_profile = {
            'total_samples': int(len(y)),
            'positive_samples': int(label_dist['positive']),
            'negative_samples': int(label_dist['negative']),
            'positive_ratio': float(label_dist['positive_ratio']),
        }

        print("\nStep 4: Splitting train/test sets...")
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        self.last_dataset_profile.update({
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
        })
        self._enforce_min_sample_guard()

        print("\nStep 5: Training model...")
        model = self._train_model(X_train, y_train, X_test, y_test, feature_names, model_type)
        print("Model training completed!")
        self.last_classification_metrics = self._evaluate_classification_metrics(model, X_train, y_train, X_test, y_test)
        self.last_overfit_risk = self._assess_overfit_risk()

        print("\nTop 10 important features:")
        top_features = model.get_top_features(top_k=10)
        for i, (feature_name, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature_name}: {importance:.4f}")

        return model

    def _iter_chan_bsp(self, chan: CChan, direction: str = 'all') -> List:
        bsp_store = chan[0].bs_point_lst.getSortedBspList()
        if direction == 'all':
            return bsp_store
        if direction == 'buy':
            return [bsp for bsp in bsp_store if bsp.is_buy]
        if direction == 'sell':
            return [bsp for bsp in bsp_store if not bsp.is_buy]
        raise ValueError(f"Unsupported direction: {direction}")

    def _extract_bsp_from_chan_list(self, chan_list: List[CChan]) -> List:
        signal_direction = self.training_config.get('signal_direction', 'buy')
        bsp_list = []

        for chan in chan_list:
            for bsp in self._iter_chan_bsp(chan, signal_direction):
                if bsp.klu is None:
                    continue
                bsp_list.append(bsp)

        # 样本必须按全市场时间排序，否则时间切分没有意义。
        bsp_list.sort(
            key=lambda bsp: (
                getattr(getattr(bsp, 'klu', None), 'time', None).ts if getattr(bsp, 'klu', None) is not None else -1,
                getattr(getattr(bsp, 'klu', None), 'idx', -1),
                getattr(getattr(getattr(bsp, 'bi', None), 'parent_seg', None), 'idx', -1),
            )
        )
        return bsp_list

    def _extract_features(self, bsp_list: List) -> Tuple[np.ndarray, List[str]]:
        feature_names = self.feature_extractor.get_feature_names()
        features_list = [self.feature_extractor.extract(bsp) for bsp in bsp_list]
        X = np.array(
            [[feat_dict.get(name, 0.0) for name in feature_names] for feat_dict in features_list],
            dtype=float,
        )
        return X, feature_names

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(X) < 2:
            raise ValueError("Need at least 2 samples to split train/test data.")

        test_size = self.training_config.get('test_size', 0.2)
        random_state = self.model_config.get('lightgbm_params', {}).get('random_state', 42)
        use_time_series_split = self.training_config.get('use_time_series_split', True)

        if use_time_series_split:
            n_splits = int(self.training_config.get('time_series_splits', 5))
            max_valid_splits = max(2, min(n_splits, len(X) - 1))

            if len(X) >= max_valid_splits + 1:
                splitter = TimeSeriesSplit(n_splits=max_valid_splits)
                train_idx, test_idx = None, None
                for train_idx, test_idx in splitter.split(X):
                    ...
                assert train_idx is not None and test_idx is not None
                self.last_split_info = {
                    'mode': 'walk_forward_last_fold',
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'n_splits': max_valid_splits,
                }
                return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

            split_idx = int(len(X) * (1 - test_size))
            split_idx = min(max(split_idx, 1), len(X) - 1)
            self.last_split_info = {
                'mode': 'time_holdout',
                'train_size': split_idx,
                'test_size': len(X) - split_idx,
                'n_splits': 1,
            }
            return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        self.last_split_info = {
            'mode': 'random',
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_splits': 1,
        }
        return X_train, X_test, y_train, y_test

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        model_type: str = None,
    ) -> BaseModel:
        if model_type is None:
            model_type = self.model_config.get('model_type', 'lightgbm')

        param_key = f'{model_type}_params'
        model_params = self.model_config.get(param_key, {})
        model = ModelFactory.create_model(model_type, model_params)

        train_kwargs = {
            'early_stopping_rounds': self.training_config.get('early_stopping_rounds', 50),
            'verbose_eval': self.training_config.get('verbose_eval', 10),
        }

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            feature_names=feature_names,
            **train_kwargs,
        )
        return model

    def _training_guardrails(self) -> Dict[str, Any]:
        return {
            'enforced': bool(self.training_config.get('enforce_min_samples', True)),
            'min_total_samples': int(self.training_config.get('min_total_samples', 100)),
            'min_train_samples': int(self.training_config.get('min_train_samples', 80)),
            'min_test_samples': int(self.training_config.get('min_test_samples', 20)),
            'max_auc_gap': float(self.training_config.get('max_auc_gap', 0.10)),
            'max_f1_gap': float(self.training_config.get('max_f1_gap', 0.15)),
            'min_test_auc': float(self.training_config.get('min_test_auc', 0.55)),
        }

    def _enforce_min_sample_guard(self) -> None:
        guardrails = self._training_guardrails()
        total_samples = int(self.last_dataset_profile.get('total_samples', 0) or 0)
        train_samples = int(self.last_dataset_profile.get('train_samples', 0) or 0)
        test_samples = int(self.last_dataset_profile.get('test_samples', 0) or 0)

        violations = []
        if total_samples < guardrails['min_total_samples']:
            violations.append(f"total={total_samples} < {guardrails['min_total_samples']}")
        if train_samples < guardrails['min_train_samples']:
            violations.append(f"train={train_samples} < {guardrails['min_train_samples']}")
        if test_samples < guardrails['min_test_samples']:
            violations.append(f"test={test_samples} < {guardrails['min_test_samples']}")

        if violations and guardrails['enforced']:
            raise ValueError(
                "Insufficient samples for reliable training: "
                + ", ".join(violations)
                + ". Increase the training universe or date range."
            )

    def _evaluate_classification_metrics(
        self,
        model: BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        train_metrics = self._calculate_split_metrics(model, X_train, y_train)
        test_metrics = self._calculate_split_metrics(model, X_test, y_test)

        generalization_gap: Dict[str, float] = {}
        for metric_name in sorted(set(train_metrics) & set(test_metrics)):
            generalization_gap[metric_name] = float(train_metrics[metric_name] - test_metrics[metric_name])

        return {
            'train': train_metrics,
            'test': test_metrics,
            'generalization_gap': generalization_gap,
        }

    def _calculate_split_metrics(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        y_pred = model.predict(X)
        y_proba_raw = model.predict_proba(X)
        y_proba = self._extract_positive_class_probability(y_proba_raw)
        return Metrics.calculate_classification_metrics(y, y_pred, y_proba)

    @staticmethod
    def _extract_positive_class_probability(y_proba: np.ndarray) -> np.ndarray:
        values = np.asarray(y_proba, dtype=float)
        if values.ndim == 2:
            if values.shape[1] == 0:
                return np.zeros(values.shape[0], dtype=float)
            if values.shape[1] == 1:
                return values[:, 0]
            return values[:, 1]
        return values.reshape(-1)

    def _assess_overfit_risk(self) -> Dict[str, Any]:
        guardrails = self._training_guardrails()
        reasons: List[str] = []
        risk_level = 'low'

        total_samples = int(self.last_dataset_profile.get('total_samples', 0) or 0)
        train_samples = int(self.last_dataset_profile.get('train_samples', 0) or 0)
        test_samples = int(self.last_dataset_profile.get('test_samples', 0) or 0)
        positive_ratio = float(self.last_dataset_profile.get('positive_ratio', 0.0) or 0.0)

        if total_samples < guardrails['min_total_samples']:
            reasons.append(f"total sample count {total_samples} is below recommended minimum {guardrails['min_total_samples']}")
            risk_level = 'high'
        if train_samples < guardrails['min_train_samples']:
            reasons.append(f"train sample count {train_samples} is below recommended minimum {guardrails['min_train_samples']}")
            risk_level = 'high'
        if test_samples < guardrails['min_test_samples']:
            reasons.append(f"test sample count {test_samples} is below recommended minimum {guardrails['min_test_samples']}")
            risk_level = 'high'
        if positive_ratio < 0.15 or positive_ratio > 0.85:
            reasons.append(f"label positive ratio {positive_ratio:.2%} indicates strong class imbalance")
            if risk_level != 'high':
                risk_level = 'medium'

        gap_metrics = self.last_classification_metrics.get('generalization_gap', {})
        auc_gap = float(gap_metrics.get('auc', 0.0) or 0.0)
        f1_gap = float(gap_metrics.get('f1', 0.0) or 0.0)
        if auc_gap > guardrails['max_auc_gap']:
            reasons.append(f"auc gap {auc_gap:.4f} exceeds limit {guardrails['max_auc_gap']:.4f}")
            risk_level = 'high'
        if f1_gap > guardrails['max_f1_gap']:
            reasons.append(f"f1 gap {f1_gap:.4f} exceeds limit {guardrails['max_f1_gap']:.4f}")
            risk_level = 'high'

        test_metrics = self.last_classification_metrics.get('test', {})
        test_auc = float(test_metrics.get('auc', 0.0) or 0.0)
        if test_auc < guardrails['min_test_auc']:
            reasons.append(f"test auc {test_auc:.4f} is below target {guardrails['min_test_auc']:.4f}")
            if risk_level != 'high':
                risk_level = 'medium'

        return {
            'level': risk_level,
            'reasons': reasons,
            'sample_guard': guardrails,
        }
