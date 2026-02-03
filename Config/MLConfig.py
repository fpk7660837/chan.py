"""
机器学习配置管理

集中管理所有ML相关配置，包括特征工程、标签构建、模型参数、训练配置、回测配置等
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class MLConfig:
    """机器学习全局配置类"""

    # 特征工程配置
    feature_config: Dict[str, Any] = field(default_factory=lambda: {
        'use_bi_features': True,        # 使用笔特征
        'use_seg_features': True,       # 使用线段特征
        'use_zs_features': True,        # 使用中枢特征
        'use_klu_features': True,       # 使用K线技术指标特征
        'use_multi_level': True,        # 使用多级别特征
        'level_list': ['day', 'week'],  # 使用的级别列表（日线+周线）
    })

    # 标签构建配置（基于未来收益率策略）
    label_config: Dict[str, Any] = field(default_factory=lambda: {
        'label_strategy': 'future_return',  # 标签策略：未来收益率
        'lookforward_bars': 20,             # 未来N根K线
        'threshold_pct': 0.05,              # 收益率阈值（5%）
        'use_highest_for_buy': True,        # 买点使用最高价
        'use_lowest_for_sell': True,        # 卖点使用最低价
    })

    # 模型配置
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_type': 'lightgbm',  # 默认使用LightGBM，可切换为xgboost/randomforest
        'lightgbm_params': {
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
        },
        'xgboost_params': {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        },
        'randomforest_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
        },
    })

    # 训练配置
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'test_size': 0.2,           # 测试集比例
        'cv_folds': 5,              # 交叉验证折数
        'use_time_series_split': True,  # 使用时间序列分割
        'early_stopping_rounds': 50,    # 早停轮数
        'verbose_eval': 10,         # 训练日志频率
    })

    # 预测配置
    prediction_config: Dict[str, Any] = field(default_factory=lambda: {
        'score_threshold': 0.7,     # 预测分数阈值
        'top_k': 5,                 # 返回前K个高分买卖点
        'return_proba': True,       # 返回概率而非类别
    })

    # 回测配置
    backtest_config: Dict[str, Any] = field(default_factory=lambda: {
        'score_threshold': 0.7,     # 交易信号阈值
        'holding_period': 20,       # 持仓周期（根K线）
        'initial_capital': 100000,  # 初始资金
        'commission_rate': 0.0003,  # 手续费率
        'slippage': 0.001,          # 滑点
        'max_position': 1.0,        # 最大仓位
    })

    # 评估指标配置（基于用户需求）
    evaluation_config: Dict[str, Any] = field(default_factory=lambda: {
        'metrics': [
            'auc',                  # AUC
            'precision',            # 精确率
            'recall',               # 召回率
            'f1',                   # F1分数
            'sharpe_ratio',         # 夏普比率
            'calmar_ratio',         # 卡玛比率
            'win_rate',             # 胜率
            'profit_loss_ratio',    # 盈亏比
            'max_drawdown',         # 最大回撤
        ],
    })

    # 模型保存配置
    model_io_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_dir': './models',            # 模型保存目录
        'save_format': 'pickle',            # 保存格式：pickle/joblib
        'include_feature_names': True,      # 保存特征名称
        'include_metadata': True,           # 保存元数据
    })

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'feature_config': self.feature_config,
            'label_config': self.label_config,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'prediction_config': self.prediction_config,
            'backtest_config': self.backtest_config,
            'evaluation_config': self.evaluation_config,
            'model_io_config': self.model_io_config,
        }

    def get_model_params(self, model_type: str = None) -> Dict[str, Any]:
        """获取指定模型的参数"""
        if model_type is None:
            model_type = self.model_config['model_type']

        param_key = f'{model_type}_params'
        if param_key in self.model_config:
            return self.model_config[param_key]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """更新指定类型的配置"""
        config_attr = f'{config_type}_config'
        if hasattr(self, config_attr):
            getattr(self, config_attr).update(updates)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
