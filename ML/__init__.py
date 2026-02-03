"""
机器学习模块 - 缠论框架ML能力扩展

提供完整的特征工程→模型训练→预测→回测评估体系
"""

__version__ = "1.0.0"

from .FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from .Models.ModelFactory import ModelFactory
from .Training.Trainer import Trainer
from .Prediction.Predictor import Predictor
from .Evaluation.Metrics import Metrics
from .Backtest.MLBacktest import MLBacktest

__all__ = [
    'BSPFeatureExtractor',
    'ModelFactory',
    'Trainer',
    'Predictor',
    'Metrics',
    'MLBacktest',
]
