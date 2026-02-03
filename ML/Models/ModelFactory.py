"""
模型工厂

实现不同算法的可切换性
"""

from typing import Dict, Any
from .BaseModel import BaseModel
from .LGBMModel import LGBMModel
from .XGBModel import XGBModel
from .RFModel import RFModel


class ModelFactory:
    """模型工厂类"""

    # 支持的模型类型
    _MODEL_REGISTRY = {
        'lightgbm': LGBMModel,
        'lgbm': LGBMModel,
        'xgboost': XGBModel,
        'xgb': XGBModel,
        'randomforest': RFModel,
        'rf': RFModel,
    }

    @classmethod
    def create_model(cls, model_type: str, params: Dict[str, Any] = None) -> BaseModel:
        """
        创建模型实例

        Args:
            model_type: 模型类型，支持：
                - 'lightgbm', 'lgbm': LightGBM模型
                - 'xgboost', 'xgb': XGBoost模型
                - 'randomforest', 'rf': RandomForest模型
            params: 模型参数字典

        Returns:
            模型实例

        Raises:
            ValueError: 如果模型类型不支持
        """
        model_type_lower = model_type.lower()

        if model_type_lower not in cls._MODEL_REGISTRY:
            supported = ', '.join(cls._MODEL_REGISTRY.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported}")

        model_class = cls._MODEL_REGISTRY[model_type_lower]
        return model_class(params)

    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """
        注册新的模型类型

        Args:
            model_type: 模型类型标识
            model_class: 模型类（必须继承自BaseModel）
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class must inherit from BaseModel")

        cls._MODEL_REGISTRY[model_type.lower()] = model_class

    @classmethod
    def get_supported_models(cls) -> list:
        """获取支持的模型类型列表"""
        return list(cls._MODEL_REGISTRY.keys())
