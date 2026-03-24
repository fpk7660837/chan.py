"""
训练模块

提供标签构建、数据集构建、模型训练等功能
"""

from .LabelBuilder import LabelBuilder
try:
    from .Trainer import Trainer
except ModuleNotFoundError:
    Trainer = None

__all__ = [
    'LabelBuilder',
    'Trainer',
]
