"""
模型IO工具

提供模型的保存和加载功能
"""

import os
import pickle
import json
from typing import Dict, Any, Optional
from datetime import datetime
from ..Models.BaseModel import BaseModel


class ModelIO:
    """模型IO管理器"""

    def __init__(self, model_dir: str = './models'):
        """
        初始化ModelIO

        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir

        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)

    def save(self, model: BaseModel, version: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        保存模型

        Args:
            model: 模型对象
            version: 版本号（如'v1.0'），如果为None则使用时间戳
            metadata: 元数据（训练参数、特征名称等）

        Returns:
            保存的文件路径
        """
        # 生成版本号
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 模型文件名
        model_filename = f'model_{version}.pkl'
        model_path = os.path.join(self.model_dir, model_filename)

        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved to: {model_path}")

        # 保存元数据
        if metadata is not None:
            metadata_filename = f'metadata_{version}.json'
            metadata_path = os.path.join(self.model_dir, metadata_filename)

            # 添加额外信息
            metadata['version'] = version
            metadata['save_time'] = datetime.now().isoformat()
            metadata['model_type'] = model.__class__.__name__

            # 添加特征名称
            if hasattr(model, 'feature_names') and model.feature_names is not None:
                metadata['feature_names'] = model.feature_names

            # 保存
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"Metadata saved to: {metadata_path}")

        return model_path

    def load(self, version: str = None) -> BaseModel:
        """
        加载模型

        Args:
            version: 版本号，如果为None则加载最新版本

        Returns:
            模型对象
        """
        # 如果没有指定版本，加载最新版本
        if version is None:
            version = self._get_latest_version()
            if version is None:
                raise FileNotFoundError(f"No models found in {self.model_dir}")

        # 模型文件路径
        model_filename = f'model_{version}.pkl'
        model_path = os.path.join(self.model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Model loaded from: {model_path}")

        return model

    def load_metadata(self, version: str = None) -> Dict[str, Any]:
        """
        加载元数据

        Args:
            version: 版本号，如果为None则加载最新版本

        Returns:
            元数据字典
        """
        # 如果没有指定版本，加载最新版本
        if version is None:
            version = self._get_latest_version()
            if version is None:
                raise FileNotFoundError(f"No metadata found in {self.model_dir}")

        # 元数据文件路径
        metadata_filename = f'metadata_{version}.json'
        metadata_path = os.path.join(self.model_dir, metadata_filename)

        if not os.path.exists(metadata_path):
            return {}

        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def list_versions(self) -> list:
        """
        列出所有保存的版本

        Returns:
            版本号列表，按时间降序排列
        """
        versions = []

        for filename in os.listdir(self.model_dir):
            if filename.startswith('model_') and filename.endswith('.pkl'):
                version = filename[6:-4]  # 提取版本号
                versions.append(version)

        # 排序
        versions.sort(reverse=True)

        return versions

    def _get_latest_version(self) -> Optional[str]:
        """获取最新版本号"""
        versions = self.list_versions()
        return versions[0] if len(versions) > 0 else None

    def delete_version(self, version: str):
        """
        删除指定版本的模型和元数据

        Args:
            version: 版本号
        """
        # 删除模型文件
        model_path = os.path.join(self.model_dir, f'model_{version}.pkl')
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Deleted model: {model_path}")

        # 删除元数据文件
        metadata_path = os.path.join(self.model_dir, f'metadata_{version}.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"Deleted metadata: {metadata_path}")
