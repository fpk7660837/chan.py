"""
Database - 数据库抽象层

提供统一的数据库操作接口
"""

import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CDatabase(ABC):
    """数据库抽象基类"""

    @abstractmethod
    def connect(self):
        """连接数据库"""
        pass

    @abstractmethod
    def close(self):
        """关闭连接"""
        pass

    @abstractmethod
    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        """
        执行SQL语句（INSERT/UPDATE/DELETE）

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            影响的行数
        """
        pass

    @abstractmethod
    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        查询数据

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            结果列表，每行是一个字典
        """
        pass

    @abstractmethod
    def query_one(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict]:
        """
        查询单条数据

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            结果字典或None
        """
        pass


class CSQLiteDatabase(CDatabase):
    """SQLite数据库实现"""

    def __init__(self, db_path: str):
        """
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.connect()

    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
            logger.info(f"已连接到数据库: {self.db_path}")
        except Exception as e:
            logger.error(f"连接数据库失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("数据库连接已关闭")

    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        """执行SQL语句"""
        if not self.conn:
            raise Exception("数据库未连接")

        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            self.conn.commit()
            return cursor.rowcount
        except Exception as e:
            self.conn.rollback()
            logger.error(f"执行SQL失败: {sql}, 错误: {e}")
            raise

    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        """批量执行SQL语句"""
        if not self.conn:
            raise Exception("数据库未连接")

        try:
            cursor = self.conn.cursor()
            cursor.executemany(sql, params_list)
            self.conn.commit()
            return cursor.rowcount
        except Exception as e:
            self.conn.rollback()
            logger.error(f"批量执行SQL失败: {sql}, 错误: {e}")
            raise

    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict]:
        """查询数据"""
        if not self.conn:
            raise Exception("数据库未连接")

        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"查询失败: {sql}, 错误: {e}")
            raise

    def query_one(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict]:
        """查询单条数据"""
        results = self.query(sql, params)
        return results[0] if results else None

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        sql = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        result = self.query_one(sql, (table_name,))
        return result is not None

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.close()
