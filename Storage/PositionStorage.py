"""
PositionStorage - 持仓状态存储

存储和管理策略持仓状态
"""

from typing import Dict, Optional
import logging
from datetime import datetime

from Storage.Database import CDatabase
from Backtest.Position import CPosition

logger = logging.getLogger(__name__)


class CPositionStorage:
    """持仓状态存储"""

    def __init__(self, db: CDatabase):
        """
        Args:
            db: 数据库对象
        """
        self.db = db
        self._create_tables()

    def _create_tables(self):
        """创建表结构"""

        sql = """
        CREATE TABLE IF NOT EXISTS positions (
            code TEXT PRIMARY KEY,
            volume INTEGER NOT NULL,
            available INTEGER NOT NULL,
            cost_price REAL NOT NULL,
            current_price REAL NOT NULL,
            market_value REAL NOT NULL,
            cost_value REAL NOT NULL,
            profit REAL NOT NULL,
            profit_rate REAL NOT NULL,
            hold_days INTEGER NOT NULL,
            first_buy_time TEXT,
            updated_at TEXT NOT NULL
        )
        """

        self.db.execute(sql)
        logger.info("持仓存储表已创建")

    def save_position(self, position: CPosition):
        """
        保存持仓

        Args:
            position: CPosition对象
        """
        sql = """
        INSERT OR REPLACE INTO positions (
            code, volume, available, cost_price, current_price,
            market_value, cost_value, profit, profit_rate, hold_days,
            first_buy_time, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            position.code,
            position.volume,
            position.available,
            position.cost_price,
            position.current_price,
            position.market_value,
            position.cost_value,
            position.profit,
            position.profit_rate,
            position.hold_days,
            str(position.first_buy_time) if position.first_buy_time else None,
            datetime.now().isoformat()
        )

        self.db.execute(sql, params)
        logger.debug(f"已保存持仓: {position.code}")

    def get_position(self, code: str) -> Optional[CPosition]:
        """
        获取持仓

        Args:
            code: 股票代码

        Returns:
            CPosition对象或None
        """
        sql = "SELECT * FROM positions WHERE code=?"
        row = self.db.query_one(sql, (code,))

        if not row:
            return None

        # 从数据库记录重建CPosition对象
        position = CPosition(row['code'])
        position.volume = row['volume']
        position.available = row['available']
        position.cost_price = row['cost_price']
        position.current_price = row['current_price']
        position.market_value = row['market_value']
        position.cost_value = row['cost_value']
        position.profit = row['profit']
        position.profit_rate = row['profit_rate']
        position.hold_days = row['hold_days']

        # first_buy_time需要特殊处理
        if row['first_buy_time']:
            from Common.CTime import CTime
            # 简单解析，假设格式为 "YYYY-MM-DD HH:MM"
            time_str = row['first_buy_time']
            # 这里需要根据实际CTime的字符串格式进行解析
            # 暂时设为None
            position.first_buy_time = None

        return position

    def get_all_positions(self) -> Dict[str, CPosition]:
        """
        获取所有持仓

        Returns:
            {code: CPosition对象}
        """
        sql = "SELECT * FROM positions WHERE volume > 0"
        rows = self.db.query(sql)

        positions = {}
        for row in rows:
            position = CPosition(row['code'])
            position.volume = row['volume']
            position.available = row['available']
            position.cost_price = row['cost_price']
            position.current_price = row['current_price']
            position.market_value = row['market_value']
            position.cost_value = row['cost_value']
            position.profit = row['profit']
            position.profit_rate = row['profit_rate']
            position.hold_days = row['hold_days']

            positions[row['code']] = position

        return positions

    def delete_position(self, code: str):
        """
        删除持仓

        Args:
            code: 股票代码
        """
        sql = "DELETE FROM positions WHERE code=?"
        self.db.execute(sql, (code,))
        logger.info(f"已删除持仓: {code}")

    def clear_all_positions(self):
        """清空所有持仓"""
        sql = "DELETE FROM positions"
        self.db.execute(sql)
        logger.info("已清空所有持仓")

    def update_positions_price(self, prices: Dict[str, float]):
        """
        批量更新持仓价格

        Args:
            prices: {code: price}
        """
        for code, price in prices.items():
            position = self.get_position(code)
            if position and position.volume > 0:
                position.update_price(price)
                self.save_position(position)
