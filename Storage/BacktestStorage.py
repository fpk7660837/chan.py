"""
BacktestStorage - 回测结果存储

将回测结果持久化到数据库
"""

import json
from datetime import datetime
from typing import Optional, List, Dict
import logging

from Storage.Database import CDatabase

logger = logging.getLogger(__name__)


class CBacktestStorage:
    """回测结果存储"""

    def __init__(self, db: CDatabase):
        """
        Args:
            db: 数据库对象
        """
        self.db = db
        self._create_tables()

    def _create_tables(self):
        """创建表结构"""

        # 回测运行记录表
        sql_runs = """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            initial_capital REAL NOT NULL,
            final_value REAL,
            total_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            trade_count INTEGER,
            win_rate REAL,
            created_at TEXT NOT NULL,
            config TEXT
        )
        """

        # 交易记录表
        sql_trades = """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            direction TEXT NOT NULL,
            volume INTEGER NOT NULL,
            price REAL NOT NULL,
            time TEXT NOT NULL,
            commission REAL,
            tax REAL,
            slippage REAL,
            reason TEXT,
            profit REAL,
            profit_rate REAL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
        )
        """

        # 权益曲线表
        sql_equity = """
        CREATE TABLE IF NOT EXISTS backtest_equity (
            equity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            time TEXT NOT NULL,
            total_value REAL NOT NULL,
            positions_value REAL NOT NULL,
            cash REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
        )
        """

        self.db.execute(sql_runs)
        self.db.execute(sql_trades)
        self.db.execute(sql_equity)

        logger.info("回测存储表已创建")

    def save_backtest(self, result) -> int:
        """
        保存回测结果

        Args:
            result: CBacktestResult对象

        Returns:
            run_id
        """
        # 保存回测运行记录
        sql_run = """
        INSERT INTO backtest_runs (
            strategy_name, start_time, end_time, initial_capital,
            final_value, total_return, max_drawdown, sharpe_ratio,
            trade_count, win_rate, created_at, config
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        metrics = result.metrics
        params = (
            result.strategy_name,
            result.start_time,
            result.end_time,
            result.initial_capital,
            metrics.get('final_value'),
            metrics.get('total_return'),
            metrics.get('max_drawdown'),
            metrics.get('sharpe_ratio'),
            metrics.get('trade_count'),
            metrics.get('win_rate'),
            datetime.now().isoformat(),
            json.dumps(metrics)  # 存储完整的metrics
        )

        self.db.execute(sql_run, params)

        # 获取run_id
        run_id = self.db.query_one("SELECT last_insert_rowid() as id")['id']

        # 保存交易记录
        if result.trades:
            sql_trade = """
            INSERT INTO backtest_trades (
                run_id, code, direction, volume, price, time,
                commission, tax, slippage, reason, profit, profit_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            trade_params = []
            for trade in result.trades:
                trade_params.append((
                    run_id,
                    trade.code,
                    trade.direction,
                    trade.volume,
                    trade.price,
                    str(trade.time),
                    trade.commission,
                    trade.tax,
                    trade.slippage,
                    trade.reason,
                    trade.profit,
                    trade.profit_rate
                ))

            self.db.execute_many(sql_trade, trade_params)

        # 保存权益曲线
        if result.equity_curve:
            sql_equity = """
            INSERT INTO backtest_equity (
                run_id, time, total_value, positions_value, cash
            ) VALUES (?, ?, ?, ?, ?)
            """

            equity_params = []
            for point in result.equity_curve:
                time, total_value, positions_value, cash = point
                equity_params.append((
                    run_id,
                    str(time),
                    total_value,
                    positions_value,
                    cash
                ))

            self.db.execute_many(sql_equity, equity_params)

        logger.info(f"回测结果已保存，run_id={run_id}")
        return run_id

    def load_backtest(self, run_id: int) -> Optional[Dict]:
        """
        加载回测结果

        Args:
            run_id: 回测运行ID

        Returns:
            回测结果字典
        """
        # 加载回测记录
        sql_run = "SELECT * FROM backtest_runs WHERE run_id=?"
        run_data = self.db.query_one(sql_run, (run_id,))

        if not run_data:
            return None

        # 加载交易记录
        sql_trades = "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY time"
        trades = self.db.query(sql_trades, (run_id,))

        # 加载权益曲线
        sql_equity = "SELECT * FROM backtest_equity WHERE run_id=? ORDER BY time"
        equity = self.db.query(sql_equity, (run_id,))

        return {
            'run': run_data,
            'trades': trades,
            'equity': equity
        }

    def list_backtests(self, limit: int = 100) -> List[Dict]:
        """
        列出历史回测

        Args:
            limit: 返回记录数

        Returns:
            回测列表
        """
        sql = """
        SELECT run_id, strategy_name, start_time, end_time,
               initial_capital, final_value, total_return, max_drawdown,
               trade_count, created_at
        FROM backtest_runs
        ORDER BY created_at DESC
        LIMIT ?
        """

        return self.db.query(sql, (limit,))

    def delete_backtest(self, run_id: int):
        """
        删除回测结果

        Args:
            run_id: 回测运行ID
        """
        self.db.execute("DELETE FROM backtest_equity WHERE run_id=?", (run_id,))
        self.db.execute("DELETE FROM backtest_trades WHERE run_id=?", (run_id,))
        self.db.execute("DELETE FROM backtest_runs WHERE run_id=?", (run_id,))

        logger.info(f"已删除回测结果，run_id={run_id}")
