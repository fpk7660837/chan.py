import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Common.CEnum import AUTYPE, KL_TYPE
from Common.ChanException import CChanException
from DataAPI.SQLiteDailyBarAPI import (
    LOCAL_DB_ENV_VAR,
    CSQLiteDailyBarAPI,
    ensure_daily_bar_schema,
    resolve_local_db_path,
    upsert_daily_bars,
)


class SQLiteDailyBarApiTests(unittest.TestCase):
    def test_resolve_local_db_path_prefers_env_var(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "market.sqlite3"
            with mock.patch.dict(os.environ, {LOCAL_DB_ENV_VAR: str(db_path)}, clear=False):
                self.assertEqual(resolve_local_db_path(), db_path.resolve())

    def test_sqlite_daily_bar_api_reads_qfq_daily_bars(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "market.sqlite3"
            conn = sqlite3.connect(db_path)
            ensure_daily_bar_schema(conn)
            upsert_daily_bars(
                conn,
                code="600519",
                rows=[
                    {
                        "trade_date": "2024-01-02",
                        "open": 100.0,
                        "high": 110.0,
                        "low": 99.0,
                        "close": 108.0,
                        "volume": 1000.0,
                        "amount": 105000.0,
                    },
                    {
                        "trade_date": "2024-01-03",
                        "open": 108.0,
                        "high": 112.0,
                        "low": 107.0,
                        "close": 111.0,
                        "volume": 900.0,
                        "amount": 100000.0,
                    },
                ],
            )
            conn.close()

            with mock.patch.dict(os.environ, {LOCAL_DB_ENV_VAR: str(db_path)}, clear=False):
                api = CSQLiteDailyBarAPI(
                    code="600519",
                    k_type=KL_TYPE.K_DAY,
                    begin_date="2024-01-01",
                    end_date="2024-01-31",
                    autype=AUTYPE.QFQ,
                )
                bars = list(api.get_kl_data())

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].time.toDateStr("-"), "2024-01-02")
        self.assertEqual(bars[0].open, 100.0)
        self.assertEqual(bars[0].close, 108.0)
        self.assertEqual(bars[0].trade_info.metric["volume"], 1000.0)
        self.assertEqual(bars[0].trade_info.metric["turnover"], 105000.0)

    def test_sqlite_daily_bar_api_rejects_non_daily_k_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "market.sqlite3"
            with mock.patch.dict(os.environ, {LOCAL_DB_ENV_VAR: str(db_path)}, clear=False):
                api = CSQLiteDailyBarAPI(
                    code="600519",
                    k_type=KL_TYPE.K_WEEK,
                    begin_date="2024-01-01",
                    end_date="2024-01-31",
                    autype=AUTYPE.QFQ,
                )
                with self.assertRaisesRegex(CChanException, "only supports K_DAY"):
                    list(api.get_kl_data())
