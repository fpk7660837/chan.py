import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from App import sync_a_share_daily_to_sqlite as sync


class SyncAShareDailyToSQLiteTests(unittest.TestCase):
    def test_parse_args_accepts_db_path(self):
        with mock.patch(
            "sys.argv",
            ["prog", "--begin", "2020-01-01", "--db-path", "./tmp/market.sqlite3", "--universe", "hs300"],
        ):
            args = sync.parse_args()

        self.assertEqual(args.begin, "2020-01-01")
        self.assertEqual(args.db_path, "./tmp/market.sqlite3")
        self.assertEqual(args.universe, "hs300")

    def test_main_writes_rows_into_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "market.sqlite3"
            with mock.patch(
                "sys.argv",
                ["prog", "--begin", "2020-01-01", "--end", "2020-01-31", "--db-path", str(db_path), "--codes", "600519"],
            ), mock.patch.object(
                sync,
                "load_universe",
                return_value=[("600519", "贵州茅台")],
            ), mock.patch.object(
                sync,
                "_fetch_daily_qfq_bars",
                return_value=[
                    {
                        "trade_date": "2020-01-02",
                        "open": 100.0,
                        "high": 110.0,
                        "low": 99.0,
                        "close": 108.0,
                        "volume": 1000.0,
                        "amount": 105000.0,
                    }
                ],
            ):
                rc = sync.main()

            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT code, trade_date, open, close, volume, amount FROM daily_bars WHERE code = '600519'"
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(rc, 0)
        self.assertEqual(row, ("600519", "2020-01-02", 100.0, 108.0, 1000.0, 105000.0))
