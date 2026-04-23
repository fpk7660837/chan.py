import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from App import import_stock_bar_1m_raw_to_clickhouse as importer


def _write_zip_csv(path: Path, member: str, *, code: str = "sh600000", name: str = "浦发银行"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "时间,代码,名称,开盘价,收盘价,最高价,最低价,成交量,成交额,涨幅,振幅",
        f"2024-01-02 09:30:00,{code},{name},6.63,6.62,6.64,6.61,1530,1014390,0.0,0.0",
        f"2024-01-02 09:31:00,{code},{name},6.62,6.65,6.66,6.62,7192,4768371,0.0,0.0",
    ]
    mode = "a" if path.exists() else "w"
    with zipfile.ZipFile(path, mode, compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, "\ufeff" + "\n".join(rows) + "\n")


class ImportStockBar1mRawToClickHouseTests(unittest.TestCase):
    def test_selects_only_ok_raw_1m_archives(self):
        rows = [
            {"archive_path": "/tmp/raw_1m.zip", "level": "1m", "adjust": "raw", "status": "ok", "year_start": "2024"},
            {"archive_path": "/tmp/qfq_1m.zip", "level": "1m", "adjust": "qfq", "status": "ok", "year_start": "2024"},
            {"archive_path": "/tmp/raw_30m.zip", "level": "30m", "adjust": "raw", "status": "ok", "year_start": "2024"},
            {"archive_path": "/tmp/bad.zip", "level": "1m", "adjust": "raw", "status": "error", "year_start": "2024"},
        ]

        selected = importer.select_raw_1m_archives(rows)

        self.assertEqual([row["archive_path"] for row in selected], ["/tmp/raw_1m.zip"])

    def test_iter_member_bars_maps_chinese_csv_to_clickhouse_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "2024_1min.zip"
            _write_zip_csv(archive_path, "sh600000_2024.csv")

            bars = list(importer.iter_member_bars(archive_path, "sh600000_2024.csv", source="stock_bar_zip"))

        self.assertEqual(len(bars), 2)
        self.assertEqual(
            bars[0],
            {
                "symbol": "600000.SH",
                "level": "1m",
                "trade_time": "2024-01-02 09:30:00",
                "open": "6.63",
                "high": "6.64",
                "low": "6.61",
                "close": "6.62",
                "volume": "1530",
                "amount": "1014390",
                "source": "stock_bar_zip",
            },
        )

    def test_iter_archive_bars_streams_all_members_for_bulk_import(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "2024_1min.zip"
            _write_zip_csv(archive_path, "sh600000_2024.csv")
            _write_zip_csv(archive_path, "sz000001_2024.csv", code="sz000001", name="平安银行")

            bars = list(importer.iter_archive_bars(archive_path, source="bulk"))
            limited = list(importer.iter_archive_bars(archive_path, source="bulk", member_limit=1))

        self.assertEqual(len(bars), 4)
        self.assertEqual([bar["symbol"] for bar in bars], ["600000.SH", "600000.SH", "000001.SZ", "000001.SZ"])
        self.assertEqual({bar["source"] for bar in bars}, {"bulk"})
        self.assertEqual(len(limited), 2)

    def test_member_identity_extracts_symbol_level_and_year_for_idempotent_replace(self):
        identity = importer.member_identity("/tmp/2024_1min.zip", "sh600000_2024.csv")

        self.assertEqual(identity, {"symbol": "600000.SH", "level": "1m", "year": "2024"})

    def test_clickhouse_insert_command_filters_existing_symbol_year_level_keys(self):
        command = importer.clickhouse_insert_command(
            Path("/tmp/.env"),
            Path("/tmp/docker-compose.yml"),
            symbol="600000.SH",
            level="1m",
            year="2024",
        )

        query = command[-1]
        self.assertIn("INSERT INTO market.bars", query)
        self.assertIn("FROM input(", query)
        self.assertIn("NOT IN", query)
        self.assertIn("symbol = '600000.SH'", query)
        self.assertIn("level = '1m'", query)
        self.assertIn("trade_time >= toDateTime64('2024-01-01 00:00:00', 3, 'Asia/Shanghai')", query)
        self.assertIn("trade_time < toDateTime64('2025-01-01 00:00:00', 3, 'Asia/Shanghai')", query)

    def test_clickhouse_bulk_insert_command_uses_single_plain_csv_stream(self):
        command = importer.clickhouse_bulk_insert_command(Path("/tmp/.env"), Path("/tmp/docker-compose.yml"))

        query = command[-1]
        self.assertIn("INSERT INTO market.bars", query)
        self.assertIn("FORMAT CSV", query)
        self.assertNotIn("FROM input(", query)
        self.assertNotIn("NOT IN", query)

    def test_status_db_tracks_completed_members(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "status.sqlite3"
            conn = sqlite3.connect(db_path)
            try:
                importer.ensure_status_schema(conn)
                self.assertFalse(importer.is_completed(conn, "/tmp/2024_1min.zip", "sh600000_2024.csv"))

                importer.mark_completed(
                    conn,
                    archive_path="/tmp/2024_1min.zip",
                    member="sh600000_2024.csv",
                    row_count=2,
                    year="2024",
                    level="1m",
                )

                self.assertTrue(importer.is_completed(conn, "/tmp/2024_1min.zip", "sh600000_2024.csv"))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
