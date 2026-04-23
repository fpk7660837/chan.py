import sqlite3
import tempfile
import unittest
from pathlib import Path

from App import export_market_bars_to_parquet as exporter


class ExportMarketBarsToParquetTests(unittest.TestCase):
    def test_partition_path_uses_level_year_and_month_directories(self):
        path = exporter.partition_output_path(Path("/tmp/parquet"), level="1m", year_month="202401")

        self.assertEqual(path, Path("/tmp/parquet/bars/level=1m/year=2024/month=01/part.parquet"))

    def test_export_query_filters_one_month_with_trade_time_range(self):
        query = exporter.export_query(level="1m", year_month="202401")

        self.assertIn("FORMAT Parquet", query)
        self.assertIn("level = '1m'", query)
        self.assertIn("trade_time >= toDateTime64('2024-01-01 00:00:00', 3, 'Asia/Shanghai')", query)
        self.assertIn("trade_time < toDateTime64('2024-02-01 00:00:00', 3, 'Asia/Shanghai')", query)
        self.assertIn("symbol, level, trade_time, trade_date", query)

    def test_status_schema_tracks_completed_partitions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "status.sqlite3"
            conn = sqlite3.connect(db_path)
            try:
                exporter.ensure_status_schema(conn)
                self.assertFalse(exporter.is_completed(conn, level="1m", year_month="202401", expected_rows=10))
                output_path = Path(tmp_dir) / "part.parquet"
                output_path.write_bytes(b"PAR1")

                exporter.mark_completed(
                    conn,
                    level="1m",
                    year_month="202401",
                    row_count=10,
                    bytes_written=123,
                    output_path=str(output_path),
                    seconds=1.2,
                )

                self.assertTrue(exporter.is_completed(conn, level="1m", year_month="202401", expected_rows=10))
                self.assertFalse(exporter.is_completed(conn, level="1m", year_month="202401", expected_rows=11))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
