import sqlite3
import socket
import tempfile
import unittest
from pathlib import Path

from App import import_tushare_adj_factors as importer


class ImportTushareAdjFactorsTests(unittest.TestCase):
    def test_normalizes_tushare_rows_to_clickhouse_values(self):
        rows = importer.normalize_adj_factor_rows(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102", "adj_factor": 123.45},
                {"ts_code": "600000.SH", "trade_date": "20240102", "adj_factor": "98.7"},
            ]
        )

        self.assertEqual(
            rows,
            [
                ("000001.SZ", "2024-01-02", 123.45, "tushare"),
                ("600000.SH", "2024-01-02", 98.7, "tushare"),
            ],
        )

    def test_rejects_bad_tushare_response_code(self):
        with self.assertRaisesRegex(RuntimeError, "bad token"):
            importer.parse_tushare_response({"code": 2002, "msg": "bad token"})

    def test_parse_tushare_response_maps_fields_to_dicts(self):
        rows = importer.parse_tushare_response(
            {
                "code": 0,
                "msg": "",
                "data": {
                    "fields": ["ts_code", "trade_date", "adj_factor"],
                    "items": [["000001.SZ", "20240102", 123.45]],
                },
            }
        )

        self.assertEqual(rows, [{"ts_code": "000001.SZ", "trade_date": "20240102", "adj_factor": 123.45}])

    def test_status_db_tracks_completed_trade_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            conn = sqlite3.connect(Path(tmp_dir) / "status.sqlite3")
            try:
                importer.ensure_status_schema(conn)
                self.assertFalse(importer.is_completed(conn, "20240102"))

                importer.mark_completed(conn, trade_date="20240102", row_count=2)

                self.assertTrue(importer.is_completed(conn, "20240102"))
            finally:
                conn.close()

    def test_insert_command_targets_adj_factors_csv(self):
        command = importer.clickhouse_insert_command(Path("/tmp/.env"), Path("/tmp/docker-compose.yml"))

        query = command[-1]
        self.assertIn("INSERT INTO market.adj_factors", query)
        self.assertIn("FORMAT CSV", query)
        self.assertIn("symbol, trade_date, adj_factor, source", query)
        self.assertIn("NOT IN", query)

    def test_fetch_retries_socket_timeout(self):
        calls = []
        original_request = importer.request_tushare_json

        def fake_request(**kwargs):
            calls.append(kwargs["trade_date"])
            if len(calls) == 1:
                raise socket.timeout("timed out")
            return {
                "code": 0,
                "msg": "",
                "data": {
                    "fields": ["ts_code", "trade_date", "adj_factor"],
                    "items": [["000001.SZ", "20240102", 123.45]],
                },
            }

        importer.request_tushare_json = fake_request
        try:
            rows = importer.fetch_adj_factor(
                token="not-a-real-token",
                trade_date="20240102",
                retries=2,
                retry_sleep=0,
            )
        finally:
            importer.request_tushare_json = original_request

        self.assertEqual(len(calls), 2)
        self.assertEqual(rows[0]["ts_code"], "000001.SZ")

    def test_parse_args_accepts_workers(self):
        args = importer.parse_args(["--workers", "4"])

        self.assertEqual(args.workers, 4)

    def test_parse_args_accepts_max_requests_per_minute(self):
        args = importer.parse_args(["--max-requests-per-minute", "180"])

        self.assertEqual(args.max_requests_per_minute, 180)

    def test_fetch_normalized_waits_on_pacer_before_request(self):
        calls = []
        original_fetch = importer.fetch_adj_factor

        class FakePacer:
            def wait(self):
                calls.append("wait")

        def fake_fetch(**kwargs):
            calls.append(kwargs["trade_date"])
            return [{"ts_code": "000001.SZ", "trade_date": kwargs["trade_date"], "adj_factor": 123.45}]

        importer.fetch_adj_factor = fake_fetch
        try:
            trade_date, rows = importer.fetch_normalized_adj_factor(
                token="not-a-real-token",
                trade_date="20240102",
                api_url="http://example.invalid",
                timeout=1,
                retries=1,
                retry_sleep=0,
                source="tushare",
                pacer=FakePacer(),
            )
        finally:
            importer.fetch_adj_factor = original_fetch

        self.assertEqual(calls, ["wait", "20240102"])
        self.assertEqual(trade_date, "20240102")
        self.assertEqual(rows, [("000001.SZ", "2024-01-02", 123.45, "tushare")])


if __name__ == "__main__":
    unittest.main()
