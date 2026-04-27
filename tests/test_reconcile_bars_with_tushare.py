import urllib.error
import unittest
from unittest import mock

from App import reconcile_bars_with_tushare as reconcile


class ReconcileBarsWithTushareTests(unittest.TestCase):
    def test_year_month_bounds_cover_calendar_month(self):
        start_date, end_date, start_time, end_time = reconcile.year_month_bounds("202506")

        self.assertEqual(start_date, "2025-06-01")
        self.assertEqual(end_date, "2025-06-30")
        self.assertEqual(start_time, "2025-06-01 09:00:00")
        self.assertEqual(end_time, "2025-06-30 15:30:00")

    def test_parse_tushare_minute_response_maps_fields_to_dicts(self):
        rows = reconcile.parse_tushare_minute_response(
            {
                "code": 0,
                "msg": "",
                "data": {
                    "fields": ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"],
                    "items": [["000001.SZ", "2025-06-03 09:30:00", 11.54, 11.53, 11.56, 11.53, 1735300, 20031256]],
                },
            }
        )

        self.assertEqual(
            rows,
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_time": "2025-06-03 09:30:00",
                    "open": 11.54,
                    "close": 11.53,
                    "high": 11.56,
                    "low": 11.53,
                    "vol": 1735300,
                    "amount": 20031256,
                }
            ],
        )

    @mock.patch("App.reconcile_bars_with_tushare.time.sleep")
    @mock.patch("App.reconcile_bars_with_tushare.urllib.request.urlopen")
    def test_request_tushare_minute_json_retries_on_transient_http_error(self, urlopen_mock, sleep_mock):
        success_response = mock.MagicMock()
        success_response.__enter__.return_value.read.return_value = b'{"code": 0, "data": {"fields": [], "items": []}}'
        urlopen_mock.side_effect = [
            urllib.error.HTTPError(
                url="http://api.tushare.pro",
                code=502,
                msg="Bad Gateway",
                hdrs=None,
                fp=None,
            ),
            success_response,
        ]

        payload = reconcile.request_tushare_minute_json(
            api_url="http://api.tushare.pro",
            token="token",
            symbol="000001.SZ",
            start_time="2025-06-01 09:00:00",
            end_time="2025-06-30 15:30:00",
            timeout=10.0,
            retries=1,
            retry_backoff=0.0,
        )

        self.assertEqual(payload["code"], 0)
        self.assertEqual(urlopen_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.0)

    def test_normalize_tushare_minute_rows_sorts_ascending(self):
        rows = reconcile.normalize_tushare_minute_rows(
            [
                {
                    "ts_code": "000001.sz",
                    "trade_time": "2025-06-03 09:31:00",
                    "open": 10.8,
                    "close": 12,
                    "high": 12,
                    "low": 10.8,
                    "vol": 2000,
                    "amount": 22000,
                },
                {
                    "ts_code": "000001.sz",
                    "trade_time": "2025-06-03 09:30:00",
                    "open": 10,
                    "close": 11,
                    "high": 11,
                    "low": 10,
                    "vol": 1000,
                    "amount": 10500,
                },
            ]
        )

        self.assertEqual([row["trade_time"] for row in rows], ["2025-06-03 09:30:00", "2025-06-03 09:31:00"])
        self.assertEqual(rows[0]["symbol"], "000001.SZ")

    def test_summarize_day_uses_first_open_last_close_and_volume_multiplier(self):
        rows = [
            {"trade_time": "2025-06-03 09:30:00", "open": 10.0, "close": 11.0, "high": 11.0, "low": 10.0, "volume": 10, "amount": 10500},
            {"trade_time": "2025-06-03 09:31:00", "open": 11.0, "close": 12.0, "high": 12.0, "low": 10.8, "volume": 20, "amount": 22000},
            {"trade_time": "2025-06-03 09:32:00", "open": 12.0, "close": 11.5, "high": 12.0, "low": 11.4, "volume": 30, "amount": 34500},
        ]

        summary = reconcile.summarize_day(rows, volume_multiplier=100.0)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["open"], 10.0)
        self.assertEqual(summary["close"], 11.5)
        self.assertEqual(summary["high"], 12.0)
        self.assertEqual(summary["low"], 10.0)
        self.assertEqual(summary["volume"], 6000.0)
        self.assertEqual(summary["amount"], 67000.0)

    def test_analyze_minute_alignment_detects_previous_close_shift(self):
        tushare_rows = [
            {"trade_time": "2025-06-03 09:30:00", "open": 10.0, "close": 11.0, "high": 11.0, "low": 10.0, "volume": 1000.0, "amount": 10500.0},
            {"trade_time": "2025-06-03 09:31:00", "open": 10.8, "close": 12.0, "high": 12.0, "low": 10.8, "volume": 2000.0, "amount": 22000.0},
            {"trade_time": "2025-06-03 09:32:00", "open": 11.7, "close": 11.5, "high": 12.0, "low": 11.4, "volume": 3000.0, "amount": 34500.0},
        ]
        clickhouse_rows = [
            {"trade_time": "2025-06-03 09:30:00", "open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0, "volume": 10.0, "amount": 10500.0},
            {"trade_time": "2025-06-03 09:31:00", "open": 11.0, "close": 11.9, "high": 12.0, "low": 10.8, "volume": 20.0, "amount": 22000.0},
            {"trade_time": "2025-06-03 09:32:00", "open": 12.0, "close": 11.5, "high": 12.0, "low": 11.4, "volume": 30.0, "amount": 34500.0},
        ]

        diagnostics = reconcile.analyze_minute_alignment(clickhouse_rows, tushare_rows)

        self.assertEqual(diagnostics["common_rows"], 3)
        self.assertEqual(diagnostics["same_timestamp_open_matches"], 1)
        self.assertEqual(diagnostics["previous_close_open_matches"], 2)
        self.assertEqual(diagnostics["same_timestamp_close_matches"], 1)
        self.assertEqual(diagnostics["same_timestamp_high_matches"], 2)
        self.assertEqual(diagnostics["same_timestamp_low_matches"], 3)

    def test_group_rows_by_trade_date_groups_and_sorts_rows(self):
        grouped = reconcile.group_rows_by_trade_date(
            [
                {"trade_time": "2025-06-04 09:31:00", "open": 10.2},
                {"trade_time": "2025-06-03 09:31:00", "open": 10.1},
                {"trade_time": "2025-06-03 09:30:00", "open": 10.0},
            ]
        )

        self.assertEqual(list(grouped), ["2025-06-03", "2025-06-04"])
        self.assertEqual(
            [row["trade_time"] for row in grouped["2025-06-03"]],
            ["2025-06-03 09:30:00", "2025-06-03 09:31:00"],
        )

    def test_build_day_result_classifies_tushare_zero_volume_placeholder(self):
        tushare_rows = [
            {
                "trade_time": "2025-06-04 09:30:00",
                "open": 7.37,
                "close": 7.37,
                "high": 7.37,
                "low": 7.37,
                "volume": 0.0,
                "amount": 0.0,
            },
            {
                "trade_time": "2025-06-04 09:31:00",
                "open": 7.37,
                "close": 7.37,
                "high": 7.37,
                "low": 7.37,
                "volume": 0.0,
                "amount": 0.0,
            },
        ]

        result = reconcile.build_day_result(
            symbol="000506.SZ",
            trade_date="2025-06-04",
            clickhouse_rows=[],
            tushare_rows=tushare_rows,
        )

        self.assertEqual(result["status"], "tushare_zero_volume_placeholder")
        self.assertEqual(result["clickhouse_rows"], 0)
        self.assertEqual(result["tushare_rows"], 2)

    def test_summarize_results_aggregates_matches_and_missing_days(self):
        summary = reconcile.summarize_results(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": "2025-06-03",
                    "status": "ok",
                    "clickhouse_rows": 241,
                    "tushare_rows": 241,
                    "diff": {
                        "open": 0.0,
                        "close": 0.0,
                        "high": 0.01,
                        "low": -0.01,
                        "volume_shares": 48.0,
                        "amount": 1024.0,
                    },
                    "tushare_day": {"amount": 2000000.0},
                    "minute_alignment": {
                        "common_rows": 241,
                        "same_timestamp_open_matches": 200,
                        "previous_close_open_matches": 30,
                        "same_timestamp_close_matches": 241,
                        "same_timestamp_high_matches": 240,
                        "same_timestamp_low_matches": 240,
                    },
                },
                {
                    "symbol": "000002.SZ",
                    "trade_date": "2025-06-04",
                    "status": "missing_in_tushare",
                    "clickhouse_rows": 241,
                    "tushare_rows": 0,
                },
            ]
        )

        self.assertEqual(summary["symbol_day_count"], 2)
        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["missing_in_tushare_count"], 1)
        self.assertEqual(summary["row_match_count"], 1)
        self.assertEqual(summary["day_open_match_count"], 1)
        self.assertEqual(summary["day_close_match_count"], 1)
        self.assertEqual(summary["day_high_match_le_0_01_count"], 1)
        self.assertEqual(summary["day_low_match_le_0_01_count"], 1)
        self.assertEqual(summary["max_abs_volume_shares_diff"], 48.0)
        self.assertAlmostEqual(summary["max_abs_amount_diff_bps"], 5.12, places=6)
        self.assertAlmostEqual(summary["avg_same_ts_open_match_rate"], 200 / 241, places=6)
        self.assertAlmostEqual(summary["avg_prev_close_open_match_rate"], 30 / 240, places=6)


if __name__ == "__main__":
    unittest.main()
