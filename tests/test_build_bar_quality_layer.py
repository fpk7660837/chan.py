import unittest
from pathlib import Path

from App import build_bar_quality_layer as quality


class BuildBarQualityLayerTests(unittest.TestCase):
    def test_month_bounds_returns_clickhouse_time_range(self):
        self.assertEqual(
            quality.month_bounds("202401"),
            ("2024-01-01 00:00:00", "2024-02-01 00:00:00"),
        )
        self.assertEqual(
            quality.month_bounds("202412"),
            ("2024-12-01 00:00:00", "2025-01-01 00:00:00"),
        )

    def test_insert_anomalies_query_uses_month_range_and_rules(self):
        query = quality.insert_anomalies_query(level="1m", year_month="202401")

        self.assertIn("INSERT INTO market.bar_anomalies", query)
        self.assertIn("trade_time >= toDateTime64('2024-01-01 00:00:00', 3, 'Asia/Shanghai')", query)
        self.assertIn("trade_time < toDateTime64('2024-02-01 00:00:00', 3, 'Asia/Shanghai')", query)
        self.assertIn("arrayJoin(rules) AS rule", query)
        self.assertIn("non_positive_price", query)
        self.assertIn("bad_ohlc", query)
        self.assertIn("stock_price_gt_10000", query)
        self.assertIn("amount_price_volume_mismatch", query)

    def test_schema_sql_creates_anomaly_table_and_clean_view(self):
        sql = quality.schema_sql()

        self.assertIn("CREATE TABLE IF NOT EXISTS market.bar_anomalies", sql)
        self.assertIn("ENGINE = ReplacingMergeTree(updated_at)", sql)
        self.assertIn("ORDER BY (level, symbol, trade_time, rule)", sql)
        self.assertIn("CREATE OR REPLACE VIEW market.bars_1m_clean", sql)
        self.assertIn("NOT IN", sql)

    def test_report_paths_are_under_quality_report_dir(self):
        paths = quality.report_paths(Path("/tmp/reports"))

        self.assertEqual(paths["summary"].name, "bars_1m_quality_summary.json")
        self.assertEqual(paths["by_rule"].name, "bars_1m_anomalies_by_rule.csv")
        self.assertEqual(paths["samples"].name, "bars_1m_anomaly_samples.csv")


if __name__ == "__main__":
    unittest.main()
