import unittest

from App import repair_bars_from_anomalies as repair


class RepairBarsFromAnomaliesTests(unittest.TestCase):
    def test_repairable_rules_include_zero_volume_positive_amount_after_threshold_tightening(self):
        self.assertIn("non_positive_price", repair.REPAIRABLE_RULES)
        self.assertIn("stock_price_gt_10000", repair.REPAIRABLE_RULES)
        self.assertIn("zero_volume_positive_amount", repair.REPAIRABLE_RULES)

    def test_delete_repairable_bars_query_targets_level_month_and_rules(self):
        query = repair.delete_repairable_bars_query(level="1m", year_month="202512")

        self.assertIn("ALTER TABLE market.bars DELETE", query)
        self.assertIn("level = '1m'", query)
        self.assertIn("toYYYYMM(trade_time) = 202512", query)
        self.assertIn("rule IN ('non_positive_price'", query)
        self.assertIn("SETTINGS mutations_sync = 2", query)

    def test_delete_repairable_anomalies_query_keeps_anomaly_table_in_sync(self):
        query = repair.delete_repairable_anomalies_query(level="1m", year_month="202512")

        self.assertIn("ALTER TABLE market.bar_anomalies DELETE", query)
        self.assertIn("level = '1m'", query)
        self.assertIn("toYYYYMM(trade_time) = 202512", query)
        self.assertIn("rule IN ('non_positive_price'", query)
        self.assertIn("SETTINGS mutations_sync = 2", query)


if __name__ == "__main__":
    unittest.main()
