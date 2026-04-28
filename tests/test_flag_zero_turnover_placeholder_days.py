import unittest

from App import flag_zero_turnover_placeholder_days as flags


class FlagZeroTurnoverPlaceholderDaysTests(unittest.TestCase):
    def test_year_windows_split_large_ranges_into_partition_safe_chunks(self):
        windows = flags.year_windows("2015-01-05", "2025-12-31")

        self.assertEqual(
            windows[:3],
            [
                ("2015-01-05", "2015-12-31"),
                ("2016-01-01", "2016-12-31"),
                ("2017-01-01", "2017-12-31"),
            ],
        )
        self.assertEqual(windows[-1], ("2025-01-01", "2025-12-31"))
        self.assertEqual(len(windows), 11)

    def test_insert_query_writes_zero_turnover_placeholder_status(self):
        query = flags.insert_placeholder_flags_query(
            level="1m",
            start_date="2015-01-05",
            end_date="2025-12-31",
            source="derived_zero_turnover_placeholder_2015_2025_v1",
        )

        self.assertIn("INSERT INTO market.bar_status_flags", query)
        self.assertIn("'zero_turnover_placeholder' AS status", query)
        self.assertIn("count() AS day_bar_count", query)
        self.assertIn("sum(volume) AS day_volume_sum", query)
        self.assertIn("uniqExact(close) AS day_close_values", query)

    def test_delete_query_scopes_source_and_status(self):
        query = flags.delete_placeholder_flags_query(
            level="1m",
            source="derived_zero_turnover_placeholder_2015_2025_v1",
        )

        self.assertIn("ALTER TABLE market.bar_status_flags DELETE", query)
        self.assertIn("level = '1m'", query)
        self.assertIn("status = 'zero_turnover_placeholder'", query)
        self.assertIn("source = 'derived_zero_turnover_placeholder_2015_2025_v1'", query)


if __name__ == "__main__":
    unittest.main()
