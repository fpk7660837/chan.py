import unittest

from App import flag_delisting_status_windows as flags


class FlagDelistingStatusWindowsTests(unittest.TestCase):
    def test_default_windows_include_expected_statuses(self):
        statuses = {window.status for window in flags.DEFAULT_WINDOWS}

        self.assertEqual(
            statuses,
            {
                "delisting_arrangement",
                "pre_delist_below_1yuan",
                "post_delist_placeholder",
            },
        )
        self.assertEqual(len(flags.DEFAULT_WINDOWS), 13)

    def test_expand_window_rows_emits_one_row_per_trade_date(self):
        window = flags.StatusWindow(
            symbol="600804.SH",
            level="1m",
            start_date="2025-07-01",
            end_date="2025-07-02",
            status="post_delist_placeholder",
            note="sample",
        )

        rows = flags.expand_window_rows([window], source="manual_review")

        self.assertEqual(
            rows,
            [
                ("600804.SH", "1m", "2025-07-01", "post_delist_placeholder", "sample", "manual_review"),
                ("600804.SH", "1m", "2025-07-02", "post_delist_placeholder", "sample", "manual_review"),
            ],
        )

    def test_expand_window_rows_can_limit_to_existing_trade_dates(self):
        window = flags.StatusWindow(
            symbol="600804.SH",
            level="1m",
            start_date="2025-07-01",
            end_date="2025-07-03",
            status="post_delist_placeholder",
            note="sample",
        )

        rows = flags.expand_window_rows(
            [window],
            source="manual_review",
            allowed_dates={("600804.SH", "2025-07-01"), ("600804.SH", "2025-07-03")},
        )

        self.assertEqual(
            rows,
            [
                ("600804.SH", "1m", "2025-07-01", "post_delist_placeholder", "sample", "manual_review"),
                ("600804.SH", "1m", "2025-07-03", "post_delist_placeholder", "sample", "manual_review"),
            ],
        )

    def test_schema_sql_creates_bar_status_flags_table(self):
        sql = flags.schema_sql()

        self.assertIn("CREATE TABLE IF NOT EXISTS market.bar_status_flags", sql)
        self.assertIn("status LowCardinality(String)", sql)
        self.assertIn("note String", sql)
        self.assertIn("ENGINE = ReplacingMergeTree(updated_at)", sql)
        self.assertIn("ORDER BY (level, symbol, trade_date, status)", sql)

    def test_delete_existing_flags_query_scopes_source_and_level(self):
        query = flags.delete_existing_flags_query(level="1m", source="manual_review_2026_04_28")

        self.assertIn("ALTER TABLE market.bar_status_flags DELETE", query)
        self.assertIn("level = '1m'", query)
        self.assertIn("source = 'manual_review_2026_04_28'", query)
        self.assertIn("mutations_sync = 2", query)

    def test_insert_flags_query_contains_values_for_each_row(self):
        rows = [
            ("600804.SH", "1m", "2025-06-10", "delisting_arrangement", "delisting window", "manual_review"),
            ("600804.SH", "1m", "2025-07-01", "post_delist_placeholder", "post delist placeholder", "manual_review"),
        ]

        query = flags.insert_flags_query(rows)

        self.assertIn("INSERT INTO market.bar_status_flags", query)
        self.assertIn("'600804.SH'", query)
        self.assertIn("'2025-06-10'", query)
        self.assertIn("'delisting_arrangement'", query)
        self.assertIn("'post_delist_placeholder'", query)


if __name__ == "__main__":
    unittest.main()
