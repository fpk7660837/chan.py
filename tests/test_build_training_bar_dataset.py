import unittest

from App import build_training_bar_dataset as dataset


class BuildTrainingBarDatasetTests(unittest.TestCase):
    def test_dataset_filter_query_excludes_anomalies_and_placeholder_days(self):
        query = dataset.dataset_filter_query(
            level="1m",
            start_date="2015-01-05",
            end_date="2024-12-31",
            include_delisting_arrangement=False,
            include_pre_delist_below_1yuan=False,
        )

        self.assertIn("FROM market.bars AS b", query)
        self.assertIn("market.bar_status_flags", query)
        self.assertIn("market.bar_anomalies", query)
        self.assertIn("NOT IN", query)
        self.assertIn("'zero_turnover_placeholder'", query)
        self.assertNotIn("WITH placeholder_days AS", query)

    def test_dataset_filter_query_can_keep_delisting_arrangement_but_drop_pre_delist(self):
        query = dataset.dataset_filter_query(
            level="1m",
            start_date="2015-01-05",
            end_date="2025-12-31",
            include_delisting_arrangement=True,
            include_pre_delist_below_1yuan=False,
        )

        self.assertIn("'post_delist_placeholder'", query)
        self.assertIn("'pre_delist_below_1yuan'", query)
        self.assertIn("'zero_turnover_placeholder'", query)
        self.assertNotIn("'delisting_arrangement', 'pre_delist_below_1yuan'", query)

    def test_profile_query_counts_rows_symbols_and_flagged_days(self):
        query = dataset.profile_query(
            level="1m",
            start_date="2015-01-05",
            end_date="2024-12-31",
            include_delisting_arrangement=False,
            include_pre_delist_below_1yuan=False,
        )

        self.assertIn("count() AS kept_rows", query)
        self.assertIn("kept_symbols", query)
        self.assertIn("kept_trade_dates", query)
        self.assertIn("excluded_placeholder_days", query)
        self.assertIn("excluded_status_flag_days", query)
        self.assertIn("zero_turnover_placeholder", query)

    def test_create_view_sql_uses_expected_balanced_view_name(self):
        sql = dataset.create_view_sql(
            view_name="market.bars_1m_train_balanced",
            level="1m",
            start_date="2015-01-05",
            end_date="2024-12-31",
            include_delisting_arrangement=False,
            include_pre_delist_below_1yuan=False,
        )

        self.assertIn("CREATE OR REPLACE VIEW market.bars_1m_train_balanced AS", sql)
        self.assertIn("FROM market.bars AS b", sql)
        self.assertIn("market.bar_status_flags", sql)

    def test_default_view_configs_include_three_named_profiles(self):
        configs = dataset.default_view_configs()

        self.assertEqual(
            set(configs),
            {
                "market.bars_1m_train_balanced",
                "market.bars_1m_holdout_2025_strict",
                "market.bars_1m_holdout_2025_research",
            },
        )

    def test_default_aggregate_view_configs_include_balanced_and_strict_multilevel_views(self):
        configs = dataset.default_aggregate_view_configs()

        self.assertEqual(
            set(configs),
            {
                "market.bars_5m_train_balanced",
                "market.bars_30m_train_balanced",
                "market.bars_day_train_balanced",
                "market.bars_week_train_balanced",
                "market.bars_5m_holdout_2025_strict",
                "market.bars_30m_holdout_2025_strict",
                "market.bars_day_holdout_2025_strict",
                "market.bars_week_holdout_2025_strict",
            },
        )

    def test_create_aggregate_view_sql_uses_source_view_and_close_auction_fold_for_5m(self):
        sql = dataset.create_aggregate_view_sql(
            view_name="market.bars_5m_train_balanced",
            source_view="market.bars_1m_train_balanced",
            target_level="5m",
        )

        self.assertIn("CREATE OR REPLACE VIEW market.bars_5m_train_balanced AS", sql)
        self.assertIn("FROM market.bars_1m_train_balanced", sql)
        self.assertIn("'5m' AS level", sql)
        self.assertIn("toIntervalMinute(5)", sql)
        self.assertIn("toStartOfInterval", sql)
        self.assertIn("argMin(open, source_trade_time)", sql)
        self.assertIn("argMax(close, source_trade_time)", sql)

    def test_create_aggregate_view_sql_uses_week_bucket_for_week_level(self):
        sql = dataset.create_aggregate_view_sql(
            view_name="market.bars_week_train_balanced",
            source_view="market.bars_1m_train_balanced",
            target_level="week",
        )

        self.assertIn("CREATE OR REPLACE VIEW market.bars_week_train_balanced AS", sql)
        self.assertIn("FROM market.bars_1m_train_balanced", sql)
        self.assertIn("'week' AS level", sql)
        self.assertIn("toMonday(trade_date)", sql)
        self.assertIn("toDateTime64(week_trade_date, 3, 'Asia/Shanghai')", sql)


if __name__ == "__main__":
    unittest.main()
