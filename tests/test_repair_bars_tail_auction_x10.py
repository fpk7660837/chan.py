import unittest

from App import repair_bars_tail_auction_x10 as repair


class RepairBarsTailAuctionX10Tests(unittest.TestCase):
    def test_default_trade_dates_cover_known_local_fix_batches(self):
        self.assertEqual(repair.DEFAULT_TRADE_DATES, ("2010-11-01", "2011-02-16"))

    def test_candidate_where_sql_limits_to_tail_auction_x10_pattern(self):
        where_sql = repair.candidate_where_sql(level="1m", trade_dates=("2010-11-01", "2011-02-16"))

        self.assertIn("level = '1m'", where_sql)
        self.assertIn("trade_date IN ('2010-11-01', '2011-02-16')", where_sql)
        self.assertIn("(symbol, trade_date) IN", where_sql)
        self.assertIn("count() = 241", where_sql)
        self.assertIn("max(high) / min(low) > 10", where_sql)
        self.assertIn("toHour(trade_time) = 15", where_sql)
        self.assertIn("toMinute(trade_time) = 0", where_sql)
        self.assertIn("open / close > 9", where_sql)
        self.assertIn("open / close < 11", where_sql)
        self.assertIn("high / close > 9", where_sql)
        self.assertIn("high / close < 11", where_sql)
        self.assertIn("low / close > 0.8", where_sql)
        self.assertIn("low / close < 1.2", where_sql)

    def test_export_candidates_query_includes_repaired_values(self):
        query = repair.export_candidates_query(level="1m", trade_dates=("2010-11-01",))

        self.assertIn("SELECT", query)
        self.assertIn("toString(trade_time) AS trade_time_text", query)
        self.assertIn("open / 10.0 AS repaired_open", query)
        self.assertIn("high / 10.0 AS repaired_high", query)
        self.assertIn("ORDER BY trade_date, symbol, trade_time", query)
        self.assertIn("FORMAT CSVWithNames", query)

    def test_update_candidates_query_only_updates_open_and_high(self):
        query = repair.update_candidates_query(level="1m", trade_dates=("2011-02-16",))

        self.assertIn("ALTER TABLE market.bars UPDATE", query)
        self.assertIn("open = open / 10.0", query)
        self.assertIn("high = high / 10.0", query)
        self.assertNotIn("low =", query)
        self.assertNotIn("close =", query)
        self.assertIn("SETTINGS mutations_sync = 2", query)


if __name__ == "__main__":
    unittest.main()
