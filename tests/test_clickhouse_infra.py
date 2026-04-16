import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INFRA = ROOT / "infra" / "clickhouse"


class ClickHouseInfraTests(unittest.TestCase):
    def test_compose_pins_image_and_mounts_persistent_market_data(self):
        compose = (INFRA / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("clickhouse/clickhouse-server:", compose)
        self.assertNotIn("clickhouse/clickhouse-server:latest", compose)
        self.assertIn("${MARKET_DATA_ROOT}/clickhouse/data:/var/lib/clickhouse", compose)
        self.assertIn("${MARKET_DATA_ROOT}/clickhouse/logs:/var/log/clickhouse-server", compose)
        self.assertIn("${MARKET_DATA_ROOT}/parquet:/market-data/parquet", compose)
        self.assertIn("nofile:", compose)

    def test_env_example_documents_required_local_paths_and_credentials(self):
        env_example = (INFRA / ".env.example").read_text(encoding="utf-8")

        self.assertIn("MARKET_DATA_ROOT=/Users/kevinfu/market-data", env_example)
        self.assertIn("CLICKHOUSE_DB=market", env_example)
        self.assertIn("CLICKHOUSE_USER=chan", env_example)
        self.assertIn("CLICKHOUSE_PASSWORD=", env_example)

    def test_init_sql_creates_market_bars_merge_tree_for_ohlcv_data(self):
        sql = (INFRA / "init" / "001_create_market.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE DATABASE IF NOT EXISTS market", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS market.bars", sql)
        self.assertIn("MergeTree", sql)
        self.assertIn("PARTITION BY (level, toYYYYMM(trade_time))", sql)
        self.assertIn("ORDER BY (ts_code, level, trade_time)", sql)
        for column in ("ts_code", "level", "trade_time", "open", "high", "low", "close", "volume", "amount"):
            self.assertIn(column, sql)

    def test_init_sql_creates_adj_factors_table_without_query_views(self):
        sql = (INFRA / "init" / "001_create_market.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS market.adj_factors", sql)
        self.assertIn("adj_factor Float64", sql)
        self.assertIn("trade_date Date", sql)
        self.assertIn("ENGINE = ReplacingMergeTree(updated_at)", sql)
        self.assertIn("PARTITION BY toYYYYMM(trade_date)", sql)
        self.assertIn("ORDER BY (ts_code, trade_date)", sql)
        self.assertNotIn("CREATE VIEW", sql)
        self.assertNotIn("CREATE MATERIALIZED VIEW", sql)


if __name__ == "__main__":
    unittest.main()
