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
        self.assertIn("ORDER BY (symbol, level, trade_time)", sql)
        self.assertNotIn("ts_code", sql)
        for column in ("symbol", "level", "trade_time", "open", "high", "low", "close", "volume", "amount"):
            self.assertIn(column, sql)

    def test_init_sql_creates_adj_factors_table(self):
        sql = (INFRA / "init" / "001_create_market.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS market.adj_factors", sql)
        self.assertIn("adj_factor Float64", sql)
        self.assertIn("trade_date Date", sql)
        self.assertIn("ENGINE = ReplacingMergeTree(updated_at)", sql)
        self.assertIn("PARTITION BY toYYYYMM(trade_date)", sql)
        self.assertIn("ORDER BY (symbol, trade_date)", sql)

    def test_init_sql_creates_bar_anomalies_table_and_clean_view(self):
        sql = (INFRA / "init" / "001_create_market.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS market.bar_anomalies", sql)
        self.assertIn("rule LowCardinality(String)", sql)
        self.assertIn("ORDER BY (level, symbol, trade_time, rule)", sql)
        self.assertIn("CREATE VIEW IF NOT EXISTS market.bars_1m_clean", sql)
        self.assertIn("market.bar_anomalies", sql)
        self.assertNotIn("CREATE MATERIALIZED VIEW", sql)

    def test_init_sql_creates_bar_status_flags_table(self):
        sql = (INFRA / "init" / "001_create_market.sql").read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS market.bar_status_flags", sql)
        self.assertIn("status LowCardinality(String)", sql)
        self.assertIn("trade_date Date", sql)
        self.assertIn("note String", sql)
        self.assertIn("ORDER BY (level, symbol, trade_date, status)", sql)

    def test_management_script_keeps_schema_management_to_create_sql(self):
        script = (ROOT / "scripts" / "clickhouse_market.sh").read_text(encoding="utf-8")

        self.assertIn("001_create_market.sql", script)
        self.assertNotIn("ALTER TABLE", script)
        self.assertNotIn("has_legacy_column", script)
        self.assertNotIn("recreate_empty_legacy_ts_code_tables", script)
        self.assertNotIn("DROP TABLE IF EXISTS market.bars", script)
        self.assertNotIn("DROP TABLE IF EXISTS market.adj_factors", script)


if __name__ == "__main__":
    unittest.main()
