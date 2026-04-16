# ClickHouse Market Data

Local ClickHouse deployment for A-share OHLCV data.

## Layout

```text
/Users/kevinfu/market-data/
  clickhouse/
    data/
    logs/
    config/
  parquet/
  backup/
    clickhouse/
```

Parquet files are the long-term portable copy. ClickHouse is the query layer and can be rebuilt from Parquet if needed.

## Commands

```bash
scripts/clickhouse_market.sh init-env
scripts/clickhouse_market.sh start
scripts/clickhouse_market.sh init
scripts/clickhouse_market.sh status
scripts/clickhouse_market.sh client
scripts/clickhouse_market.sh backup
scripts/clickhouse_market.sh stop
```

The local `.env` file is intentionally ignored by git because it contains the ClickHouse password.
