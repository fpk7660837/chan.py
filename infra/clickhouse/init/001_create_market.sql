CREATE DATABASE IF NOT EXISTS market;

CREATE TABLE IF NOT EXISTS market.bars
(
    symbol LowCardinality(String),
    level LowCardinality(String),
    trade_time DateTime64(3, 'Asia/Shanghai'),
    trade_date Date MATERIALIZED toDate(trade_time),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64 DEFAULT 0,
    amount Float64 DEFAULT 0,
    source LowCardinality(String) DEFAULT '',
    updated_at DateTime64(3, 'Asia/Shanghai') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY (level, toYYYYMM(trade_time))
ORDER BY (symbol, level, trade_time)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS market.adj_factors
(
    symbol LowCardinality(String),
    trade_date Date,
    adj_factor Float64,
    source LowCardinality(String) DEFAULT '',
    updated_at DateTime64(3, 'Asia/Shanghai') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(trade_date)
ORDER BY (symbol, trade_date)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS market.bar_anomalies
(
    symbol LowCardinality(String),
    level LowCardinality(String),
    trade_time DateTime64(3, 'Asia/Shanghai'),
    trade_date Date,
    rule LowCardinality(String),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    amount Float64,
    source LowCardinality(String) DEFAULT '',
    updated_at DateTime64(3, 'Asia/Shanghai') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY (level, toYYYYMM(trade_time))
ORDER BY (level, symbol, trade_time, rule)
SETTINGS index_granularity = 8192;

CREATE VIEW IF NOT EXISTS market.bars_1m_clean AS
SELECT *
FROM market.bars
WHERE level = '1m'
  AND open > 0
  AND high > 0
  AND low > 0
  AND close > 0
  AND high >= greatest(open, low, close)
  AND low <= least(open, high, close)
  AND volume >= 0
  AND amount >= 0
  AND NOT isNaN(open)
  AND NOT isNaN(high)
  AND NOT isNaN(low)
  AND NOT isNaN(close)
  AND NOT isNaN(volume)
  AND NOT isNaN(amount)
  AND open <= 10000
  AND high <= 10000
  AND low <= 10000
  AND close <= 10000
  AND (symbol, level, trade_time) NOT IN
      (SELECT symbol, level, trade_time FROM market.bar_anomalies WHERE level = '1m');
