CREATE DATABASE IF NOT EXISTS market;

CREATE TABLE IF NOT EXISTS market.bars
(
    ts_code LowCardinality(String),
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
ORDER BY (ts_code, level, trade_time)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS market.adj_factors
(
    ts_code LowCardinality(String),
    trade_date Date,
    adj_factor Float64,
    source LowCardinality(String) DEFAULT '',
    updated_at DateTime64(3, 'Asia/Shanghai') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(trade_date)
ORDER BY (ts_code, trade_date)
SETTINGS index_granularity = 8192;
