from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


ROOT = Path(__file__).resolve().parent.parent
LOCAL_DB_ENV_VAR = "CHAN_LOCAL_DB_PATH"
DEFAULT_LOCAL_DB_PATH = ROOT / "data" / "market_data.sqlite3"
LOCAL_SQLITE_DATA_SRC = "custom:SQLiteDailyBarAPI.CSQLiteDailyBarAPI"


def normalize_code(code: str) -> str:
    return str(code).strip().replace(".SH", "").replace(".SZ", "").replace("sh.", "").replace("sz.", "")


def resolve_local_db_path(explicit_path: Optional[str | Path] = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()

    env_path = os.environ.get(LOCAL_DB_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_LOCAL_DB_PATH.resolve()


def local_db_exists(explicit_path: Optional[str | Path] = None) -> bool:
    return resolve_local_db_path(explicit_path).exists()


def connect_local_db(explicit_path: Optional[str | Path] = None) -> sqlite3.Connection:
    db_path = resolve_local_db_path(explicit_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def ensure_daily_bar_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            adjust TEXT NOT NULL DEFAULT 'qfq',
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL DEFAULT 0,
            amount REAL NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (code, trade_date, adjust)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_daily_bars_code_date
        ON daily_bars (code, trade_date)
        """
    )
    conn.commit()


def normalize_trade_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    if not text:
        raise ValueError("trade_date cannot be empty")

    if len(text) >= 10 and text[4] in "-/" and text[7] in "-/":
        return f"{text[:4]}-{text[5:7]}-{text[8:10]}"
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"

    raise ValueError(f"Unsupported trade_date format: {value}")


def upsert_daily_bars(
    conn: sqlite3.Connection,
    *,
    code: str,
    rows: Sequence[Dict[str, Any]],
    adjust: str = "qfq",
    source: str = "akshare",
) -> int:
    if not rows:
        return 0

    normalized_code = normalize_code(code)
    updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    payload = [
        (
            normalized_code,
            normalize_trade_date(row["trade_date"]),
            adjust,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row.get("volume", 0.0) or 0.0),
            float(row.get("amount", 0.0) or 0.0),
            source,
            updated_at,
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO daily_bars (
            code, trade_date, adjust, open, high, low, close, volume, amount, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, trade_date, adjust) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            amount = excluded.amount,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def _row_to_klu_dict(row: Sequence[Any]) -> Dict[str, Any]:
    trade_date, open_price, high_price, low_price, close_price, volume, amount = row
    normalized_date = normalize_trade_date(trade_date)
    return {
        DATA_FIELD.FIELD_TIME: CTime(
            int(normalized_date[:4]),
            int(normalized_date[5:7]),
            int(normalized_date[8:10]),
            0,
            0,
        ),
        DATA_FIELD.FIELD_OPEN: str2float(open_price),
        DATA_FIELD.FIELD_HIGH: str2float(high_price),
        DATA_FIELD.FIELD_LOW: str2float(low_price),
        DATA_FIELD.FIELD_CLOSE: str2float(close_price),
        DATA_FIELD.FIELD_VOLUME: str2float(volume),
        DATA_FIELD.FIELD_TURNOVER: str2float(amount),
    }


class CSQLiteDailyBarAPI(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        self.db_path = resolve_local_db_path()
        super(CSQLiteDailyBarAPI, self).__init__(normalize_code(code), k_type, begin_date, end_date, autype)

    def get_kl_data(self) -> Iterable[CKLine_Unit]:
        if self.k_type != KL_TYPE.K_DAY:
            raise CChanException("SQLiteDailyBarAPI only supports K_DAY", ErrCode.SRC_DATA_TYPE_ERR)
        if self.autype not in (None, AUTYPE.QFQ):
            raise CChanException("SQLiteDailyBarAPI only supports AUTYPE.QFQ", ErrCode.SRC_DATA_TYPE_ERR)
        if not self.db_path.exists():
            raise CChanException(f"local market data db not found: {self.db_path}", ErrCode.SRC_DATA_NOT_FOUND)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT trade_date, open, high, low, close, volume, amount
                FROM daily_bars
                WHERE code = ? AND adjust = 'qfq'
                  AND (? IS NULL OR trade_date >= ?)
                  AND (? IS NULL OR trade_date <= ?)
                ORDER BY trade_date
                """,
                (
                    self.code,
                    self.begin_date,
                    self.begin_date,
                    self.end_date,
                    self.end_date,
                ),
            )
            for row in cursor:
                yield CKLine_Unit(_row_to_klu_dict(row))
        finally:
            conn.close()

    def SetBasciInfo(self):
        self.name = self.code
        self.is_stock = True

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass
