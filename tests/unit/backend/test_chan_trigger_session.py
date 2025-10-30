from datetime import datetime

from web.backend.services.chan_trigger import ChanTriggerSession, LEVEL_TO_KL_TYPE


def test_chan_trigger_feed_returns_snapshot():
    session = ChanTriggerSession(symbol="demo", lv_list=[LEVEL_TO_KL_TYPE["1m"]])
    tick = {
        "time": datetime(2023, 1, 1, 9, 30),
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "volume": 1000.0,
    }
    snapshot = session.feed("1m", tick)
    assert snapshot["symbol"] == "demo"
    assert snapshot["level"] == "1m"
    assert snapshot["close"] == 10.2
    assert snapshot["klc_count"] >= 1
