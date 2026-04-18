from dataclasses import dataclass, field

from ML.Training.LabelBuilder import LabelBuilder
from ML.Training.MultiLevelSampleBuilder import BuyEntryEvent, ExitWarningEvent


@dataclass
class FakeTime:
    ts: int


@dataclass
class FakeBar:
    idx: int
    open: float
    close: float
    time: FakeTime


@dataclass
class FakeBSP:
    klu: FakeBar
    is_buy: bool
    type: list[str]
    is_segbsp: bool = False
    features: dict = field(default_factory=dict)
    bi: object = None


def test_buy_entry_label_uses_next_5m_open_for_entry_and_exit():
    entry_bsp = FakeBSP(klu=FakeBar(idx=1, open=100.0, close=101.0, time=FakeTime(ts=30)), is_buy=True, type=["T1"])
    exit_bsp = FakeBSP(klu=FakeBar(idx=2, open=104.0, close=103.0, time=FakeTime(ts=120)), is_buy=False, type=["T1"])
    event = BuyEntryEvent(
        position_id="600519:30:T1",
        code="600519",
        name="Kweichow Moutai",
        bsp_type="T1",
        entry_bsp=entry_bsp,
        entry_klu=FakeBar(idx=10, open=100.0, close=101.0, time=FakeTime(ts=35)),
        exit_warning_bsp=None,
        exit_confirm_bsp=exit_bsp,
        exit_klu=FakeBar(idx=26, open=105.0, close=104.0, time=FakeTime(ts=125)),
        context={"day": None, "30m": entry_bsp, "5m": None},
    )

    builder = LabelBuilder({"round_trip_cost_pct": 0.01})
    label, ret = builder.label_buy_entry(event)

    assert label == 1
    assert round(ret, 4) == 0.04


def test_buy_entry_label_drops_samples_without_confirmed_30m_exit():
    entry_bsp = FakeBSP(klu=FakeBar(idx=1, open=100.0, close=101.0, time=FakeTime(ts=30)), is_buy=True, type=["T1"])
    event = BuyEntryEvent(
        position_id="600519:30:T1",
        code="600519",
        name="Kweichow Moutai",
        bsp_type="T1",
        entry_bsp=entry_bsp,
        entry_klu=FakeBar(idx=10, open=100.0, close=101.0, time=FakeTime(ts=35)),
        exit_warning_bsp=None,
        exit_confirm_bsp=None,
        exit_klu=None,
        context={"day": None, "30m": entry_bsp, "5m": None},
    )

    builder = LabelBuilder()

    assert builder.label_buy_entry(event) is None


def test_exit_warning_label_requires_30m_confirm_within_horizon():
    entry_bsp = FakeBSP(klu=FakeBar(idx=1, open=100.0, close=101.0, time=FakeTime(ts=30)), is_buy=True, type=["T1"])
    warning_bsp = FakeBSP(klu=FakeBar(idx=10, open=103.0, close=102.0, time=FakeTime(ts=45)), is_buy=False, type=["T2"])
    confirm_bsp = FakeBSP(klu=FakeBar(idx=4, open=98.0, close=97.0, time=FakeTime(ts=300)), is_buy=False, type=["T1"])
    entry_event = BuyEntryEvent(
        position_id="600519:30:T1",
        code="600519",
        name="Kweichow Moutai",
        bsp_type="T1",
        entry_bsp=entry_bsp,
        entry_klu=FakeBar(idx=10, open=100.0, close=101.0, time=FakeTime(ts=35)),
        exit_warning_bsp=warning_bsp,
        exit_confirm_bsp=confirm_bsp,
        exit_klu=FakeBar(idx=60, open=99.0, close=98.0, time=FakeTime(ts=305)),
        context={"day": None, "30m": entry_bsp, "5m": None},
    )
    warning_event = ExitWarningEvent(
        position_id="600519:30:T1",
        code="600519",
        name="Kweichow Moutai",
        warning_bsp=warning_bsp,
        confirm_bsp=confirm_bsp,
        confirm_delay_bars_30m=9,
        warning_klu=warning_bsp.klu,
        context={"day": None, "30m": entry_bsp, "5m": warning_bsp},
        entry_event=entry_event,
    )

    builder = LabelBuilder({"exit_warning_confirmation_horizon_30m": 8})

    assert builder.label_exit_warning(warning_event) == 0
