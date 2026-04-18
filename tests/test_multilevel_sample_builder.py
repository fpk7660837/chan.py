from dataclasses import dataclass, field

from Common.CEnum import BSP_TYPE
from ML.FeatureEngine.MultiLevelExtractor import MultiLevelExtractor
from ML.Training.MultiLevelDataLoader import InstrumentMultiLevelData
from ML.Training.MultiLevelSampleBuilder import MultiLevelSampleBuilder


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


def _make_bars(start: int, stop: int, step: int) -> list[FakeBar]:
    bars = []
    for idx, ts in enumerate(range(start, stop + step, step), 1):
        bars.append(FakeBar(idx=idx, open=100.0 + idx, close=100.5 + idx, time=FakeTime(ts=ts)))
    return bars


class TestMultiLevelSampleBuilder:
    def test_build_buy_entry_events_keeps_all_30m_bsp_types(self):
        bars_5m = _make_bars(0, 150, 5)
        bars_30m = _make_bars(0, 150, 30)
        bars_day = [FakeBar(idx=1, open=100.0, close=101.0, time=FakeTime(ts=0))]

        data = InstrumentMultiLevelData(
            code="600519",
            name="Kweichow Moutai",
            level_bars={"day": bars_day, "30m": bars_30m, "5m": bars_5m},
            level_bsps={
                "day": [FakeBSP(klu=bars_day[0], is_buy=True, type=["T1"])],
                "30m": [
                    FakeBSP(klu=bars_30m[1], is_buy=True, type=["T1"]),
                    FakeBSP(klu=bars_30m[2], is_buy=True, type=["T2"]),
                    FakeBSP(klu=bars_30m[3], is_buy=True, type=["T3A"]),
                    FakeBSP(klu=bars_30m[4], is_buy=False, type=["T1"]),
                ],
                "5m": [
                    FakeBSP(klu=bars_5m[7], is_buy=True, type=["T1"]),
                    FakeBSP(klu=bars_5m[10], is_buy=True, type=["T2"]),
                    FakeBSP(klu=bars_5m[13], is_buy=True, type=["T3A"]),
                    FakeBSP(klu=bars_5m[20], is_buy=False, type=["T1"]),
                ],
            },
        )

        builder = MultiLevelSampleBuilder()
        events = builder.build_buy_entry_events([data])

        assert len(events) == 3
        assert {event.bsp_type for event in events} == {"T1", "T2", "T3A"}

    def test_build_exit_warning_events_uses_first_5m_reverse_bsp_per_position(self):
        bars_5m = _make_bars(0, 150, 5)
        bars_30m = _make_bars(0, 150, 30)
        bars_day = [FakeBar(idx=1, open=100.0, close=101.0, time=FakeTime(ts=0))]

        data = InstrumentMultiLevelData(
            code="600519",
            name="Kweichow Moutai",
            level_bars={"day": bars_day, "30m": bars_30m, "5m": bars_5m},
            level_bsps={
                "day": [FakeBSP(klu=bars_day[0], is_buy=True, type=["T1"])],
                "30m": [
                    FakeBSP(klu=bars_30m[1], is_buy=True, type=["T1"]),
                    FakeBSP(klu=bars_30m[4], is_buy=False, type=["T1"]),
                ],
                "5m": [
                    FakeBSP(klu=bars_5m[7], is_buy=False, type=["T1"]),
                    FakeBSP(klu=bars_5m[9], is_buy=False, type=["T2"]),
                ],
            },
        )

        builder = MultiLevelSampleBuilder()
        warnings = builder.build_exit_warning_events([data])

        assert len(warnings) == 1
        assert warnings[0].warning_bsp.klu.time.ts == bars_5m[7].time.ts

    def test_multilevel_extractor_zero_fills_missing_levels(self):
        bars_30m = _make_bars(0, 60, 30)
        bsp_30m = FakeBSP(klu=bars_30m[1], is_buy=True, type=[BSP_TYPE.T1])

        extractor = MultiLevelExtractor(
            {
                "level_list": ["day", "30m", "5m"],
                "use_bi_features": False,
                "use_seg_features": False,
                "use_zs_features": False,
                "use_klu_features": False,
            }
        )

        features = extractor.extract_multi_level({"30m": bsp_30m})

        assert len(features) == len(extractor.get_feature_names())
        assert features["30m_bsp_type"] == 1.0
        assert features["day_bsp_type"] == 0.0
        assert features["5m_recent_return_20"] == 0.0
