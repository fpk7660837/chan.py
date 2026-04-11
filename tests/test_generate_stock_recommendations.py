from datetime import datetime
import unittest
from types import SimpleNamespace
from unittest import mock

from App import generate_stock_recommendations as reco


class FakeFrame:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for index, row in enumerate(self._rows):
            yield index, row


class RecommendationUniverseTests(unittest.TestCase):
    def test_parse_args_accepts_universe_flag(self):
        with mock.patch("sys.argv", ["prog", "--as-of", "2024-12-31", "--universe", "hs300"]):
            args = reco.parse_args()

        self.assertEqual(args.universe, "hs300")

    def test_load_universe_prefers_codes_over_codes_file_and_universe(self):
        args = SimpleNamespace(
            codes="600519,000333",
            codes_file="codes.txt",
            universe="hs300",
            limit=20,
        )

        result = reco.load_universe(args)

        self.assertEqual(result, [("600519", ""), ("000333", "")])

    def test_load_universe_prefers_codes_file_over_named_universe(self):
        args = SimpleNamespace(
            codes=None,
            codes_file="codes.txt",
            universe="hs300",
            limit=20,
        )

        with mock.patch.object(
            reco,
            "load_codes_from_file",
            return_value=[("600036", "招商银行")],
        ) as load_codes_from_file_mock, mock.patch.object(
            reco,
            "load_named_universe",
            return_value=[("600519", "贵州茅台")],
        ) as load_named_universe_mock:
            result = reco.load_universe(args)

        self.assertEqual(result, [("600036", "招商银行")])
        load_codes_from_file_mock.assert_called_once()
        load_named_universe_mock.assert_not_called()

    def test_load_universe_honors_limit_for_named_universe(self):
        args = SimpleNamespace(codes=None, codes_file=None, universe="hs300", limit=10)
        named_universe = [(f"600{i:03d}", f"name-{i}") for i in range(20)]

        with mock.patch.object(reco, "load_named_universe", return_value=named_universe):
            result = reco.load_universe(args)

        self.assertEqual(result, named_universe[:10])

    def test_load_universe_resolves_hs300_from_csindex(self):
        args = SimpleNamespace(codes=None, codes_file=None, universe="hs300", limit=None)
        fake_ak = SimpleNamespace(
            index_stock_cons_csindex=mock.Mock(
                return_value=FakeFrame(
                    [
                        {"成分券代码": "600519", "成分券名称": "贵州茅台"},
                        {"成分券代码": "000333", "成分券名称": "美的集团"},
                    ]
                )
            )
        )

        with mock.patch.object(reco, "ak", fake_ak):
            result = reco.load_universe(args)

        self.assertEqual(result, [("600519", "贵州茅台"), ("000333", "美的集团")])

    def test_get_hs300_stocks_falls_back_to_sina_when_csindex_fails(self):
        fake_ak = SimpleNamespace(
            index_stock_cons_csindex=mock.Mock(side_effect=RuntimeError("csindex unavailable")),
            index_stock_cons_sina=mock.Mock(
                return_value=FakeFrame(
                    [
                        {"code": "600519", "name": "贵州茅台"},
                        {"code": "000333", "name": "美的集团"},
                    ]
                )
            ),
        )

        with mock.patch.object(reco, "ak", fake_ak):
            result = reco.get_hs300_stocks()

        self.assertEqual(result, [("600519", "贵州茅台"), ("000333", "美的集团")])
        fake_ak.index_stock_cons_sina.assert_called_once_with(symbol="000300")

    def test_load_universe_rejects_unknown_named_universe(self):
        args = SimpleNamespace(codes=None, codes_file=None, universe="unknown", limit=None)

        with self.assertRaisesRegex(ValueError, "Unsupported universe"):
            reco.load_universe(args)

    def test_load_universe_surfaces_named_universe_failure_without_fallback(self):
        args = SimpleNamespace(codes=None, codes_file=None, universe="hs300", limit=None)

        with mock.patch.object(reco, "load_named_universe", side_effect=RuntimeError("boom")), mock.patch.object(
            reco, "get_tradable_stocks"
        ) as get_tradable_stocks_mock:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                reco.load_universe(args)

        get_tradable_stocks_mock.assert_not_called()

    def test_get_hs300_stocks_raises_when_provider_returns_empty_rows(self):
        fake_ak = SimpleNamespace(index_stock_cons_csindex=mock.Mock(return_value=FakeFrame([])))

        with mock.patch.object(reco, "ak", fake_ak):
            with self.assertRaisesRegex(RuntimeError, "resolved to zero constituents"):
                reco.get_hs300_stocks()

    def test_resolve_market_data_src_defaults_hs300_to_baostock(self):
        self.assertEqual(reco.resolve_market_data_src("hs300"), reco.DATA_SRC.BAO_STOCK)
        self.assertEqual(reco.resolve_market_data_src(None), reco.DATA_SRC.AKSHARE)

    def test_resolve_market_data_src_prefers_local_sqlite_when_available(self):
        with mock.patch.object(reco, "local_market_data_exists", return_value=True):
            self.assertEqual(reco.resolve_market_data_src("hs300"), reco.LOCAL_SQLITE_DATA_SRC)
            self.assertEqual(reco.resolve_market_data_src(None), reco.LOCAL_SQLITE_DATA_SRC)

    def test_load_chan_pool_uses_local_sqlite_data_src_when_available(self):
        fake_bar = SimpleNamespace(time=SimpleNamespace(year=2024, month=12, day=31))
        fake_chan = mock.MagicMock()
        fake_chan.__getitem__.return_value = SimpleNamespace(klu_iter=lambda: iter([fake_bar]))

        with mock.patch.object(reco, "resolve_market_data_src", return_value=reco.LOCAL_SQLITE_DATA_SRC), mock.patch.object(
            reco,
            "CChan",
            return_value=fake_chan,
        ) as chan_cls:
            chan_list, code_name_map, skipped = reco.load_chan_pool(
                universe=[("600519", "贵州茅台")],
                as_of=datetime(2024, 12, 31),
                history_days=60,
                stale_days=20,
                universe_name="hs300",
            )

        self.assertEqual(len(chan_list), 1)
        self.assertEqual(code_name_map, {"600519": "贵州茅台"})
        self.assertEqual(skipped, [])
        self.assertEqual(chan_cls.call_args.kwargs["data_src"], reco.LOCAL_SQLITE_DATA_SRC)
