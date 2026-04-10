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
