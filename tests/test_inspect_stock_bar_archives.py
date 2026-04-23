import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from App import inspect_stock_bar_archives as inspect


def _write_zip_csv(path: Path, member: str, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        text_rows = ["时间,代码,名称,开盘价,收盘价,最高价,最低价,成交量,成交额,涨幅,振幅"]
        text_rows.extend(",".join(str(value) for value in row) for row in rows)
        archive.writestr(member, "\ufeff" + "\n".join(text_rows) + "\n")


class InspectStockBarArchivesTests(unittest.TestCase):
    def test_build_manifest_classifies_level_adjust_years_and_zip_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "A股_分时数据_沪深"
            _write_zip_csv(
                source / "1分钟_按年汇总" / "2024_1min.zip",
                "sh600000_2024.csv",
                [["2024-01-02 09:30:00", "sh600000", "浦发银行", 6.63, 6.63, 6.63, 6.63, 1530, 1014390, 0, 0]],
            )
            _write_zip_csv(
                source / "1分钟_前复权_按年汇总" / "2024_1min.zip",
                "sh600000_2024.csv",
                [["2024-01-02 09:30:00", "sh600000", "浦发银行", 5.90, 5.90, 5.90, 5.90, 1530, 1014390, 0, 0]],
            )

            rows = inspect.build_manifest(source)

        self.assertEqual(len(rows), 2)
        raw = next(row for row in rows if row["adjust"] == "raw")
        qfq = next(row for row in rows if row["adjust"] == "qfq")
        self.assertEqual(raw["level"], "1m")
        self.assertEqual(raw["year_start"], "2024")
        self.assertEqual(raw["year_end"], "2024")
        self.assertEqual(raw["entry_count"], "1")
        self.assertEqual(raw["first_member"], "sh600000_2024.csv")
        self.assertEqual(qfq["level"], "1m")

    def test_detect_adjustment_samples_compares_raw_and_qfq_archives_without_extracting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "A股_分时数据_沪深"
            _write_zip_csv(
                source / "1分钟_按年汇总" / "2024_1min.zip",
                "sh600000_2024.csv",
                [["2024-01-02 09:30:00", "sh600000", "浦发银行", 6.63, 6.63, 6.63, 6.63, 1530, 1014390, 0, 0]],
            )
            _write_zip_csv(
                source / "1分钟_前复权_按年汇总" / "2024_1min.zip",
                "sh600000_2024.csv",
                [["2024-01-02 09:30:00", "sh600000", "浦发银行", 5.90, 5.90, 5.90, 5.90, 1530, 1014390, 0, 0]],
            )

            samples = inspect.detect_adjustment_samples(source, sample_limit=3)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["symbol"], "600000.SH")
        self.assertEqual(samples[0]["level"], "1m")
        self.assertEqual(samples[0]["year"], "2024")
        self.assertEqual(samples[0]["raw_close"], "6.63")
        self.assertEqual(samples[0]["qfq_close"], "5.9")
        self.assertEqual(samples[0]["same_volume"], "true")
        self.assertEqual(samples[0]["detected"], "raw_and_qfq_pair")

    def test_write_reports_creates_manifest_and_detection_csvs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "A股_分时数据_沪深"
            output = Path(tmp_dir) / "market-data"
            _write_zip_csv(
                source / "1分钟_按年汇总" / "2024_1min.zip",
                "sh600000_2024.csv",
                [["2024-01-02 09:30:00", "sh600000", "浦发银行", 6.63, 6.63, 6.63, 6.63, 1530, 1014390, 0, 0]],
            )
            _write_zip_csv(
                source / "1分钟_前复权_按年汇总" / "2024_1min.zip",
                "sh600000_2024.csv",
                [["2024-01-02 09:30:00", "sh600000", "浦发银行", 5.90, 5.90, 5.90, 5.90, 1530, 1014390, 0, 0]],
            )

            manifest_path, detection_path = inspect.write_reports(source, output, sample_limit=2)
            with manifest_path.open(newline="", encoding="utf-8") as fp:
                manifest_rows = list(csv.DictReader(fp))
            with detection_path.open(newline="", encoding="utf-8") as fp:
                detection_rows = list(csv.DictReader(fp))

        self.assertEqual(len(manifest_rows), 2)
        self.assertEqual(len(detection_rows), 1)
        self.assertIn("manifests", manifest_path.parts)
        self.assertIn("adjustment_detection", detection_path.parts)


if __name__ == "__main__":
    unittest.main()
