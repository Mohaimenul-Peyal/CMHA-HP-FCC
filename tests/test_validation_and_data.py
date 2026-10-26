import csv
import hashlib
import json
from pathlib import Path
import unittest

from cmha_runner import ROOT
from cmha_results import summarize_run, validate_fold


class ValidationTests(unittest.TestCase):
    def test_fcc_triangle_counts_nonconsecutive_contact_only(self):
        result = validate_fold("HHH", [[0, 0, 0], [1, 1, 0], [1, 0, 1]], 3, 1)
        self.assertEqual(result["HP_energy"], -1)
        self.assertEqual(validate_fold("HPH", [[0, 0, 0], [1, 1, 0], [1, 0, 1]], 3)["HH"], 1)

    def test_invalid_geometries_rejected(self):
        bad = [([[0, 0, 0], [1, 0, 0], [1, 0, 1]], 3),
               ([[0, 0, 0], [1, 1, 0], [0, 0, 0]], 3),
               ([[0, 0, 0], [3, 1, 0], [1, 0, 1]], 3),
               ([[0, 0, 0], [1.25, .75, 0], [1, 0, 1]], 3),
               ([[0, 0, 0], [True, 1, 0], [1, 0, 1]], 3)]
        for coords, side in bad:
            with self.assertRaises(ValueError):
                validate_fold("HHH", coords, side)

    def test_wrong_saved_contact_count_rejected(self):
        with self.assertRaises(ValueError):
            validate_fold("HHH", [[0, 0, 0], [1, 1, 0], [1, 0, 1]], 3, 2)

    def test_all_primary_best_folds(self):
        folds = json.loads((ROOT / "data/selected_folds.json").read_text())
        self.assertEqual(len(folds), 31)
        for name, fold in folds.items():
            result = validate_fold(fold["seq"], fold["coords"], fold["L"], fold["HH"])
            self.assertTrue(result["valid"], name)

    def test_sequence_hashes_and_s3_length(self):
        with (ROOT / "benchmarks.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            self.assertEqual(len(row["hp_sequence"]), int(row["N"]))
            self.assertEqual(hashlib.sha256(row["hp_sequence"].encode()).hexdigest(), row["sequence_sha256"])
        self.assertEqual(next(r for r in rows if r["benchmark"] == "S3FCC")["N"], "161")

    def test_measured_h1_example(self):
        result = summarize_run(ROOT / "examples/H1")
        self.assertEqual(result["workers"], 16)
        self.assertEqual(result["best_HH"], 69)
        self.assertEqual(result["workers_reaching_target"], 16)
        self.assertEqual(result["mean_worker_best_HH"], 69)
        self.assertEqual(result["sample_SD_worker_best_HH"], 0)

    def test_reporting_and_ablation_controls_remain_separate(self):
        data = json.loads((ROOT / "data/paper_data.json").read_text())
        primary = next(r for r in data["primary"] if r["benchmark"] == "3mr7")
        control = next(r for r in data["ablations"] if r["benchmark"] == "3mr7" and r["variant"] == "full_cmha_repeat")
        self.assertNotEqual(primary["mean_HH"], control["mean_HH"])
        self.assertNotIn("seeds", data)


if __name__ == "__main__":
    unittest.main()
