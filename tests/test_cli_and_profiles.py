import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cmha_runner import ROOT, ablation_names, make_plan, parser, primary_names, read_json
from run_batch import main as batch_main


class ProfileTests(unittest.TestCase):
    def test_complete_profile_inventory(self):
        self.assertEqual(len(primary_names()), 31)
        self.assertEqual(len(ablation_names()), 7)
        self.assertEqual(len(list((ROOT / "profiles/ablations").glob("*/*.json"))), 28)

    def test_all_primary_plans_keep_recorded_profiles(self):
        for name in primary_names():
            args = parser().parse_args(["--benchmark", name, "--dry-run"])
            plan = make_plan(args)
            self.assertTrue(plan["report_profile_unchanged"])
            self.assertEqual(plan["overrides"], {})
            self.assertEqual(plan["engine"], "CMHA")
            cfg = read_json(plan["profile"])
            self.assertEqual(cfg["workers"], 16)
            self.assertEqual(cfg["seed"], 21013001)
            self.assertEqual(cfg["late_cycle_cap_fraction"], .125)
            self.assertEqual(cfg["cycle_steps"], [80000, 160000, 320000, 640000])
            self.assertEqual(cfg["stage_block_max"], [12, 16, 20, 24])

    def test_ablation_engine_dispatch(self):
        for variant in ablation_names():
            args = parser().parse_args(["--benchmark", "3no3", "--ablation", variant, "--dry-run"])
            plan = make_plan(args)
            self.assertEqual(plan["engine"], "CMHA_no_heat_bath" if variant == "without_direct_heat_bath" else "CMHA")
            self.assertTrue(plan["report_profile_unchanged"])

    def test_overrides_are_explicit(self):
        args = parser().parse_args(["--benchmark", "H1", "--workers", "2", "--wall-seconds", "3",
                                   "--max-proposals", "64", "--target-stop", "disabled"])
        plan = make_plan(args)
        self.assertFalse(plan["report_profile_unchanged"])
        self.assertEqual(plan["overrides"]["max_proposals_per_worker"], 64)
        self.assertIn("--no-target-stop", plan["engine_args"])

    def test_existing_experiments_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "H1/existing").mkdir(parents=True)
            args = parser().parse_args(["--benchmark", "H1", "--out-dir", tmp, "--run-name", "existing"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                make_plan(args)

    def test_traversal_and_missing_custom_fields_rejected(self):
        args = parser().parse_args(["--benchmark", "H1", "--run-name", "../bad"])
        with self.assertRaises(ValueError):
            make_plan(args)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text('{"benchmark":"custom"}')
            args = parser().parse_args(["--profile", str(path)])
            with self.assertRaisesRegex(ValueError, "fields"):
                make_plan(args)

    def test_dry_run_imports_correct_engine_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in ([], ["--ablation", "without_direct_heat_bath"]):
                target = Path(tmp) / ("ablation" if extra else "full")
                command = [sys.executable, "-B", str(ROOT / "run_cmha.py"), "--benchmark", "3no3",
                           "--dry-run", "--out-dir", str(target), *extra]
                result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, check=True)
                parsed = json.loads(result.stdout)
                self.assertEqual(parsed["effective_config"]["workers"], 16)
                self.assertFalse(target.exists())

    def test_batch_is_plan_only_and_suite_budget_is_correct(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(output):
            self.assertEqual(batch_main(["--suite", "F90", "--out-dir", str(Path(tmp) / "new")]), 0)
            self.assertFalse((Path(tmp) / "new").exists())
        plan, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(plan["benchmarks"], 5)
        self.assertEqual(plan["sequential_max_annealing_hours"], 5)

    def test_missing_requested_ablation_is_not_silently_dropped(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            batch_main(["--benchmarks", "H1", "3no3", "--ablation", "single_block"])


if __name__ == "__main__":
    unittest.main()
