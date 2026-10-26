#!/usr/bin/env python3
"""Plan a sequential benchmark batch; require --execute to launch simulations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from cmha_runner import ROOT, primary_names, ablation_names, read_json, positive_float, positive_int


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--suite", help="Suite name from benchmarks.csv, or all (default)", default=None)
    selection.add_argument("--benchmarks", nargs="+", choices=primary_names())
    ap.add_argument("--ablation", choices=ablation_names())
    ap.add_argument("--order", choices=("as-listed", "longest-first", "shortest-first"), default="as-listed")
    ap.add_argument("--base-seed", type=positive_int)
    ap.add_argument("--workers", type=positive_int)
    ap.add_argument("--wall-seconds", type=positive_float)
    ap.add_argument("--target-stop", choices=("profile", "enabled", "disabled"), default="profile")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args(argv)
    rows = read_json(ROOT / "data/paper_data.json")["primary"]
    suites = {row["suite"] for row in rows}
    if args.suite not in (None, "all") and args.suite not in suites:
        ap.error(f"Unknown suite; choose all or {', '.join(sorted(suites))}")
    names = args.benchmarks or [row["benchmark"] for row in rows if args.suite in (None, "all", row["suite"])]
    if len(set(names)) != len(names):
        ap.error("Do not list a benchmark twice in one batch")
    if args.ablation:
        if args.benchmarks:
            missing = [name for name in names if not (ROOT / "profiles/ablations" / args.ablation / f"{name}.json").is_file()]
            if missing:
                ap.error(f"Ablation profiles do not exist for: {', '.join(missing)}")
        names = [name for name in names if (ROOT / "profiles/ablations" / args.ablation / f"{name}.json").is_file()]
    if not names:
        ap.error("No matching profiles")
    sizes = {row["benchmark"]: row["N"] for row in rows}
    if args.order != "as-listed":
        names.sort(key=sizes.get, reverse=args.order == "longest-first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    jobs = []
    for name in names:
        profile = ROOT / "profiles/ablations" / args.ablation / f"{name}.json" if args.ablation else ROOT / "profiles/primary" / f"{name}.json"
        cfg = read_json(profile)
        command = [sys.executable, "-B", str(ROOT / "run_cmha.py"), "--benchmark", name,
                   "--out-dir", str(args.out_dir.resolve()), "--run-name", f"CMHA_BATCH_{name}_{args.ablation or 'full'}_{stamp}"]
        if args.ablation:
            command += ["--ablation", args.ablation]
        for attr, flag in (("base_seed", "--base-seed"), ("workers", "--workers"), ("wall_seconds", "--wall-seconds")):
            if getattr(args, attr) is not None:
                command += [flag, str(getattr(args, attr))]
        command += ["--target-stop", args.target_stop]
        jobs.append(dict(benchmark=name, profile=profile.relative_to(ROOT).as_posix(),
                         max_annealing_seconds=args.wall_seconds or cfg["wall_seconds"], argv=command))
    total = sum(job["max_annealing_seconds"] for job in jobs)
    plan = dict(benchmarks=len(jobs), sequential_max_annealing_hours=total/3600,
                timing_note="Parallel workers; sequential benchmarks. Add SAW/JIT/export overhead; target stopping can shorten runs.", jobs=jobs)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Plan only. Add --execute to launch this batch.")
        return 0
    control = args.out_dir.resolve() / "batches" / stamp
    control.mkdir(parents=True, exist_ok=False)
    (control / "batch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    completed = []
    for job in jobs:
        returncode = subprocess.call(job["argv"], cwd=ROOT)
        completed.append(dict(benchmark=job["benchmark"], returncode=returncode))
        temporary = control / "batch_status.tmp"
        temporary.write_text(json.dumps(dict(completed=completed, total=len(jobs)), indent=2) + "\n")
        temporary.replace(control / "batch_status.json")
        if returncode and not args.continue_on_error:
            return returncode
    return 1 if any(row["returncode"] for row in completed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
