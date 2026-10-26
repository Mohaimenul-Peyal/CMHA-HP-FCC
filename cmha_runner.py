"""Portable public interface to the original CMHA search snapshots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def primary_names():
    return sorted(path.stem for path in (ROOT / "profiles/primary").glob("*.json"))


def ablation_names():
    return sorted(path.name for path in (ROOT / "profiles/ablations").iterdir() if path.is_dir())


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--benchmark", choices=primary_names())
    selection.add_argument("--profile", type=Path, help="Custom JSON profile, based on a bundled primary profile")
    ap.add_argument("--ablation", choices=ablation_names())
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--run-name", help="Optional single directory name; existing runs are never overwritten")
    ap.add_argument("--base-seed", type=positive_int)
    ap.add_argument("--workers", type=positive_int)
    ap.add_argument("--wall-seconds", type=positive_float)
    ap.add_argument("--max-proposals", type=positive_int, help="Testing override, not a reported wall-budget experiment")
    ap.add_argument("--target-stop", choices=("profile", "enabled", "disabled"), default="profile")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print the effective configuration; no output directory or JIT compilation")
    ap.add_argument("--list-benchmarks", action="store_true")
    ap.add_argument("--list-ablations", action="store_true")
    ap.add_argument("--version", action="version", version=f"CMHA-HP-FCC {VERSION}")
    return ap


def make_plan(args):
    if args.profile:
        if args.ablation:
            raise ValueError("Use --benchmark with --ablation; custom profiles use the full engine")
        profile = args.profile.resolve()
    elif args.benchmark:
        parent = ROOT / "profiles" / ("ablations" if args.ablation else "primary")
        profile = parent / args.ablation / f"{args.benchmark}.json" if args.ablation else parent / f"{args.benchmark}.json"
    else:
        raise ValueError("Choose --benchmark, --profile, or a listing option")
    if not profile.is_file():
        raise ValueError(f"Profile not found: {profile}")
    original = read_json(profile)
    required = {"benchmark", "seq", "l_size", "seed", "workers", "wall_seconds", "target_stop_enabled"}
    if not isinstance(original, dict) or not required <= original.keys():
        raise ValueError("Profile must be a JSON object containing the standard benchmark/settings fields")
    benchmark = original["benchmark"]
    if args.benchmark and benchmark != args.benchmark:
        raise ValueError("Profile benchmark does not match its requested identifier")
    engine_name = "CMHA_no_heat_bath" if args.ablation == "without_direct_heat_bath" else "CMHA"
    engine_dir = ROOT / "engines" / engine_name
    overrides = {}
    for attribute, key in (("base_seed", "seed"), ("workers", "workers"),
                           ("wall_seconds", "wall_seconds"), ("max_proposals", "max_proposals_per_worker")):
        value = getattr(args, attribute)
        if value is not None and value != original.get(key):
            overrides[key] = value
    stop = original["target_stop_enabled"] if args.target_stop == "profile" else args.target_stop == "enabled"
    if stop != original["target_stop_enabled"]:
        overrides["target_stop_enabled"] = stop
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    name = args.run_name or f"CMHA_{benchmark}_{args.ablation or 'full'}_L{original['l_size']}_{stamp}"
    if name in (".", "..") or any(ch in name for ch in "/\\:") or not name:
        raise ValueError("--run-name must be one safe directory name")
    if benchmark in (".", "..") or any(ch in benchmark for ch in "/\\:") or not benchmark:
        raise ValueError("Profile benchmark must be one safe directory name")
    output = args.out_dir.resolve() / benchmark / name
    if output.exists() and not args.dry_run:
        raise ValueError(f"Refusing to overwrite an existing experiment: {output}")
    engine_args = ["--profile", str(profile), "--out-dir", str(args.out_dir.resolve()), "--run-name", name]
    for attr, flag in (("base_seed", "--seed"), ("workers", "--workers"),
                       ("wall_seconds", "--wall-seconds"), ("max_proposals", "--max-proposals")):
        value = getattr(args, attr)
        if value is not None:
            engine_args += [flag, str(value)]
    if not stop:
        engine_args.append("--no-target-stop")
    elif args.target_stop == "enabled" or original.get("target_stop_scope") == "all_workers":
        engine_args.append("--all-workers-target-stop")
    else:
        engine_args.append("--target-stop")
    if args.dry_run:
        engine_args.append("--dry-run")
    return dict(benchmark=benchmark, variant=args.ablation or "full", engine=engine_name,
                engine_dir=engine_dir, engine_args=engine_args, profile=profile, output=output,
                overrides=overrides, report_profile_unchanged=not overrides)


def load_engine(directory):
    directory = directory.resolve()
    loaded = sys.modules.get("improved.RA_Q_RBHA_FCC_v2.run_ra_q_rbha_fcc")
    if loaded is not None and directory not in Path(loaded.__file__).resolve().parents:
        raise RuntimeError("Use a separate Python process for each engine variant")
    sys.path.insert(0, str(directory))
    return importlib.import_module("improved.RA_Q_RBHA_FCC_v2.run_ra_q_rbha_fcc")


def execute(args):
    plan = make_plan(args)
    multiprocessing.freeze_support()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    engine = load_engine(plan["engine_dir"])
    parsed, effective, inherited = engine.parse_config(plan["engine_args"])
    identity = dict(method="Contact-Guided Multi-Block Heat-Bath Annealing", short_name="CMHA",
        software_version=VERSION, benchmark=plan["benchmark"], variant=plan["variant"],
        profile_sha256=hashlib.sha256(plan["profile"].read_bytes()).hexdigest(),
        profile=plan["profile"].relative_to(ROOT).as_posix() if plan["profile"].is_relative_to(ROOT) else str(plan["profile"]),
        overrides=plan["overrides"], reporting_profile_unchanged=plan["report_profile_unchanged"],
        output_directory=str(plan["output"]), public_argv=[sys.executable, *sys.argv])
    if args.dry_run:
        print(json.dumps(dict(identity=identity, effective_config=effective), indent=2))
        return 0
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(engine.__file__).resolve()), *plan["engine_args"]]
        engine.run(parsed, effective, inherited)
    finally:
        sys.argv = old_argv
        if plan["output"].is_dir():
            (plan["output"] / "cmha_invocation.json").write_text(json.dumps(identity, indent=2) + "\n")
    from cmha_results import write_summary
    summary = write_summary(plan["output"])
    print(f"CMHA validated result: {summary['best_HH']} contacts; {plan['output']}")
    return 0


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.list_benchmarks:
        for name in primary_names():
            row = read_json(ROOT / "profiles/primary" / f"{name}.json")
            print(f"{name:12} N={len(row['seq']):3} L={row['l_size']:2} workers={row['workers']:2} budget={row['wall_seconds']:7g}s target_stop={row['target_stop_enabled']}")
        return 0
    if args.list_ablations:
        print("\n".join(ablation_names()))
        return 0
    try:
        return execute(args)
    except (ValueError, FileNotFoundError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
