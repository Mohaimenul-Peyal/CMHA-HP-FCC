"""Run the matched six-hour R1-R3 L=19 experiment with the frozen UTLC kernel."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_NAME = "UTLC_Q_RBHA_FCC_R1_R3_L19_fullbudget_21600s_seed21013001"
MANIFEST = "fcc_r1_r3_utlc_benchmarks.csv"
BASELINE_FILES = (
    "run_hp3d_fcc_utlc_q_rbha.py", "export_3d_FCC_HP_cluster_qubo_light.py",
    "fcc_lattice.py", "hp3d_stage_common.py", "rb_qubo_3d_common.py",
    "plot_fcc_hp_fold.py", "validate_fcc_fold.py", "requirements.txt",
)
R_FILES = (MANIFEST, "run_r1_r3_utlc_fullbudget.py", "summarize_utlc_r1_r3.py",
           "verify_r_suite_sources.py", "r_suite_sequence_verification.json", "README_R1_R3.md",
           "lattice_change_verification.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ps_command(argv):
    return "& " + " ".join("'" + str(value).replace("'", "''") + "'" for value in argv) + "\n"


def load_rows(path, only=""):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [r["benchmark"] for r in rows] != ["R1FCC", "R2FCC", "R3FCC"]:
        raise ValueError("Expected R1FCC, R2FCC, R3FCC in order")
    for row in rows:
        if int(row["l_size"]) != 19:
            raise ValueError("This comparison requires L=19 for every benchmark")
        seq = row["hp_sequence"]
        if len(seq) != 200 or seq.count("H") != 100 or set(seq) != {"H", "P"}:
            raise ValueError(f"Invalid R-suite sequence: {row['benchmark']}")
    selected = set(only.split(",")) if only else {r["benchmark"] for r in rows}
    if selected - {r["benchmark"] for r in rows}:
        raise ValueError("Unknown --only benchmark")
    return [row for row in rows if row["benchmark"] in selected]


def solver_command(row, args, snapshot):
    return [sys.executable, "-u", str(snapshot / BASELINE_FILES[0]),
            "--seq", row["hp_sequence"], "--target_energy", row["target_energy"],
            "--benchmark", row["benchmark"], "--out_dir", str(args.runs_dir),
            "--run_name", args.run_name, "--workers", str(args.workers),
            "--wall_seconds", str(args.wall_seconds), "--disable_target_stop",
            "--l_size", row["l_size"], "--cycle_steps", "80000,160000,320000,640000",
            "--cycles_without_improvement", "2", "--control_chunk_steps", "8",
            "--beam_width", "24", "--near_target_beam_width", "48",
            "--near_target_gap", row["near_target_gap"], "--min_block_len", "2",
            "--max_block_len", "12", "--max_block_len_cap", row["max_block_len_cap"],
            "--cluster_multipliers", "1.0,0.66,0.33,0.0", "--pair_hh_reward", "-2.0",
            "--cluster_excess_reward", "-0.5", "--cluster_threshold", "2",
            "--t0", "8.0", "--t1", "0.02", "--seed", "21013001"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[6] / "runs")
    parser.add_argument("--run-name", default=DEFAULT_NAME)
    parser.add_argument("--wall-seconds", type=float, default=21600)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only", default="")
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and print commands without launching")
    args = parser.parse_args()
    if args.wall_seconds <= 0 or not 1 <= args.workers <= 16:
        parser.error("Positive wall budget and 1-16 workers required")
    if not args.run_name or any(c in args.run_name for c in '/\\:') or args.run_name in (".", ".."):
        parser.error("--run-name must be a single safe folder name")
    args.runs_dir = args.runs_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    rows = load_rows(HERE / MANIFEST, args.only)
    control = args.runs_dir / (args.run_name + "_batch_control")
    run_dirs = [args.runs_dir / row["benchmark"] / args.run_name for row in rows]
    if control.exists() or any(path.exists() for path in run_dirs):
        raise FileExistsError("Output exists. Choose a new --run-name; previous data are never overwritten.")
    for name in BASELINE_FILES:
        if not (args.source_dir / name).is_file():
            raise FileNotFoundError(args.source_dir / name)
    for name in R_FILES:
        if not (HERE / name).is_file():
            raise FileNotFoundError(HERE / name)
    verification = json.loads((HERE / "r_suite_sequence_verification.json").read_text(encoding="utf-8"))
    if hashlib.sha256((HERE / MANIFEST).read_bytes()).hexdigest() != verification["manifest_sha256"]:
        raise ValueError("Manifest no longer matches the source verification report")

    audit = json.loads((HERE / "lattice_change_verification.json").read_text(encoding="utf-8"))
    for name in BASELINE_FILES:
        if hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest() != audit["unchanged_solver_sha256"][name]:
            raise ValueError(f"Solver source differs from the six-hour baseline: {name}")
    if args.dry_run:
        print(json.dumps({"verified": True, "baseline": audit["baseline_run_name"],
                          "run_name": args.run_name, "L": 19, "site_count": 3430,
                          "commands": [solver_command(row, args, control / "source_snapshot") for row in rows]}, indent=2))
        return

    snapshot = control / "source_snapshot"
    snapshot.mkdir(parents=True)
    hashes = {}
    for name in BASELINE_FILES + R_FILES:
        source = (args.source_dir if name in BASELINE_FILES else HERE) / name
        shutil.copy2(source, snapshot / name)
        hashes[name] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    write_json(control / "source_sha256.json", hashes)
    shutil.copy2(HERE / MANIFEST, control / MANIFEST)
    batch_argv = [sys.executable, "-u", str(snapshot / Path(__file__).name),
                  "--runs-dir", str(args.runs_dir), "--run-name", args.run_name,
                  "--workers", str(args.workers), "--wall-seconds", str(args.wall_seconds)]
    if args.only:
        batch_argv += ["--only", args.only]
    (control / "batch_run_command.ps1").write_text(
        "# Exact original command; use a new --run-name for another experiment.\n" + ps_command(batch_argv), encoding="utf-8")
    status = {"algorithm": "UTLC-Q-RBHA-FCC", "pid": os.getpid(), "started_utc": utc_now(),
              "state": "initializing", "run_name": args.run_name,
              "workers": args.workers, "wall_seconds_per_worker": args.wall_seconds,
              "python": sys.executable, "python_version": sys.version,
              "completed": [], "queued": [r["benchmark"] for r in rows]}
    write_json(control / "batch_status.json", status)
    commands = []
    for row, run_dir in zip(rows, run_dirs):
        run_dir.mkdir(parents=True)
        command = solver_command(row, args, snapshot)
        commands.append(command)
        (run_dir / "launch_command.ps1").write_text(ps_command(command), encoding="utf-8")
        write_json(run_dir / "experiment_manifest.json", {
            "benchmark": row, "command_argv": command, "source_snapshot": str(snapshot),
            "source_sha256": hashes, "sequence_verification": verification,
            "lattice_rationale": "Matched lattice-size test against the completed six-hour L=17 run. L=19 has 3430 even-parity FCC sites. The 21600-second budget, sequences, targets, seeds, initialization and all other solver settings remain unchanged.",
            "lattice_change_verification": audit,
        })
    try:
        for row, run_dir, command in zip(rows, run_dirs, commands):
            status.update(state="running", current=row["benchmark"], current_started_utc=utc_now())
            status["queued"].remove(row["benchmark"])
            write_json(control / "batch_status.json", status)
            print(f"[start] {row['benchmark']} N=200 L={row['l_size']} workers={args.workers} wall={args.wall_seconds}s", flush=True)
            with (run_dir / "solver_stdout.log").open("x", encoding="utf-8") as out, (run_dir / "solver_stderr.log").open("x", encoding="utf-8") as err:
                process = subprocess.Popen(command, cwd=snapshot, stdout=out, stderr=err)
                status["solver_pid"] = process.pid
                write_json(control / "batch_status.json", status)
                exit_code = process.wait()
            if exit_code:
                raise RuntimeError(f"{row['benchmark']} solver exited {exit_code}; inspect solver_stderr.log")
            summary_command = [sys.executable, str(snapshot / "summarize_utlc_r1_r3.py"),
                               "--benchmarks_csv", str(snapshot / MANIFEST), "--runs_dir", str(args.runs_dir),
                               "--run_name", args.run_name, "--output_stem", "RESULT_" + args.run_name]
            subprocess.run(summary_command, check=True)
            status["completed"].append(row["benchmark"])
            write_json(control / "batch_status.json", status)
            print(f"[done] {row['benchmark']}; all worker folds validated and statistics updated", flush=True)
        if not args.only:
            subprocess.run(summary_command + ["--require_complete"], check=True)
        status.update(state="completed", current=None, solver_pid=None, finished_utc=utc_now())
        write_json(control / "batch_status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=str(exc), finished_utc=utc_now())
        write_json(control / "batch_status.json", status)
        raise


if __name__ == "__main__":
    main()
