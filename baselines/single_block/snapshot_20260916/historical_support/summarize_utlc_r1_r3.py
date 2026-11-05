#!/usr/bin/env python3
"""Validate and consolidate full-budget UTLC-Q-RBHA-FCC R1-R3 statistics."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_at_or_above(worker: dict[str, Any], threshold: int) -> float | None:
    values = [
        float(seconds)
        for hh, seconds in worker.get("first_recorded_seconds_by_HH", {}).items()
        if int(hh) >= threshold
    ]
    return min(values) if values else None


def validate_best(best: dict[str, Any], sequence: str, l_size: int) -> int:
    """Independent coordinate-based score, without importing the SA scorer."""
    assert best["seq"] == sequence
    points = [tuple(p) for p in best["coords"]]
    assert len(points) == len(sequence) == len(set(points))
    assert all(len(p) == 3 and all(isinstance(x, int) and 0 <= x < l_size for x in p)
               and sum(p) % 2 == 0 for p in points)
    def adjacent(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) == 2
    assert all(adjacent(a, b) for a, b in zip(points, points[1:]))
    hh = sum(adjacent(points[i], points[j])
             for i in range(len(sequence)) if sequence[i] == "H"
             for j in range(i + 2, len(sequence)) if sequence[j] == "H")
    assert hh == int(best["HH"]) and bool(best["valid"])
    return hh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks_csv", required=True)
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_stem", required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    with Path(args.benchmarks_csv).open(newline="", encoding="utf-8") as handle:
        benchmarks = list(csv.DictReader(handle))
    runs_dir = Path(args.runs_dir)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for benchmark in benchmarks:
        benchmark_id = benchmark["benchmark"]
        run_dir = runs_dir / benchmark_id / args.run_name
        result_path = run_dir / "q_rbha_result.json"
        workers_path = run_dir / "worker_results.json"
        if not result_path.exists() or not workers_path.exists():
            missing.append(benchmark_id)
            continue

        result = read_json(result_path)
        workers = read_json(workers_path)
        configured_workers = int(result["config"]["workers"])
        if len(workers) != configured_workers:
            raise SystemExit(
                f"{benchmark_id}: expected {configured_workers} workers, found {len(workers)}"
            )
        config = result["config"]
        assert not config["target_stop_enabled"]
        assert config["seq"] == benchmark["hp_sequence"]
        assert config["l_size"] == int(benchmark["l_size"])
        assert {int(w["worker_id"]) for w in workers} == set(range(configured_workers))
        validation = []
        for worker in workers:
            assert worker["seed"] == int(config["seed"]) + int(worker["worker_id"]) * 1000003
            assert not worker["target_stop_enabled"]
            assert float(worker["wall_seconds"]) >= float(config["wall_seconds"]) - 0.05
            hh = validate_best(worker["best"], benchmark["hp_sequence"], config["l_size"])
            validation.append({"worker": worker["worker_id"], "HH": hh, "valid": True})
        saved_best = read_json(run_dir / "best.json")
        assert validate_best(saved_best, benchmark["hp_sequence"], config["l_size"]) == max(w["best"]["HH"] for w in workers)
        (run_dir / "validation_report.json").write_text(json.dumps({
            "all_valid": True, "coordinate_score_independently_recomputed": True,
            "complete_worker_budgets": True, "workers": validation,
        }, indent=2), encoding="utf-8")

        target = abs(int(float(benchmark["target_energy"])))
        worker_best = [int(worker["best"]["HH"]) for worker in workers]
        worker_walls = [float(worker["wall_seconds"]) for worker in workers]
        best_hh = max(worker_best)
        target_times = [
            value
            for worker in workers
            if (value := first_at_or_above(worker, target)) is not None
        ]
        best_times = [
            value
            for worker in workers
            if (value := first_at_or_above(worker, best_hh)) is not None
        ]
        mean_hh = statistics.fmean(worker_best)
        sd_hh = statistics.stdev(worker_best) if len(worker_best) > 1 else 0.0
        successes = sum(value >= target for value in worker_best)

        rows.append({
            "benchmark": benchmark_id,
            "reference_instance": benchmark["reference_instance"],
            "sequence_length": int(benchmark["length"]),
            "reported_length": int(benchmark["reported_length"]),
            "lattice": "3d_fcc",
            "L": int(result["config"]["l_size"]),
            "site_count": int(result["config"]["site_count"]),
            "native_HH_target": target,
            "native_energy_target": -target,
            "IWL_best_HH": abs(int(float(benchmark["iwl_best_energy"]))),
            "IWL_best_energy": int(float(benchmark["iwl_best_energy"])),
            "IWL_mean_energy": float(benchmark["iwl_mean_energy"]),
            "best_HH": best_hh,
            "best_energy": -best_hh,
            "mean_worker_best_HH": round(mean_hh, 6),
            "mean_worker_best_energy": round(-mean_hh, 6),
            "sample_SD_worker_best_HH": round(sd_hh, 6),
            "median_worker_best_HH": round(float(statistics.median(worker_best)), 6),
            "minimum_worker_best_HH": min(worker_best),
            "gap_to_native_contacts": target - best_hh,
            "contacts_vs_IWL_best": best_hh - abs(int(float(benchmark["iwl_best_energy"]))),
            "native_attainment_pct": round(100.0 * best_hh / target, 6),
            "workers_reaching_native": successes,
            "worker_success_rate_pct": round(100.0 * successes / len(worker_best), 6),
            "workers_recorded": len(worker_best),
            "earliest_native_hit_seconds": round(min(target_times), 6) if target_times else "",
            "earliest_best_HH_seconds": round(min(best_times), 6) if best_times else "",
            "configured_wall_seconds_per_worker": float(result["config"]["wall_seconds"]),
            "mean_worker_wall_seconds": round(statistics.fmean(worker_walls), 6),
            "batch_wall_seconds": round(float(result["wall_seconds"]), 6),
            "total_worker_steps": int(result["total_worker_steps"]),
            "total_transitions": int(result["total_transitions"]),
            "base_seed": int(result["config"]["seed"]),
            "target_stop_enabled": bool(result["config"]["target_stop_enabled"]),
            "sequence_sha256": hashlib.sha256(benchmark["hp_sequence"].encode("ascii")).hexdigest(),
            "statistical_unit": "independent worker-best fold (full wall-time budget)",
            "all_worker_best_folds_valid": all(bool(worker["best"]["valid"]) for worker in workers),
            "run_dir": str(run_dir),
        })

    if args.require_complete and missing:
        raise SystemExit("Missing completed results: " + ", ".join(missing))

    csv_path = runs_dir / f"{args.output_stem}.csv"
    json_path = runs_dir / f"{args.output_stem}.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(json.dumps({
        "suite": "R1-R3 200-residue FCC HP benchmarks",
        "algorithm": "UTLC-Q-RBHA-FCC",
        "statistical_unit": "independent worker-best fold after the complete wall-time budget",
        "run_name": args.run_name,
        "first_hit_time_definition": "Seconds from each worker's annealing clock; excludes parent initialization. Recorded at kernel-chunk boundaries.",
        "completed": len(rows),
        "missing": missing,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "completed": len(rows),
        "missing": missing,
        "csv": str(csv_path),
        "json": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
