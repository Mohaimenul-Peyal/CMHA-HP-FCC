"""Independently validate the frozen folds and summary statistics; stdlib only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = tuple(
    (x, y, z)
    for x in (-1, 0, 1)
    for y in (-1, 0, 1)
    for z in (-1, 0, 1)
    if x * x + y * y + z * z == 2
)


def validate(seq, lattice, coordinates, expected=None):
    points = [tuple(p) for p in coordinates]
    assert len(points) == len(seq)
    assert len(set(points)) == len(seq), "Site collision"
    for point in points:
        assert len(point) == 3
        assert all(isinstance(x, int) and 0 <= x < lattice for x in point)
        assert sum(point) % 2 == 0, "Not an FCC site"
    for a, b in zip(points, points[1:]):
        assert sum((x - y) ** 2 for x, y in zip(a, b)) == 2, "Broken link"
    pair_count = sum(
        seq[i] == seq[j] == "H"
        and sum((x - y) ** 2 for x, y in zip(points[i], points[j])) == 2
        for i in range(len(seq) - 2)
        for j in range(i + 2, len(seq))
    )
    occupied = {p: i for i, p in enumerate(points)}
    spatial_count = 0
    for i, (x, y, z) in enumerate(points):
        if seq[i] != "H":
            continue
        for dx, dy, dz in DIRECTIONS:
            j = occupied.get((x + dx, y + dy, z + dz), -1)
            if j > i + 1 and seq[j] == "H":
                spatial_count += 1
    assert pair_count == spatial_count, "Independent contact counters disagree"
    if expected is not None:
        assert pair_count == expected, "Saved contact score mismatch"


def main():
    data = json.loads((ROOT / "data/paper_data.json").read_text())
    primary = {r["benchmark"]: r for r in data["primary"]}
    grouped = {}
    counts = {"primary": 0, "ablation": 0}
    with (ROOT / "results/worker_best_folds.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            bench = primary[row["benchmark"]]
            assert hashlib.sha256(bench["seq"].encode()).hexdigest() == bench["sequence_sha256"]
            validate(bench["seq"], bench["L"], json.loads(row["coords_json"]), int(row["best_HH"]))
            key = row["study"], row["variant"], row["benchmark"]
            grouped.setdefault(key, []).append(int(row["best_HH"]))
            counts[row["study"]] += 1
    for row in data["primary"]:
        check_summary(row, grouped[("primary", "full", row["benchmark"])])
    for row in data["ablations"]:
        check_summary(row, grouped[("ablation", row["variant"], row["benchmark"])])
    folds = json.loads((ROOT / "data/selected_folds.json").read_text())
    for benchmark, fold in folds.items():
        validate(fold["seq"], fold["L"], fold["coords"], fold["HH"])
        assert fold["HH"] == primary[benchmark]["best_HH"]
    initials = json.loads((ROOT / "data/initial_saw_pools.json").read_text())
    initial_by_benchmark = {}
    for pool in initials:
        for walk in pool["initial_coordinates"]:
            validate(pool["seq"], pool["L"], walk)
        old = initial_by_benchmark.setdefault(pool["benchmark"], pool["initial_coordinates"])
        assert old == pool["initial_coordinates"], "Matched initial walks differ"
    assert counts == {"primary": 496, "ablation": 448}
    report = {
        "all_checks_passed": True,
        "worker_folds": counts,
        "primary_best_folds": len(folds),
        "summary_groups": len(grouped),
        "initial_walks": sum(len(x["initial_coordinates"]) for x in initials),
        "matched_initial_walks_identical": True,
        "checks": ["integrality", "FCC parity", "box bounds", "site exclusivity",
                   "chain connectivity", "pair/spatial contact agreement", "sequence hashes",
                   "best/mean/sample-SD agreement"],
    }
    destination = ROOT / "runs/validation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "archived_results_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def check_summary(row, values):
    assert len(values) == 16
    assert max(values) == row["best_HH"]
    assert abs(statistics.mean(values) - row["mean_HH"]) < 1e-8
    assert abs(statistics.stdev(values) - row["SD_HH"]) < 1e-8
    assert sorted(values) == sorted(row["worker_best_HH"])


if __name__ == "__main__":
    main()
