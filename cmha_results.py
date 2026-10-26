"""Independent geometric/contact validation and descriptive run summaries."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

FCC_DELTAS = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                   for z in (-1, 0, 1) if x*x + y*y + z*z == 2)


def validate_fold(sequence, coordinates, side, expected=None):
    if not sequence or set(sequence) - {"H", "P"}:
        raise ValueError("Sequence must contain only H and P")
    if not isinstance(side, int) or side < 2 or len(coordinates) != len(sequence):
        raise ValueError("Invalid lattice side or coordinate count")
    points = []
    for point in coordinates:
        if len(point) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or int(x) != x for x in point):
            raise ValueError("Coordinates must be finite integer triples")
        point = tuple(map(int, point))
        if any(x < 0 or x >= side for x in point) or sum(point) % 2:
            raise ValueError("Coordinate outside the FCC box")
        points.append(point)
    if len(set(points)) != len(points):
        raise ValueError("Two residues occupy the same site")
    if any(sum((x-y)**2 for x, y in zip(a, b)) != 2 for a, b in zip(points, points[1:])):
        raise ValueError("Chain link is not an FCC nearest-neighbor bond")
    contacts = sum(sequence[i] == sequence[j] == "H" and
        sum((x-y)**2 for x, y in zip(points[i], points[j])) == 2
        for i in range(len(points)-2) for j in range(i+2, len(points)))
    occupied = {point: i for i, point in enumerate(points)}
    spatial = 0
    for i, (x, y, z) in enumerate(points):
        if sequence[i] == "H":
            for dx, dy, dz in FCC_DELTAS:
                j = occupied.get((x+dx, y+dy, z+dz), -1)
                spatial += j > i+1 and sequence[j] == "H"
    if contacts != spatial or expected is not None and contacts != expected:
        raise ValueError("Independent contact counts disagree with the saved score")
    return dict(valid=True, HH=contacts, HP_energy=-contacts, residues=len(points))


def summarize_run(folder):
    folder = Path(folder)
    result = folder / "worker_results.json"
    if not result.is_file():
        raise ValueError("No completed worker_results.json; this tool does not read active progress files")
    cfg = json.loads((folder / "run_config.json").read_text(encoding="utf-8-sig"))
    workers = json.loads(result.read_text(encoding="utf-8-sig"))
    count = cfg["workers"]
    if len(workers) != count or {w["worker_id"] for w in workers} != set(range(count)):
        raise ValueError("Missing or duplicated worker records")
    values = []
    for worker in workers:
        validate_fold(cfg["seq"], worker["coords"], cfg["l_size"], worker["best_HH"])
        if worker["best_energy"] != -worker["best_HH"]:
            raise ValueError("Reported HP energy is not minus the nonconsecutive contact count")
        values.append(worker["best_HH"])
    best = max(values)
    first = [w["first_recorded"][str(best)]["recorded_wall_seconds"] for w in workers
             if str(best) in w.get("first_recorded", {})]
    target = cfg.get("target_HH")
    hits = sum(v >= target for v in values) if target is not None else None
    identity_path = folder / "cmha_invocation.json"
    identity = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    return dict(method="CMHA", variant=identity.get("variant", "archived_profile"),
        benchmark=cfg["benchmark"], N=len(cfg["seq"]), L=cfg["l_size"],
        workers=count, base_seed=cfg["seed"], best_HH=best, best_HP_energy=-best,
        mean_worker_best_HH=statistics.mean(values),
        sample_SD_worker_best_HH=statistics.stdev(values) if count > 1 else None,
        workers_at_best=values.count(best), target_HH=target, workers_reaching_target=hits,
        worker_target_hit_rate=hits/count if hits is not None else None,
        first_recorded_best_seconds=min(first) if first else None,
        configured_wall_seconds=cfg["wall_seconds"], target_stop_enabled=cfg["target_stop_enabled"],
        statistical_unit="Worker-best within one parallel execution, not repeated complete runs",
        all_worker_folds_validated=True)


def write_summary(folder):
    folder = Path(folder)
    summary = summarize_run(folder)
    (folder / "cmha_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (folder / "cmha_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary
