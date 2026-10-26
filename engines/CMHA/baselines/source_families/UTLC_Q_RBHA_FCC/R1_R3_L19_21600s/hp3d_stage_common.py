#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for separated 3D HP Stage 1/2/3 runners.

This module is intentionally HP-only.  It still uses rb_qubo_3d_common.py as the
source of truth for model export, model loading, feasible-chain validation,
energy/QUBO auditing, and standard lattice moves.

Design goals:
  * Stage 1 can solve easier benchmarks using pairwise-contact-aware SAW starts.
  * Stage 2 refines Stage 1 folds with H-bounded segment repair.
  * Stage 3 performs hotter escape/deep repair for hard 50mer/60mer cases.
  * Every result is auditable against the exported RB-QUBO model.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback for environments without numba
    def njit(*_args, **_kwargs):
        def deco(fn):
            return fn
        return deco

SOURCE_DIR = Path(__file__).resolve().parent
PARENT_SOURCE_DIR = SOURCE_DIR.parent
if str(PARENT_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SOURCE_DIR))

from rb_qubo_3d_common import (
    bridge_regrow,
    export_model,
    fold_energy,
    load_model,
    local_move,
    pivot_move,
    qubo_value_for_pos,
    random_saw,
    regrow_tail,
    reptation_move,
    validate_pos,
)

# True-3D move operators are embedded below.  They are intentionally not imported
# from experiment scratch scripts, so Stage 2/3 cannot silently degrade when an
# external file is missing.

HP_MATRIX = {"H": {"H": -1.0, "P": 0.0}, "P": {"H": 0.0, "P": 0.0}}


@njit(cache=True)
def hh_contacts_numba(pos: np.ndarray, hmask: np.ndarray, nbr_bool: np.ndarray) -> int:
    n = pos.shape[0]
    c = 0
    for i in range(n):
        if hmask[i] == 0:
            continue
        si = pos[i]
        for j in range(i + 2, n):
            if hmask[j] != 0 and nbr_bool[si, pos[j]]:
                c += 1
    return c


@njit(cache=True)
def valid_chain_numba(pos: np.ndarray, nbr_bool: np.ndarray) -> bool:
    n = pos.shape[0]
    s_count = nbr_bool.shape[0]
    occupied = np.zeros(s_count, dtype=np.bool_)
    for i in range(n):
        s = pos[i]
        if s < 0 or s >= s_count:
            return False
        if occupied[s]:
            return False
        occupied[s] = True
    for i in range(n - 1):
        if not nbr_bool[pos[i], pos[i + 1]]:
            return False
    return True


@dataclass
class BenchmarkRow:
    benchmark: str
    length: int
    hp_sequence: str
    target_energy: float


def hp_energy_from_hh(hh: int) -> float:
    return -float(hh)


def target_hh(target_energy: float | int | str | None) -> int | None:
    if target_energy is None:
        return None
    return int(abs(float(target_energy)))


def hmask_from_seq(seq: str) -> np.ndarray:
    return np.asarray([1 if a == "H" else 0 for a in seq], dtype=np.int8)


def nbr_bool_from_sets(nbr_sets: list[set[int]]) -> np.ndarray:
    n = len(nbr_sets)
    out = np.zeros((n, n), dtype=np.bool_)
    for i, ns in enumerate(nbr_sets):
        for j in ns:
            out[i, j] = True
    return out


def hh_contacts(pos: list[int], hmask: np.ndarray, nbr_bool: np.ndarray) -> int:
    return int(hh_contacts_numba(np.asarray(pos, dtype=np.int64), hmask, nbr_bool))


def valid_chain_fast(pos: list[int], nbr_bool: np.ndarray) -> bool:
    return bool(valid_chain_numba(np.asarray(pos, dtype=np.int64), nbr_bool))


def contact_set(pos: list[int], seq: str, nbr_sets: list[set[int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i, ai in enumerate(seq):
        if ai != "H":
            continue
        for j in range(i + 2, len(seq)):
            if seq[j] == "H" and pos[j] in nbr_sets[pos[i]]:
                out.add((i, j))
    return out




def choose_scored_move(candidates: list[list[int]], seq: str, nbr_sets, matrix, rng: random.Random) -> list[int] | None:
    if not candidates:
        return None
    scored: list[tuple[float, list[int]]] = []
    for cand in candidates:
        e, _counts = fold_energy(cand, seq, nbr_sets, matrix)
        scored.append((float(e) + rng.random() * 0.02, cand))
    scored.sort(key=lambda x: x[0])
    return rng.choice(scored[: min(6, len(scored))])[1]


def pull_through_move(pos: list[int], seq: str, nbr_sets, matrix, rng: random.Random) -> list[int] | None:
    n = len(pos)
    candidates: list[list[int]] = []
    for _ in range(16):
        i = rng.randrange(1, n - 1)
        if rng.random() < 0.5:
            available = (nbr_sets[pos[i - 1]] & nbr_sets[pos[i]]) - (set(pos) - {pos[-1]})
            for q in available:
                new_pos = pos[:i] + [q] + pos[i:n - 1]
                if validate_pos(new_pos, nbr_sets):
                    candidates.append(new_pos)
        else:
            available = (nbr_sets[pos[i]] & nbr_sets[pos[i + 1]]) - (set(pos) - {pos[0]})
            for q in available:
                new_pos = pos[1:i + 1] + [q] + pos[i + 1:]
                if validate_pos(new_pos, nbr_sets):
                    candidates.append(new_pos)
        if candidates:
            break
    return choose_scored_move(candidates, seq, nbr_sets, matrix, rng)


def contact_guided_pull(pos: list[int], seq: str, nbr_sets, matrix, rng: random.Random) -> list[int] | None:
    current = contact_set(pos, seq, nbr_sets)
    hs = [i for i, a in enumerate(seq) if a == "H"]
    missing = [(i, j) for ix, i in enumerate(hs) for j in hs[ix + 1:] if abs(i - j) > 1 and (i, j) not in current]
    rng.shuffle(missing)
    candidates: list[list[int]] = []
    occupied = set(pos)
    for i, j in missing[:48]:
        for moving, fixed in ((i, j), (j, i)):
            if moving < fixed:
                if moving <= 0 or moving >= len(pos) - 1:
                    continue
                available = (nbr_sets[pos[moving]] & nbr_sets[pos[moving + 1]] & nbr_sets[pos[fixed]]) - (occupied - {pos[0]})
                for q in available:
                    new_pos = pos[1:moving + 1] + [q] + pos[moving + 1:]
                    if validate_pos(new_pos, nbr_sets):
                        candidates.append(new_pos)
            else:
                if moving <= 0 or moving >= len(pos) - 1:
                    continue
                available = (nbr_sets[pos[moving - 1]] & nbr_sets[pos[moving]] & nbr_sets[pos[fixed]]) - (occupied - {pos[-1]})
                for q in available:
                    new_pos = pos[:moving] + [q] + pos[moving:len(pos) - 1]
                    if validate_pos(new_pos, nbr_sets):
                        candidates.append(new_pos)
        if candidates:
            break
    return choose_scored_move(candidates, seq, nbr_sets, matrix, rng)


def pair_crankshaft_move(pos: list[int], seq: str, nbr_sets, matrix, rng: random.Random) -> list[int] | None:
    n = len(pos)
    candidates: list[list[int]] = []
    for _ in range(16):
        i = rng.randrange(1, n - 2)
        fixed = set(pos[:i] + pos[i + 2:])
        for a in nbr_sets[pos[i - 1]] - fixed:
            for b in (nbr_sets[a] & nbr_sets[pos[i + 2]]) - fixed - {a}:
                new_pos = list(pos)
                new_pos[i] = a
                new_pos[i + 1] = b
                if validate_pos(new_pos, nbr_sets):
                    candidates.append(new_pos)
        if candidates:
            break
    return choose_scored_move(candidates, seq, nbr_sets, matrix, rng)


def rotate_about_axis(axis: int, origin: tuple[int, int, int], p: tuple[int, int, int], turns: int) -> tuple[int, int, int]:
    x, y, z = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
    for _ in range(turns % 4):
        if axis == 0:
            y, z = -z, y
        elif axis == 1:
            x, z = z, -x
        else:
            x, y = -y, x
    return (origin[0] + x, origin[1] + y, origin[2] + z)


def axis_crankshaft_move(pos: list[int], coords, coord_to_site, nbr_sets, seq, matrix, rng: random.Random) -> list[int] | None:
    n = len(pos)
    candidates: list[list[int]] = []
    for _ in range(32):
        i = rng.randrange(0, n - 4)
        j = rng.randrange(i + 3, min(n, i + 14))
        a = coords[pos[i]]
        b = coords[pos[j]]
        same = [k for k in range(3) if a[k] == b[k]]
        if len(same) != 2:
            continue
        axis = ({0, 1, 2} - set(same)).pop()
        for turns in (1, 2, 3):
            new_pos = list(pos)
            ok = True
            for r in range(i + 1, j):
                c = rotate_about_axis(axis, a, coords[pos[r]], turns)
                site = coord_to_site.get(c)
                if site is None:
                    ok = False
                    break
                new_pos[r] = site
            if ok and validate_pos(new_pos, nbr_sets):
                candidates.append(new_pos)
        if candidates:
            break
    return choose_scored_move(candidates, seq, nbr_sets, matrix, rng)


def true3d_double_regrow(pos: list[int], nbr_sets, rng: random.Random, max_len: int) -> list[int] | None:
    first = bridge_regrow(pos, nbr_sets, rng, max(4, max_len // 2))
    if first is None:
        return None
    return bridge_regrow(first, nbr_sets, rng, max(4, max_len // 2))


def normalized_key(pos: list[int], coords: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    pts = [coords[s] for s in pos]
    mins = tuple(min(c[k] for c in pts) for k in range(3))
    return tuple((c[0] - mins[0], c[1] - mins[1], c[2] - mins[2]) for c in pts)


def manhattan(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def load_rows(path: Path, only: str = "") -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            bench = raw.get("benchmark") or raw.get("id") or raw.get("name")
            seq = raw.get("hp_sequence") or raw.get("seq") or raw.get("sequence")
            target = raw.get("target_energy") or raw.get("target")
            if not bench or not seq or target is None:
                raise SystemExit("CSV must contain benchmark, hp_sequence, and target_energy columns")
            n = int(raw.get("length") or len(seq))
            if wanted and bench not in wanted and str(n) not in wanted:
                continue
            rows.append(BenchmarkRow(bench, n, seq.strip().upper(), float(target)))
    return rows


def direct_row(seq: str, target_energy: float, benchmark: str = "direct") -> BenchmarkRow:
    seq = seq.strip().upper()
    return BenchmarkRow(benchmark=benchmark, length=len(seq), hp_sequence=seq, target_energy=float(target_energy))


def export_model_if_needed(row: BenchmarkRow, run_dir: Path, l_size: int, force: bool = False) -> Path:
    model_json = run_dir / f"model_HP_L{l_size}.json"
    if force or not model_json.exists():
        export_model("HP", row.hp_sequence, model_json, l_size=l_size)
    return model_json


def model_context(model_json: Path):
    data, coords, coord_to_site, nbr_sets = load_model(model_json, "HP")
    nbr_bool = nbr_bool_from_sets(nbr_sets)
    hmask = hmask_from_seq(data["seq"])
    # Compile numba kernels early in each process.
    if len(coords) >= 3:
        _ = hh_contacts_numba(np.asarray([0, 1, 2], dtype=np.int64), np.asarray([1, 1, 1], dtype=np.int8), nbr_bool)
        _ = valid_chain_numba(np.asarray([0, 1, 2], dtype=np.int64), nbr_bool)
    return data, coords, coord_to_site, nbr_sets, nbr_bool, hmask


def qubo_audit(data: dict[str, Any], pos: list[int], energy: float, nbr_sets: list[set[int]]) -> dict[str, Any]:
    valid = validate_pos(pos, nbr_sets)
    q = qubo_value_for_pos(data, pos)
    expected = -float(data["scaled_penalties"]["lambda_onehot"]) * int(data["N"]) + float(data["scale_alpha"]) * float(energy)
    return {
        "valid": bool(valid),
        "active_qubo_value": float(q),
        "expected_feasible_qubo_value": float(expected),
        "abs_error": float(abs(q - expected)),
    }


def archive_entry(
    pos: list[int],
    seq: str,
    coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]],
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    source: str,
) -> dict[str, Any] | None:
    if not valid_chain_fast(pos, nbr_bool):
        return None
    hh = hh_contacts(pos, hmask, nbr_bool)
    contacts = sorted(contact_set(pos, seq, nbr_sets))
    key = normalized_key(pos, coords)
    return {
        "energy": -float(hh),
        "HH": int(hh),
        "score": int(hh),
        "sites": [int(x) for x in pos],
        "pos": [int(x) for x in pos],
        "coords": [list(coords[s]) for s in pos],
        "contacts": [list(x) for x in contacts],
        "n_contacts": int(hh),
        "key": [list(x) for x in key],
        "source": source,
    }


def merge_archive(entries: list[dict[str, Any]], limit: int, diversity_jaccard: float = 0.91) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[tuple[int, int, int], ...], dict[str, Any]] = {}
    for e in entries:
        if not e:
            continue
        key_data = e.get("key")
        if key_data is None and "sites" in e and "coords" in e:
            key_data = e["coords"]
        key = tuple(tuple(map(int, x)) for x in key_data)
        old = best_by_key.get(key)
        if old is None or float(e["energy"]) < float(old["energy"]):
            best_by_key[key] = e
    sorted_entries = sorted(best_by_key.values(), key=lambda e: (float(e["energy"]), -int(e.get("HH", e.get("n_contacts", 0)))))
    selected: list[dict[str, Any]] = []
    selected_contacts: list[set[tuple[int, int]]] = []
    for e in sorted_entries:
        contacts = {tuple(map(int, x)) for x in e.get("contacts", [])}
        if len(selected) < 32:
            selected.append(e)
            selected_contacts.append(contacts)
        else:
            keep = False
            for old in selected_contacts[:512]:
                union = max(1, len(contacts | old))
                if len(contacts & old) / union < diversity_jaccard:
                    keep = True
                    break
            if keep:
                selected.append(e)
                selected_contacts.append(contacts)
        if len(selected) >= limit:
            break
    return selected


def entry_to_pos(entry: dict[str, Any], coord_to_site: dict[tuple[int, int, int], int], n: int, s_count: int) -> list[int] | None:
    if "sites" in entry:
        pos = [int(x) for x in entry["sites"]]
        if len(pos) == n and all(0 <= s < s_count for s in pos):
            return pos
    if "pos" in entry:
        pos = [int(x) for x in entry["pos"]]
        if len(pos) == n and all(0 <= s < s_count for s in pos):
            return pos
    if "coords" in entry:
        pos = []
        for c in entry["coords"]:
            site = coord_to_site.get(tuple(map(int, c)))
            if site is None:
                return None
            pos.append(site)
        if len(pos) == n:
            return pos
    return None


def load_archive_json(path: Path, coord_to_site: dict[tuple[int, int, int], int], n: int, s_count: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    if isinstance(raw, list):
        candidates.extend([x for x in raw if isinstance(x, dict)])
    elif isinstance(raw, dict):
        if isinstance(raw.get("archive"), list):
            candidates.extend([x for x in raw["archive"] if isinstance(x, dict)])
        if isinstance(raw.get("best"), dict):
            candidates.append(raw["best"])
        if isinstance(raw.get("worker_results"), list):
            for wr in raw["worker_results"]:
                if isinstance(wr, dict):
                    if isinstance(wr.get("best"), dict):
                        candidates.append(wr["best"])
                    if isinstance(wr.get("archive"), list):
                        candidates.extend([x for x in wr["archive"] if isinstance(x, dict)])
        if not candidates:
            candidates.append(raw)
    out: list[dict[str, Any]] = []
    for c in candidates:
        pos = entry_to_pos(c, coord_to_site, n, s_count)
        if pos is None:
            continue
        c = dict(c)
        c["sites"] = [int(x) for x in pos]
        c["pos"] = [int(x) for x in pos]
        out.append(c)
    return out


def public_best(entry: dict[str, Any], seq: str, data: dict[str, Any], nbr_sets: list[set[int]]) -> dict[str, Any]:
    pos = [int(x) for x in entry.get("sites", entry.get("pos"))]
    energy, counts = fold_energy(pos, seq, nbr_sets, data["contact_matrix"])
    return {
        "format": "hp3d_stage_best_v1",
        "model": "HP",
        "seq": seq,
        "N": len(seq),
        "L": int(data["L"]),
        "S": int(data["S"]),
        "dimension": 3,
        "energy": float(energy),
        "score": float(-energy),
        "HH": int(counts.get("HH", 0)),
        "contact_counts": counts,
        "valid": validate_pos(pos, nbr_sets),
        "sites": pos,
        "coords": entry["coords"],
        "contacts": entry.get("contacts", []),
        "qubo_check": qubo_audit(data, pos, energy, nbr_sets),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def lattice_center(coords: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    return tuple(max(p[k] for p in coords) / 2.0 for k in range(3))  # type: ignore[return-value]


def placed_h_score(site: int, residue: int, placed: list[int], seq: str, coords, nbr_sets, center: tuple[float, float, float]) -> float:
    c = coords[site]
    score = 0.010 * ((c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2 + (c[2] - center[2]) ** 2)
    if seq[residue] != "H":
        return score
    for j, sj in enumerate(placed):
        if seq[j] != "H" or residue - j <= 1:
            continue
        if sj in nbr_sets[site]:
            score -= 1.35
        else:
            score += 0.025 * min(8, manhattan(coords[site], coords[sj]))
    return score


def pairwise_contact_saw(
    seq: str,
    coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]],
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    rng: random.Random,
    attempts: int = 800,
    h_pool: int = 2,
    p_pool: int = 4,
    random_pool_prob: float = 0.10,
) -> list[int] | None:
    center = lattice_center(coords)
    starts = list(range(len(coords)))
    rng.shuffle(starts)
    starts.sort(key=lambda s: sum((coords[s][k] - center[k]) ** 2 for k in range(3)))
    for attempt in range(max(1, attempts)):
        pos = [starts[attempt % len(starts)] if attempt < 2 * len(starts) else rng.randrange(len(coords))]
        used = {pos[0]}
        ok = True
        for r in range(1, len(seq)):
            candidates = [s for s in nbr_sets[pos[-1]] if s not in used]
            if not candidates:
                ok = False
                break
            scored = []
            for s in candidates:
                free_deg = len(nbr_sets[s] - used)
                sc = placed_h_score(s, r, pos, seq, coords, nbr_sets, center) - 0.025 * free_deg + rng.random() * 0.06
                scored.append((sc, s))
            scored.sort(key=lambda x: x[0])
            if rng.random() < random_pool_prob:
                width = min(len(scored), max(2, h_pool + p_pool))
            else:
                width = h_pool if seq[r] == "H" else p_pool
            chosen = rng.choice(scored[: max(1, min(width, len(scored)))])[1]
            pos.append(chosen)
            used.add(chosen)
        if ok and valid_chain_fast(pos, nbr_bool):
            return pos
    return None




def generate_unique_saw_pool(
    seq: str,
    coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]],
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    rng: random.Random,
    count: int,
    init_attempts: int,
    max_calls: int | None = None,
    h_pool: int = 2,
    p_pool: int = 4,
    random_pool_prob: float = 0.16,
) -> list[list[int]]:
    """Generate a pool of structurally unique randomized 3D SAW starts.

    Uniqueness is tested using the normalized 3D chain key, so translated copies
    of the same walk are not counted as different starts.  This function is used
    by Stage 1 to guarantee that trial k starts from SAW k, rather than creating
    starts opportunistically inside the worker loop.
    """
    target = max(0, int(count))
    if target == 0:
        return []
    max_calls = int(max_calls or max(target * 40, target + 64))
    pool: list[list[int]] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    calls = 0
    while len(pool) < target and calls < max_calls:
        calls += 1
        # Slightly vary construction greediness over calls.  This improves
        # diversity without changing the HP objective.
        if calls % 5 == 0:
            hp, pp, rp = max(3, h_pool + 1), max(5, p_pool + 2), min(0.35, random_pool_prob + 0.12)
        elif calls % 7 == 0:
            hp, pp, rp = max(4, h_pool + 2), max(6, p_pool + 3), min(0.45, random_pool_prob + 0.20)
        else:
            hp, pp, rp = h_pool, p_pool, random_pool_prob
        pos = pairwise_contact_saw(
            seq, coords, nbr_sets, hmask, nbr_bool, rng,
            attempts=init_attempts, h_pool=hp, p_pool=pp, random_pool_prob=rp,
        )
        if pos is None:
            continue
        key = normalized_key(pos, coords)
        if key in seen:
            continue
        seen.add(key)
        pool.append([int(x) for x in pos])
    return pool


def saw_pool_entries(
    saw_pool: list[list[int]],
    seq: str,
    coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]],
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    source_prefix: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for i, pos in enumerate(saw_pool):
        e = archive_entry(pos, seq, coords, nbr_sets, hmask, nbr_bool, f"{source_prefix}:saw{i}")
        if e is not None:
            entries.append(e)
    return entries


def choose_archive_start(
    archive: list[dict[str, Any]],
    rng: random.Random,
    coord_to_site: dict[tuple[int, int, int], int],
    n: int,
    s_count: int,
    top_prob: float = 0.70,
    top_k: int = 128,
    broad_k: int = 512,
) -> list[int] | None:
    if not archive:
        return None
    if rng.random() < top_prob:
        pool = archive[: min(top_k, len(archive))]
    else:
        pool = archive[: min(broad_k, len(archive))]
    for _ in range(12):
        e = rng.choice(pool)
        pos = entry_to_pos(e, coord_to_site, n, s_count)
        if pos is not None:
            return pos
    return None


def fallback_double_regrow(pos: list[int], nbr_sets: list[set[int]], rng: random.Random, max_len: int) -> list[int] | None:
    first = bridge_regrow(pos, nbr_sets, rng, max(4, max_len // 2))
    if first is None:
        return None
    return bridge_regrow(first, nbr_sets, rng, max(4, max_len // 2))


def basic_move(
    pos: list[int],
    coords: list[tuple[int, int, int]],
    coord_to_site: dict[tuple[int, int, int], int],
    nbr_sets: list[set[int]],
    rng: random.Random,
    regrow_max_len: int,
    attempts: dict[str, int],
) -> tuple[str, list[int] | None]:
    moves = [
        ("local", 0.12),
        ("pivot", 0.22),
        ("reptation", 0.13),
        ("end_regrow", 0.25),
        ("fragment_regrow", 0.28),
    ]
    name = weighted_choice(moves, rng)
    attempts[name] = attempts.get(name, 0) + 1
    if name == "local":
        return name, local_move(pos, nbr_sets, rng)
    if name == "pivot":
        return name, pivot_move(pos, coords, coord_to_site, nbr_sets, rng)
    if name == "reptation":
        return name, reptation_move(pos, nbr_sets, rng)
    if name == "end_regrow":
        return name, regrow_tail(pos, nbr_sets, rng, regrow_max_len)
    return name, bridge_regrow(pos, nbr_sets, rng, regrow_max_len)


def weighted_choice(items: list[tuple[str, float]], rng: random.Random) -> str:
    total = sum(w for _n, w in items)
    u = rng.random() * total
    acc = 0.0
    for name, weight in items:
        acc += weight
        if u <= acc:
            return name
    return items[-1][0]


def h_anchor_segments(seq: str) -> list[tuple[int, int]]:
    hpos = [i for i, a in enumerate(seq) if a == "H"]
    if len(hpos) < 2:
        return []
    segments: list[tuple[int, int]] = []
    for a, b in zip(hpos, hpos[1:]):
        # Keep only non-trivial H-bounded windows with at least two internal residues.
        if b > a + 2:
            segments.append((a, b))
    return sorted(set(segments))


def missing_hh_pairs(pos: list[int], seq: str, coords, nbr_sets, limit: int) -> list[tuple[int, int, int]]:
    present = contact_set(pos, seq, nbr_sets)
    hs = [i for i, a in enumerate(seq) if a == "H"]
    out: list[tuple[int, int, int]] = []
    for ix, i in enumerate(hs):
        ci = coords[pos[i]]
        for j in hs[ix + 1:]:
            if abs(i - j) <= 1 or (i, j) in present:
                continue
            d = manhattan(ci, coords[pos[j]])
            out.append((d, i, j))
    out.sort(key=lambda x: (x[0], abs(x[2] - x[1])))
    return out[:limit]


def local_partial_score(site: int, residue: int, partial: dict[int, int], seq: str, nbr_sets) -> float:
    if seq[residue] != "H":
        return 0.0
    score = 0.0
    for j, sj in partial.items():
        if seq[j] == "H" and abs(residue - j) > 1 and sj in nbr_sets[site]:
            score -= 1.0
    return score


def beam_regrow_segment(
    pos: list[int],
    seq: str,
    coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]],
    rng: random.Random,
    start: int,
    end: int,
    required: dict[int, set[int]] | None = None,
    beam_width: int = 96,
) -> list[int] | None:
    if start < 0 or end >= len(pos) or start > end:
        return None
    required = required or {}
    fixed_indices = [i for i in range(len(pos)) if i < start or i > end]
    fixed_sites = {pos[i] for i in fixed_indices}
    fixed_map = {i: pos[i] for i in fixed_indices}
    left = pos[start - 1] if start > 0 else None
    right = pos[end + 1] if end < len(pos) - 1 else None
    beam: list[tuple[float, list[int], set[int]]] = [(0.0, [], set())]
    for residue in range(start, end + 1):
        new_beam: list[tuple[float, list[int], set[int]]] = []
        for score, path, used in beam:
            if not path:
                if left is None:
                    candidates = [s for s in range(len(coords)) if s not in fixed_sites]
                    rng.shuffle(candidates)
                    candidates = candidates[:32]
                else:
                    candidates = list(nbr_sets[left] - fixed_sites)
            else:
                candidates = list(nbr_sets[path[-1]] - fixed_sites - used)
            if residue in required:
                candidates = [c for c in candidates if c in required[residue]]
            rng.shuffle(candidates)
            remaining = end - residue
            for cand in candidates:
                if cand in used:
                    continue
                if right is not None:
                    dist = manhattan(coords[cand], coords[right])
                    edges_to_right = remaining + 1
                    if dist > edges_to_right or (edges_to_right - dist) % 2 != 0:
                        continue
                partial = dict(fixed_map)
                for off, site in enumerate(path):
                    partial[start + off] = site
                sc = score + local_partial_score(cand, residue, partial, seq, nbr_sets) + rng.random() * 0.02
                new_beam.append((sc, path + [cand], used | {cand}))
        if not new_beam:
            return None
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    finals: list[tuple[float, list[int]]] = []
    for sc, path, _used in beam:
        if right is not None and right not in nbr_sets[path[-1]]:
            continue
        new_pos = list(pos)
        new_pos[start:end + 1] = path
        if validate_pos(new_pos, nbr_sets):
            finals.append((sc, new_pos))
    if not finals:
        return None
    finals.sort(key=lambda x: x[0])
    return rng.choice(finals[: min(5, len(finals))])[1]


def hsegment_repair(
    pos: list[int],
    seq: str,
    coords,
    nbr_sets,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    rng: random.Random,
    segments: list[tuple[int, int]],
    repair_beam: int,
    pair_limit: int,
) -> list[int] | None:
    cur_hh = hh_contacts(pos, hmask, nbr_bool)
    occupied = set(pos)
    candidates: list[tuple[float, list[int]]] = []
    for _dist, i, j in missing_hh_pairs(pos, seq, coords, nbr_sets, pair_limit):
        for moving, fixed in ((i, j), (j, i)):
            required_sites = set(nbr_sets[pos[fixed]]) - (occupied - {pos[moving]})
            if not required_sites:
                continue
            for start, end in segments:
                if not (start <= moving <= end) or (start <= fixed <= end):
                    continue
                cand = beam_regrow_segment(pos, seq, coords, nbr_sets, rng, start, end, {moving: required_sites}, repair_beam)
                if cand is None or not valid_chain_fast(cand, nbr_bool):
                    continue
                new_hh = hh_contacts(cand, hmask, nbr_bool)
                if new_hh >= cur_hh - 1:
                    candidates.append((-new_hh + rng.random() * 0.02, cand))
                if new_hh > cur_hh:
                    break
            if candidates and -min(x[0] for x in candidates) > cur_hh:
                break
        if candidates and -min(x[0] for x in candidates) > cur_hh:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return rng.choice(candidates[: min(4, len(candidates))])[1]


def segment_windows(n: int, moving: int, fixed: int, max_len: int, max_windows: int) -> list[tuple[int, int]]:
    lengths = [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 37, 43]
    windows: list[tuple[int, int]] = []
    for seg_len in lengths:
        if seg_len > max_len or seg_len > n - 2:
            continue
        lo = max(1, moving - seg_len + 1)
        hi = min(moving, n - seg_len - 1)
        for start in range(lo, hi + 1):
            end = start + seg_len - 1
            if start <= fixed <= end:
                continue
            windows.append((start, end))
    windows.sort(key=lambda se: (se[1] - se[0] + 1, abs((se[0] + se[1]) // 2 - moving)))
    return windows[:max_windows]


def missing_contact_window_repair(
    pos: list[int],
    seq: str,
    coords,
    nbr_sets,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    rng: random.Random,
    beam_width: int,
    max_windows: int,
    pair_limit: int,
    max_len: int = 44,
) -> list[int] | None:
    cur_hh = hh_contacts(pos, hmask, nbr_bool)
    occupied = set(pos)
    candidates: list[tuple[float, list[int]]] = []
    pairs = missing_hh_pairs(pos, seq, coords, nbr_sets, pair_limit)
    top_n = min(8, len(pairs))
    top_pairs = pairs[:top_n]
    rng.shuffle(top_pairs)
    pairs = top_pairs + pairs[top_n:]
    for _dist, i, j in pairs:
        for moving, fixed in ((i, j), (j, i)):
            required_sites = set(nbr_sets[pos[fixed]]) - (occupied - {pos[moving]})
            if not required_sites:
                continue
            for start, end in segment_windows(len(pos), moving, fixed, min(len(pos) - 2, max_len), max_windows):
                cand = beam_regrow_segment(pos, seq, coords, nbr_sets, rng, start, end, {moving: required_sites}, beam_width)
                if cand is None or not valid_chain_fast(cand, nbr_bool):
                    continue
                new_hh = hh_contacts(cand, hmask, nbr_bool)
                if new_hh >= cur_hh - 2:
                    candidates.append((-new_hh + rng.random() * 0.03, cand))
                if new_hh > cur_hh:
                    break
            if candidates and -min(x[0] for x in candidates) > cur_hh:
                break
        if candidates and -min(x[0] for x in candidates) > cur_hh:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return rng.choice(candidates[: min(5, len(candidates))])[1]


def two_sided_contact_repair(
    pos: list[int],
    seq: str,
    coords,
    nbr_sets,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    rng: random.Random,
    beam_width: int,
    max_windows: int,
    pair_limit: int,
    max_len: int = 44,
) -> list[int] | None:
    """Like missing_contact_window_repair, but lets both residues of a missing
    H-H pair move instead of pinning one side as a fixed anchor.

    missing_contact_window_repair regrows a window around `moving` while
    requiring it to land next to `fixed`, which never relocates.  At high
    contact density (e.g. >=53/55 on a packed 60mer) most remaining missing
    contacts need *both* residues to shift before they can become neighbors,
    so the one-sided move structurally cannot find them.

    This version regrows a single window that spans both residues of a
    missing pair (rather than pinning one as a fixed anchor for the other),
    so both ends of the pair are free to move within the same beam search.
    A two-residue requirement window is then accepted only if the regrow
    happens to land i and j as lattice neighbors. This is one
    beam_regrow_segment call per window, so it stays within the
    "lightweight" budget of the polish stage -- the cost comes from trying
    more windows/pairs, not from a heavier search per window.
    """
    cur_hh = hh_contacts(pos, hmask, nbr_bool)
    n = len(pos)
    candidates: list[tuple[float, list[int]]] = []
    pairs = missing_hh_pairs(pos, seq, coords, nbr_sets, pair_limit)
    top_n = min(8, len(pairs))
    top_pairs = pairs[:top_n]
    rng.shuffle(top_pairs)
    pairs = top_pairs + pairs[top_n:]

    for _dist, i, j in pairs:
        lo, hi = (i, j) if i < j else (j, i)
        span = hi - lo + 1
        if span > max_len or span > n - 2:
            continue
        # Build windows that fully contain [lo, hi] so both residues of the
        # pair are inside the regrown region and free to move relative to
        # each other.  Try the tight window first, then pad outward a bit
        # so the beam search has more room to route the chain between them.
        pad_options = [0, 2, 4, 6, 8][:max(1, max_windows // 2)]
        windows: list[tuple[int, int]] = []
        for pad in pad_options:
            start = max(1, lo - pad)
            end = min(n - 2, hi + pad)
            if end - start + 1 > max_len or end - start + 1 > n - 2:
                continue
            if (start, end) not in windows:
                windows.append((start, end))
        windows = windows[:max_windows]

        for start, end in windows:
            cand = beam_regrow_segment(pos, seq, coords, nbr_sets, rng, start, end, None, beam_width)
            if cand is None or not valid_chain_fast(cand, nbr_bool):
                continue
            if cand[j] not in nbr_sets[cand[i]]:
                # This regrow didn't happen to bring i and j together; still
                # worth keeping as a candidate if it didn't lose ground,
                # since it may have closed a *different* missing contact
                # incidentally, but it isn't the targeted repair, so only
                # keep it under the same not-worse-than-(-2) bar as the
                # one-sided repair uses.
                new_hh = hh_contacts(cand, hmask, nbr_bool)
                if new_hh >= cur_hh - 2:
                    candidates.append((-new_hh + rng.random() * 0.03, cand))
                continue
            new_hh = hh_contacts(cand, hmask, nbr_bool)
            if new_hh >= cur_hh - 2:
                candidates.append((-new_hh + rng.random() * 0.03, cand))
            if new_hh > cur_hh:
                break
        if candidates and -min(x[0] for x in candidates) > cur_hh:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return rng.choice(candidates[: min(5, len(candidates))])[1]
    return rng.choice(candidates[: min(5, len(candidates))])[1]


def true3d_move(
    pos: list[int],
    seq: str,
    coords,
    coord_to_site,
    nbr_sets,
    rng: random.Random,
    regrow_max_len: int,
    attempts: dict[str, int],
    weights: list[tuple[str, float]] | None = None,
) -> tuple[str, list[int] | None]:
    weights = weights or [
        ("contact_pull", 0.20),
        ("pull", 0.15),
        ("axis_crank", 0.10),
        ("pair_crank", 0.10),
        ("fragment_regrow", 0.14),
        ("end_regrow", 0.09),
        ("pivot", 0.10),
        ("reptation", 0.07),
        ("local", 0.05),
        ("double_regrow", 0.05),
    ]
    name = weighted_choice(weights, rng)
    attempts[name] = attempts.get(name, 0) + 1
    if name == "contact_pull":
        return name, contact_guided_pull(pos, seq, nbr_sets, HP_MATRIX, rng)
    if name == "pull":
        return name, pull_through_move(pos, seq, nbr_sets, HP_MATRIX, rng)
    if name == "axis_crank":
        return name, axis_crankshaft_move(pos, coords, coord_to_site, nbr_sets, seq, HP_MATRIX, rng)
    if name == "pair_crank":
        return name, pair_crankshaft_move(pos, seq, nbr_sets, HP_MATRIX, rng)
    if name == "double_regrow":
        return name, true3d_double_regrow(pos, nbr_sets, rng, regrow_max_len)
    # Fallback/core moves.
    if name == "fragment_regrow":
        return name, bridge_regrow(pos, nbr_sets, rng, regrow_max_len)
    if name == "end_regrow":
        return name, regrow_tail(pos, nbr_sets, rng, regrow_max_len)
    if name == "pivot":
        return name, pivot_move(pos, coords, coord_to_site, nbr_sets, rng)
    if name == "reptation":
        return name, reptation_move(pos, nbr_sets, rng)
    return "local", local_move(pos, nbr_sets, rng)


def anneal_temperature(t_init: float, t_final: float, step: int, steps: int) -> float:
    frac = step / max(1, steps - 1)
    return t_init * ((t_final / t_init) ** frac)


def result_row(stage_name: str, row: BenchmarkRow, result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    best = result["best"]
    return {
        "benchmark": row.benchmark,
        "N": row.length,
        "stage": stage_name,
        "target_energy": row.target_energy,
        "best_energy": best["energy"],
        "HH_contacts": best.get("HH", best.get("n_contacts")),
        "target_hit": bool(result.get("hit_target")),
        "wall_s": result.get("wall_seconds", result.get("runtime_seconds", 0.0)),
        "folder": str(run_dir),
    }


def save_report(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", "", "| Benchmark | N | Stage | Target E | Best E | Hit | HH | Wall s | Folder |", "|---|---:|---|---:|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r['benchmark']} | {r['N']} | {r['stage']} | {r['target_energy']} | {r['best_energy']} | "
            f"{str(r['target_hit']).lower()} | {r['HH_contacts']} | {float(r['wall_s']):.3f} | `{Path(r['folder']).name}` |"
        )
    write_json(path.with_suffix(".json"), rows)
    path.write_text("\n".join(lines), encoding="utf-8")
