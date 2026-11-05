#!/usr/bin/env python3
"""Uniform-temperature long-cycle Q-RBHA on a 3D FCC lattice.

This implementation preserves the residue-site/QUBO workflow while replacing
the original Python candidate loop with a compiled assignment-space kernel.
It does not call pivot, pull, crankshaft, reptation, or fragment-regrowth move
operators. Every proposal uses the original Cluster-RBSA random block-length
and start-index policy, then reconstructs that block between its fixed sequence
boundaries.

This traceable discovery-only variant preserves the baseline Q-RBHA-FCC
kernel while assigning every worker the same temperature endpoints and using
longer progressive cooling cycles. It provides:

* one independent sequence-aware SAW per worker;
* original Cluster-RBSA block selection: 70% short and 30% full-range blocks;
* boundary-conditioned stochastic beam construction;
* exact incremental H-H and cluster-excess deltas for candidate scoring;
* direct heat-bath transitions (no second Metropolis filter);
* a cluster-shaping coefficient that decays to zero;
* configurable progressive discovery cycles;
* configurable stage advance after stagnant cycles;
* worker-best restoration at each stage advance;
* a wider stochastic beam once the physical H-H score is near target;
* per-worker archives of structurally distinct near-target folds.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import math
import multiprocessing as mp
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*_args, **_kwargs):
        def deco(fn):
            return fn
        return deco


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from export_3d_FCC_HP_cluster_qubo_light import export_lightweight_model  # noqa: E402
from fcc_lattice import fcc_graph_distance, model_context, validate_lattice  # noqa: E402
from hp3d_stage_common import (  # noqa: E402
    BenchmarkRow,
    contact_set,
    direct_row,
    load_rows,
    pairwise_contact_saw,
    target_hh,
    valid_chain_fast,
    write_json,
)


ALGORITHM = "UTLC-Q-RBHA-FCC"
REPO_ROOT = SOURCE_DIR.parent
DEFAULT_CYCLE_STEPS = (80000, 160000, 320000, 640000)
DEFAULT_CLUSTER_MULTIPLIERS = (1.0, 0.66, 0.33, 0.0)


def parse_int_tuple(text: str, expected: int) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(values) != expected or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError(f"expected {expected} positive comma-separated integers")
    return values


def parse_float_tuple(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated numbers")
    return values


def expanding_cycle_steps(stage: int, initial_steps: tuple[int, ...]) -> int:
    """Return the cycle length, doubling indefinitely after the final base level."""
    if stage < 0:
        raise ValueError("stage must be nonnegative")
    if not initial_steps:
        raise ValueError("initial_steps must not be empty")
    if stage < len(initial_steps):
        return int(initial_steps[stage])
    return int(initial_steps[-1]) * (1 << (stage - len(initial_steps) + 1))


def persistent_stage_value(values: tuple[Any, ...], stage: int) -> Any:
    """Keep the final shaping/block parameter for all expanded cycle levels."""
    if not values:
        raise ValueError("stage values must not be empty")
    return values[min(stage, len(values) - 1)]


def neighbor_array(nbr_sets: list[set[int]]) -> np.ndarray:
    out = np.full((len(nbr_sets), 12), -1, dtype=np.int32)
    for site, values in enumerate(nbr_sets):
        for k, value in enumerate(sorted(values)):
            out[site, k] = int(value)
    return out


@njit(cache=True)
def _seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit(cache=True)
def _excess_value(degree: int, threshold: int) -> int:
    return degree - threshold if degree > threshold else 0


@njit(cache=True)
def state_metrics_numba(
    pos: np.ndarray,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    threshold: int,
) -> tuple[int, int, np.ndarray]:
    n = pos.shape[0]
    degrees = np.zeros(n, dtype=np.int16)
    hh = 0
    for i in range(n):
        if hmask[i] == 0:
            continue
        for j in range(i + 2, n):
            if hmask[j] != 0 and nbr_bool[pos[i], pos[j]]:
                hh += 1
                degrees[i] += 1
                degrees[j] += 1
    excess = 0
    for i in range(n):
        if hmask[i] != 0:
            excess += _excess_value(int(degrees[i]), threshold)
    return hh, excess, degrees


def state_metrics(
    pos: np.ndarray,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    threshold: int,
) -> tuple[int, int, np.ndarray]:
    n = int(pos.shape[0])
    degrees = np.zeros(n, dtype=np.int16)
    hh = 0
    for i in range(n):
        if int(hmask[i]) == 0:
            continue
        for j in range(i + 2, n):
            if int(hmask[j]) != 0 and bool(nbr_bool[pos[i], pos[j]]):
                hh += 1
                degrees[i] += 1
                degrees[j] += 1
    excess = sum(
        max(0, int(degrees[i]) - threshold)
        for i in range(n)
        if int(hmask[i]) != 0
    )
    return hh, excess, degrees


def _fcc_distance(coords: np.ndarray, first: int, second: int) -> int:
    dx = abs(int(coords[first, 0]) - int(coords[second, 0]))
    dy = abs(int(coords[first, 1]) - int(coords[second, 1]))
    dz = abs(int(coords[first, 2]) - int(coords[second, 2]))
    return max(dx, dy, dz, (dx + dy + dz) // 2)


@njit(cache=True)
def choose_cluster_rbsa_block_numba(n: int, min_len: int, max_len: int) -> tuple[int, int]:
    """Match Cluster-RBSA's random 70% short / 30% full-range policy."""
    max_len = max(min_len, min(max_len, n - 1))
    short_max = min(max_len, max(min_len, 8))
    if np.random.random() < 0.70:
        length = np.random.randint(min_len, short_max + 1)
    else:
        length = np.random.randint(min_len, max_len + 1)
    start = np.random.randint(0, n - length + 1)
    return int(start), int(start + length)


def choose_cluster_rbsa_block(n: int, min_len: int, max_len: int) -> tuple[int, int]:
    max_len = max(min_len, min(max_len, n - 1))
    short_max = min(max_len, max(min_len, 8))
    if float(np.random.random()) < 0.70:
        length = int(np.random.randint(min_len, short_max + 1))
    else:
        length = int(np.random.randint(min_len, max_len + 1))
    start = int(np.random.randint(0, n - length + 1))
    return start, start + length


@njit(cache=True)
def candidate_metrics_delta_numba(
    pos: np.ndarray,
    block: np.ndarray,
    a: int,
    b: int,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    current_hh: int,
    current_excess: int,
    current_degrees: np.ndarray,
    threshold: int,
) -> tuple[int, int, np.ndarray]:
    """Exact local delta for pairs with at least one endpoint in [a,b)."""
    n = pos.shape[0]
    degrees = current_degrees.copy()
    hh = current_hh
    excess = current_excess

    for i in range(n):
        if hmask[i] == 0:
            continue
        new_i = int(block[i - a]) if a <= i < b else int(pos[i])
        for j in range(i + 2, n):
            if hmask[j] == 0 or not (a <= i < b or a <= j < b):
                continue
            new_j = int(block[j - a]) if a <= j < b else int(pos[j])
            old_contact = bool(nbr_bool[pos[i], pos[j]])
            new_contact = bool(nbr_bool[new_i, new_j])
            if old_contact == new_contact:
                continue

            old_i_degree = int(degrees[i])
            old_j_degree = int(degrees[j])
            excess -= _excess_value(old_i_degree, threshold)
            excess -= _excess_value(old_j_degree, threshold)
            if new_contact:
                hh += 1
                degrees[i] += 1
                degrees[j] += 1
            else:
                hh -= 1
                degrees[i] -= 1
                degrees[j] -= 1
            excess += _excess_value(int(degrees[i]), threshold)
            excess += _excess_value(int(degrees[j]), threshold)

    return hh, excess, degrees


def candidate_metrics_delta(
    pos: np.ndarray,
    block: np.ndarray,
    a: int,
    b: int,
    hmask: np.ndarray,
    nbr_bool: np.ndarray,
    current_hh: int,
    current_excess: int,
    current_degrees: np.ndarray,
    threshold: int,
) -> tuple[int, int, np.ndarray]:
    n = int(pos.shape[0])
    degrees = current_degrees.copy()
    hh = int(current_hh)
    excess = int(current_excess)
    for i in range(n):
        if int(hmask[i]) == 0:
            continue
        new_i = int(block[i - a]) if a <= i < b else int(pos[i])
        for j in range(i + 2, n):
            if int(hmask[j]) == 0 or not (a <= i < b or a <= j < b):
                continue
            new_j = int(block[j - a]) if a <= j < b else int(pos[j])
            old_contact = bool(nbr_bool[pos[i], pos[j]])
            new_contact = bool(nbr_bool[new_i, new_j])
            if old_contact == new_contact:
                continue
            excess -= max(0, int(degrees[i]) - threshold)
            excess -= max(0, int(degrees[j]) - threshold)
            delta = 1 if new_contact else -1
            hh += delta
            degrees[i] += delta
            degrees[j] += delta
            excess += max(0, int(degrees[i]) - threshold)
            excess += max(0, int(degrees[j]) - threshold)
    return hh, excess, degrees


def valid_chain_fcc(pos: list[int], nbr_bool: np.ndarray) -> bool:
    if len(pos) != len(set(pos)):
        return False
    site_count = int(nbr_bool.shape[0])
    if any(site < 0 or site >= site_count for site in pos):
        return False
    return all(bool(nbr_bool[pos[i], pos[i + 1]]) for i in range(len(pos) - 1))


@njit(cache=True)
def _path_contains(path: np.ndarray, depth: int, site: int) -> bool:
    for k in range(depth):
        if int(path[k]) == site:
            return True
    return False


@njit(cache=True)
def boundary_beam_candidates_numba(
    pos: np.ndarray,
    a: int,
    b: int,
    hmask: np.ndarray,
    coords: np.ndarray,
    nbrs: np.ndarray,
    nbr_bool: np.ndarray,
    current_degrees: np.ndarray,
    pair_hh_reward: float,
    deficit_bias: float,
    temperature: float,
    beam_width: int,
) -> tuple[np.ndarray, int]:
    """Construct connected, collision-free block assignments with fixed boundaries."""
    n = pos.shape[0]
    m = b - a
    site_count = coords.shape[0]
    fixed = np.zeros(site_count, dtype=np.uint8)
    for i in range(n):
        if i < a or i >= b:
            fixed[pos[i]] = 1

    forward = a > 0
    if forward:
        anchor = int(pos[a - 1])
        target = int(pos[b]) if b < n else -1
    else:
        anchor = int(pos[b])
        target = -1

    paths = np.full((beam_width, m), -1, dtype=np.int32)
    scores = np.full(beam_width, 1.0e30, dtype=np.float64)
    beam_count = 1
    scores[0] = 0.0

    max_expansions = beam_width * 12
    expanded_paths = np.full((max_expansions, m), -1, dtype=np.int32)
    expanded_scores = np.full(max_expansions, 1.0e30, dtype=np.float64)

    for depth in range(m):
        expansion_count = 0
        residue = a + depth if forward else b - 1 - depth
        for parent in range(beam_count):
            previous = anchor if depth == 0 else int(paths[parent, depth - 1])
            for neighbor_index in range(12):
                site = int(nbrs[previous, neighbor_index])
                if site < 0 or fixed[site] != 0 or _path_contains(paths[parent], depth, site):
                    continue

                remaining = m - depth - 1
                if target >= 0:
                    distance = _fcc_distance(coords, site, target)
                    edges_left = remaining + 1
                    # FCC contains triangles, so the cubic parity constraint on
                    # the remaining path length does not apply.
                    if distance > edges_left:
                        continue
                    if remaining == 0 and distance != 1:
                        continue

                if expansion_count >= max_expansions:
                    continue
                for k in range(depth):
                    expanded_paths[expansion_count, k] = paths[parent, k]
                expanded_paths[expansion_count, depth] = site

                weighted_contacts = 0.0
                if hmask[residue] != 0:
                    for j in range(n):
                        if hmask[j] == 0 or abs(residue - j) <= 1:
                            continue
                        if a <= j < b:
                            if forward:
                                placed_depth = j - a
                            else:
                                placed_depth = b - 1 - j
                            if placed_depth < depth and nbr_bool[site, paths[parent, placed_depth]]:
                                deficit = max(0, 2 - int(current_degrees[residue]))
                                deficit += max(0, 2 - int(current_degrees[j]))
                                weighted_contacts += 1.0 + 0.5 * deficit_bias * float(deficit)
                        elif nbr_bool[site, pos[j]]:
                            deficit = max(0, 2 - int(current_degrees[residue]))
                            deficit += max(0, 2 - int(current_degrees[j]))
                            weighted_contacts += 1.0 + 0.5 * deficit_bias * float(deficit)

                distance_bias = 0.0
                if target >= 0:
                    distance_bias = 0.04 * float(_fcc_distance(coords, site, target))
                # Gumbel perturbation gives a stochastic beam without Python RNG.
                u = max(1.0e-12, min(1.0 - 1.0e-12, np.random.random()))
                noise = -math.log(-math.log(u)) * max(0.03, temperature) * 0.08
                expanded_scores[expansion_count] = (
                    scores[parent] + pair_hh_reward * weighted_contacts + distance_bias - noise
                )
                expansion_count += 1

        if expansion_count == 0:
            return paths, 0

        order = np.argsort(expanded_scores[:expansion_count])
        new_count = 0
        for order_index in range(expansion_count):
            source = int(order[order_index])
            duplicate = False
            for existing in range(new_count):
                same = True
                for k in range(depth + 1):
                    if paths[existing, k] != expanded_paths[source, k]:
                        same = False
                        break
                if same:
                    duplicate = True
                    break
            if duplicate:
                continue
            for k in range(depth + 1):
                paths[new_count, k] = expanded_paths[source, k]
            scores[new_count] = expanded_scores[source]
            new_count += 1
            if new_count >= beam_width:
                break
        beam_count = new_count

    if not forward:
        for row in range(beam_count):
            for left in range(m // 2):
                right = m - 1 - left
                value = paths[row, left]
                paths[row, left] = paths[row, right]
                paths[row, right] = value
    return paths, beam_count


def boundary_beam_candidates_python(
    pos: np.ndarray,
    a: int,
    b: int,
    hmask: np.ndarray,
    coords: np.ndarray,
    nbrs: np.ndarray,
    nbr_bool: np.ndarray,
    current_degrees: np.ndarray,
    pair_hh_reward: float,
    deficit_bias: float,
    temperature: float,
    beam_width: int,
) -> tuple[np.ndarray, int]:
    """FCC boundary beam with a small Python controller and array-based state.

    Keeping this controller outside Numba avoids a multi-minute compilation of
    the dynamically allocated beam orchestration on a clean installation.  The
    incremental contact scorer remains compiled.
    """
    n = int(pos.shape[0])
    m = int(b - a)
    fixed = {int(pos[i]) for i in range(n) if i < a or i >= b}
    forward = a > 0
    if forward:
        anchor = int(pos[a - 1])
        target = int(pos[b]) if b < n else -1
    else:
        anchor = int(pos[b])
        target = -1

    beam: list[tuple[list[int], float]] = [([], 0.0)]
    for depth in range(m):
        residue = a + depth if forward else b - 1 - depth
        expanded: list[tuple[list[int], float]] = []
        for path, parent_score in beam:
            previous = anchor if depth == 0 else path[-1]
            occupied_path = set(path)
            for raw_site in nbrs[previous]:
                site = int(raw_site)
                if site < 0 or site in fixed or site in occupied_path:
                    continue
                remaining = m - depth - 1
                if target >= 0:
                    distance = int(_fcc_distance(coords, site, target))
                    edges_left = remaining + 1
                    if distance > edges_left or (remaining == 0 and distance != 1):
                        continue

                weighted_contacts = 0.0
                if int(hmask[residue]) != 0:
                    for j in range(n):
                        if int(hmask[j]) == 0 or abs(residue - j) <= 1:
                            continue
                        neighbor_contact = False
                        if a <= j < b:
                            placed_depth = j - a if forward else b - 1 - j
                            if placed_depth < depth:
                                neighbor_contact = bool(nbr_bool[site, path[placed_depth]])
                        else:
                            neighbor_contact = bool(nbr_bool[site, pos[j]])
                        if neighbor_contact:
                            deficit = max(0, 2 - int(current_degrees[residue]))
                            deficit += max(0, 2 - int(current_degrees[j]))
                            weighted_contacts += 1.0 + 0.5 * deficit_bias * float(deficit)

                distance_bias = 0.0
                if target >= 0:
                    distance_bias = 0.04 * float(_fcc_distance(coords, site, target))
                u = max(1.0e-12, min(1.0 - 1.0e-12, float(np.random.random())))
                noise = -math.log(-math.log(u)) * max(0.03, temperature) * 0.08
                score = parent_score + pair_hh_reward * weighted_contacts + distance_bias - noise
                expanded.append((path + [site], score))

        if not expanded:
            return np.full((beam_width, m), -1, dtype=np.int32), 0
        expanded.sort(key=lambda item: item[1])
        beam = []
        seen: set[tuple[int, ...]] = set()
        for path, score in expanded:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            beam.append((path, score))
            if len(beam) >= beam_width:
                break

    output = np.full((beam_width, m), -1, dtype=np.int32)
    for row, (path, _score) in enumerate(beam):
        if not forward:
            path = list(reversed(path))
        output[row, :m] = np.asarray(path, dtype=np.int32)
    return output, len(beam)


def run_kernel_chunk_numba(
    pos: np.ndarray,
    current_hh: int,
    current_excess: int,
    current_degrees: np.ndarray,
    best_pos: np.ndarray,
    best_hh: int,
    best_excess: int,
    hmask: np.ndarray,
    coords: np.ndarray,
    nbrs: np.ndarray,
    nbr_bool: np.ndarray,
    threshold: int,
    pair_hh_reward: float,
    cluster_reward: float,
    cluster_tie_reward: float,
    deficit_bias: float,
    min_block_len: int,
    max_block_len: int,
    beam_width: int,
    t0: float,
    t1: float,
    cycle_length: int,
    cycle_offset: int,
    chunk_steps: int,
    target_hh_value: int,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, int, int, int, int, int]:
    transitions = 0
    valid_candidate_sets = 0
    steps_done = 0

    for local_step in range(chunk_steps):
        step_in_cycle = cycle_offset + local_step
        fraction = float(step_in_cycle) / float(max(1, cycle_length - 1))
        temperature = t0 * math.exp(math.log(t1 / t0) * fraction)

        a, b = choose_cluster_rbsa_block(pos.shape[0], min_block_len, max_block_len)
        candidate_blocks, candidate_count = boundary_beam_candidates_python(
            pos, a, b, hmask, coords, nbrs, nbr_bool,
            current_degrees, pair_hh_reward, deficit_bias, temperature, beam_width,
        )
        steps_done += 1
        if candidate_count <= 0:
            continue
        valid_candidate_sets += 1

        # Candidate zero is the current state.  The remaining entries are
        # complete feasible assignments returned by the boundary beam.
        score_count = candidate_count + 1
        scores = np.empty(score_count, dtype=np.float64)
        hh_values = np.empty(score_count, dtype=np.int32)
        excess_values = np.empty(score_count, dtype=np.int32)
        degree_values = np.empty((score_count, pos.shape[0]), dtype=np.int16)
        scores[0] = (
            pair_hh_reward * float(current_hh)
            + cluster_reward * float(current_excess)
            + cluster_tie_reward * float(current_excess)
        )
        hh_values[0] = current_hh
        excess_values[0] = current_excess
        degree_values[0, :] = current_degrees

        minimum = scores[0]
        for candidate in range(candidate_count):
            hh, excess, degrees = candidate_metrics_delta(
                pos, candidate_blocks[candidate], a, b, hmask, nbr_bool,
                current_hh, current_excess, current_degrees, threshold,
            )
            index = candidate + 1
            score = (
                pair_hh_reward * float(hh)
                + cluster_reward * float(excess)
                + cluster_tie_reward * float(excess)
            )
            scores[index] = score
            hh_values[index] = hh
            excess_values[index] = excess
            degree_values[index, :] = degrees
            if score < minimum:
                minimum = score

        total_weight = 0.0
        weights = np.empty(score_count, dtype=np.float64)
        for index in range(score_count):
            exponent = -(scores[index] - minimum) / max(temperature, 1.0e-12)
            if exponent < -700.0:
                weight = 0.0
            else:
                weight = math.exp(exponent)
            weights[index] = weight
            total_weight += weight

        selected = 0
        if total_weight > 0.0 and math.isfinite(total_weight):
            pick = np.random.random() * total_weight
            cumulative = 0.0
            for index in range(score_count):
                cumulative += weights[index]
                if cumulative >= pick:
                    selected = index
                    break
        else:
            selected = int(np.argmin(scores))

        # Direct heat-bath transition: no second Metropolis filter.
        if selected > 0:
            pos[a:b] = candidate_blocks[selected - 1, : b - a]
            current_hh = int(hh_values[selected])
            current_excess = int(excess_values[selected])
            current_degrees[:] = degree_values[selected, :]
            transitions += 1

            if current_hh > best_hh or (current_hh == best_hh and current_excess > best_excess):
                best_hh = current_hh
                best_excess = current_excess
                best_pos[:] = pos
                if best_hh >= target_hh_value:
                    break

    return (
        pos, current_hh, current_excess, current_degrees,
        best_pos, best_hh, best_excess,
        steps_done, transitions, valid_candidate_sets,
    )


def metrics_python(
    pos: list[int], seq: str, nbr_sets: list[set[int]], threshold: int
) -> tuple[int, int, list[int]]:
    degrees = [0] * len(pos)
    hh = 0
    for i, residue in enumerate(seq):
        if residue != "H":
            continue
        for j in range(i + 2, len(seq)):
            if seq[j] == "H" and pos[j] in nbr_sets[pos[i]]:
                hh += 1
                degrees[i] += 1
                degrees[j] += 1
    excess = sum(max(0, degrees[i] - threshold) for i, residue in enumerate(seq) if residue == "H")
    return hh, excess, degrees


def make_best_entry(
    pos: list[int], seq: str, coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]], nbr_bool: np.ndarray,
    threshold: int, pair_reward: float, cluster_reward: float,
    source: str,
) -> dict[str, Any]:
    hh, excess, degrees = metrics_python(pos, seq, nbr_sets, threshold)
    valid = bool(valid_chain_fcc(pos, nbr_bool))
    contacts = sorted(contact_set(pos, seq, nbr_sets)) if valid else []
    return {
        "format": "hp3d_fcc_dhb_rbsa_best_v1",
        "algorithm": ALGORITHM,
        "model": "HP",
        "seq": seq,
        "N": len(seq),
        "energy": -float(hh),
        "score": int(hh),
        "HH": int(hh),
        "pairwise_energy": float(pair_reward) * float(hh),
        "cluster_excess": int(excess),
        "cluster_energy_at_full_weight": float(pair_reward) * float(hh) + float(cluster_reward) * float(excess),
        "h_degrees": [int(x) for x in degrees],
        "valid": valid,
        "site_duplicate_pairs": len(pos) - len(set(pos)),
        "broken_chain_links": 0 if valid else sum(
            1 for i in range(len(pos) - 1) if pos[i + 1] not in nbr_sets[pos[i]]
        ),
        "sites": [int(x) for x in pos],
        "coords": [list(coords[int(site)]) for site in pos],
        "contacts": [list(pair) for pair in contacts],
        "source": source,
    }


def write_worker_progress(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one worker's recoverable progress checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def archive_add(
    archive: list[dict[str, Any]],
    pos: np.ndarray,
    hh: int,
    excess: int,
    seq: str,
    nbr_sets: list[set[int]],
    limit: int,
    minimum_contact_difference: int,
) -> None:
    """Keep high-quality folds whose nonbonded H-H contact maps differ."""
    sites = [int(x) for x in pos]
    signature = frozenset(contact_set(sites, seq, nbr_sets))
    score = (int(hh), int(excess))
    for index, entry in enumerate(archive):
        difference = len(signature.symmetric_difference(entry["signature"]))
        if difference < minimum_contact_difference:
            if score > (int(entry["HH"]), int(entry["cluster_excess"])):
                archive[index] = {
                    "sites": sites,
                    "HH": int(hh),
                    "cluster_excess": int(excess),
                    "signature": signature,
                }
            return
    archive.append({
        "sites": sites,
        "HH": int(hh),
        "cluster_excess": int(excess),
        "signature": signature,
    })
    archive.sort(key=lambda entry: (int(entry["HH"]), int(entry["cluster_excess"])), reverse=True)
    del archive[max(1, int(limit)) :]


def serializable_archive(archive: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sites": [int(x) for x in entry["sites"]],
            "HH": int(entry["HH"]),
            "cluster_excess": int(entry["cluster_excess"]),
            "contact_count": len(entry["signature"]),
        }
        for entry in archive
    ]


def worker_run(task: dict[str, Any]) -> dict[str, Any]:
    worker_id = int(task["worker_id"])
    seed = int(task["seed"])
    data, coords, _coord_to_site, nbr_sets, nbr_bool, hmask = model_context(Path(task["model_json"]))
    coords_array = np.asarray(coords, dtype=np.int16)
    neighbors = neighbor_array(nbr_sets)
    starts = [[int(x) for x in start] for start in task["starts"]]
    cycle_steps = tuple(int(x) for x in task["cycle_steps"])
    cluster_multipliers = tuple(float(x) for x in task["cluster_multipliers"])
    stage_block_max = tuple(int(x) for x in task["stage_block_max"])
    threshold = int(task["cluster_threshold"])
    pair_reward = float(task["pair_hh_reward"])
    cluster_reward_base = float(task["cluster_excess_reward"])
    target = int(task["target_hh"])
    target_stop_enabled = bool(task.get("target_stop_enabled", True))
    kernel_target = target if target_stop_enabled else np.iinfo(np.int32).max
    wall_budget = float(task["wall_seconds"])
    chunk_steps = int(task["control_chunk_steps"])
    stagnant_limit = int(task["cycles_without_improvement"])
    near_target_gap = int(task["near_target_gap"])
    near_target_beam_width = int(task["near_target_beam_width"])
    stop_event = task.get("stop_event")
    progress_path = Path(task["progress_dir"]) / f"worker_{worker_id:02d}_progress.json"

    np.random.seed(seed)
    current = np.asarray(starts[0], dtype=np.int32)
    current_hh, current_excess, current_degrees = state_metrics(current, hmask, nbr_bool, threshold)
    best_pos = current.copy()
    best_hh = int(current_hh)
    best_excess = int(current_excess)
    seq = str(data["seq"])
    first_recorded = {str(best_hh): 0.0}
    trace: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    stage = 0
    stagnant_cycles = 0
    cycle_index = 0
    restart_index = 0
    next_restore_source = "fresh_start_0"
    total_steps = 0
    transitions = 0
    candidate_sets = 0
    started = time.perf_counter()

    # Compile before starting the reported wall clock.
    _ = run_kernel_chunk_numba(
        current.copy(), int(current_hh), int(current_excess), current_degrees.copy(),
        best_pos.copy(), int(best_hh), int(best_excess), hmask, coords_array, neighbors,
        nbr_bool, threshold, pair_reward, cluster_reward_base, 0.0, 0.0,
        int(task["min_block_len"]), min(4, int(task["max_block_len"])),
        min(4, int(task["beam_width"])), float(task["t0"]), float(task["t1"]),
        2, 0, 1, kernel_target,
    )
    started = time.perf_counter()

    while time.perf_counter() - started < wall_budget:
        if stop_event is not None and stop_event.is_set():
            break
        restore_source = next_restore_source
        next_restore_source = "carry"
        phase = "discovery"
        cycle_length = expanding_cycle_steps(stage, cycle_steps)
        stage_cluster_multiplier = float(persistent_stage_value(cluster_multipliers, stage))
        stage_max_block_len = int(persistent_stage_value(stage_block_max, stage))
        active_cluster_reward = cluster_reward_base * stage_cluster_multiplier
        active_tie_reward = 0.0
        active_deficit_bias = 0.0
        active_beam_width = int(task["beam_width"])
        max_beam_width_used = active_beam_width
        active_t0 = float(task["t0"])
        active_t1 = float(task["t1"])
        cycle_start_best = best_hh
        cycle_started = time.perf_counter()
        offset = 0

        while offset < cycle_length:
            if time.perf_counter() - started >= wall_budget:
                break
            if stop_event is not None and stop_event.is_set():
                break
            this_chunk = min(chunk_steps, cycle_length - offset)
            previous_best = best_hh
            chunk_beam_width = (
                near_target_beam_width
                if int(best_hh) >= target - near_target_gap
                else active_beam_width
            )
            max_beam_width_used = max(max_beam_width_used, chunk_beam_width)
            (
                current, current_hh, current_excess, current_degrees,
                best_pos, best_hh, best_excess,
                done, moved, valid_sets,
            ) = run_kernel_chunk_numba(
                current, int(current_hh), int(current_excess), current_degrees,
                best_pos, int(best_hh), int(best_excess), hmask, coords_array, neighbors,
                nbr_bool, threshold, pair_reward,
                active_cluster_reward, active_tie_reward, active_deficit_bias,
                int(task["min_block_len"]), stage_max_block_len, chunk_beam_width,
                active_t0, active_t1, cycle_length, offset,
                this_chunk, kernel_target,
            )
            offset += int(done)
            total_steps += int(done)
            transitions += int(moved)
            candidate_sets += int(valid_sets)
            if int(best_hh) > int(previous_best):
                elapsed = time.perf_counter() - started
                first_recorded.setdefault(str(int(best_hh)), float(elapsed))
            if target_stop_enabled and int(best_hh) >= target:
                if stop_event is not None:
                    stop_event.set()
                break
            if int(done) < this_chunk:
                break

        improved = int(best_hh) > int(cycle_start_best)
        stagnant_cycles = 0 if improved else stagnant_cycles + 1
        archive_minimum_hh = max(0, target - int(task["archive_target_gap"]))
        if int(best_hh) >= archive_minimum_hh:
            archive_add(
                archive, best_pos, int(best_hh), int(best_excess), seq, nbr_sets,
                int(task["archive_limit"]), int(task["archive_min_contact_difference"]),
            )
        if int(current_hh) >= max(archive_minimum_hh, int(best_hh) - 1):
            archive_add(
                archive, current, int(current_hh), int(current_excess), seq, nbr_sets,
                int(task["archive_limit"]), int(task["archive_min_contact_difference"]),
            )
        trace.append({
            "worker_id": worker_id,
            "cycle_index": cycle_index,
            "stage": stage,
            "phase": phase,
            "cycle_steps_requested": cycle_length,
            "cycle_steps_completed": offset,
            "stage_cluster_multiplier": stage_cluster_multiplier,
            "stage_max_block_len": stage_max_block_len,
            "temperature_t0": active_t0,
            "temperature_t1": active_t1,
            "beam_width": active_beam_width,
            "max_beam_width_used": max_beam_width_used,
            "near_target_beam_threshold_HH": target - near_target_gap,
            "deficit_bias": active_deficit_bias,
            "cluster_tie_reward": active_tie_reward,
            "restore_source": restore_source,
            "archive_size": len(archive),
            "cycle_improved": bool(improved),
            "stagnant_cycles": stagnant_cycles,
            "best_HH": int(best_hh),
            "current_HH": int(current_hh),
            "elapsed_seconds": float(time.perf_counter() - started),
            "cycle_wall_seconds": float(time.perf_counter() - cycle_started),
            "restart_index": restart_index,
        })
        write_worker_progress(progress_path, {
            "algorithm": ALGORITHM,
            "worker_id": worker_id,
            "seed": seed,
            "status": "target_hit" if int(best_hh) >= target else "running",
            "target_HH": target,
            "best_HH": int(best_hh),
            "best_cluster_excess": int(best_excess),
            "best_sites": [int(x) for x in best_pos],
            "current_HH": int(current_hh),
            "stage": int(stage),
            "phase": phase,
            "cycle_index": int(cycle_index),
            "cycle_steps": int(cycle_length),
            "total_steps": int(total_steps),
            "elapsed_seconds": float(time.perf_counter() - started),
            "first_recorded_seconds_by_HH": first_recorded,
            "archive_size": len(archive),
            "last_cycle": trace[-1],
        })
        cycle_index += 1
        if target_stop_enabled and int(best_hh) >= target:
            break
        if stagnant_cycles >= stagnant_limit:
            stage += 1
            current = best_pos.copy()
            next_restore_source = "worker_best_stage_advance"
            stagnant_cycles = 0
            current_hh, current_excess, current_degrees = state_metrics(
                current, hmask, nbr_bool, threshold
            )

    wall = time.perf_counter() - started
    best = make_best_entry(
        [int(x) for x in best_pos], str(data["seq"]), coords, nbr_sets, nbr_bool,
        threshold, pair_reward, cluster_reward_base, f"worker{worker_id}:best",
    )
    write_worker_progress(progress_path, {
        "algorithm": ALGORITHM,
        "worker_id": worker_id,
        "seed": seed,
        "status": "target_hit" if int(best_hh) >= target else "completed",
        "target_HH": target,
        "best_HH": int(best_hh),
        "best_cluster_excess": int(best_excess),
        "best_sites": [int(x) for x in best_pos],
        "current_HH": int(current_hh),
        "stage": int(stage),
        "phase": trace[-1]["phase"] if trace else "not_started",
        "cycles_completed": int(cycle_index),
        "closure_cycles_completed": 0,
        "total_steps": int(total_steps),
        "elapsed_seconds": float(wall),
        "first_recorded_seconds_by_HH": first_recorded,
        "archive": serializable_archive(archive),
        "last_cycle": trace[-1] if trace else None,
    })
    return {
        "worker_id": worker_id,
        "seed": seed,
        "target_hit": int(best_hh) >= target,
        "target_stop_enabled": target_stop_enabled,
        "best": best,
        "wall_seconds": float(wall),
        "total_steps": int(total_steps),
        "transitions": int(transitions),
        "valid_candidate_sets": int(candidate_sets),
        "cycles_completed": int(cycle_index),
        "final_stage": int(stage),
        "closure_cycles_completed": 0,
        "fresh_restarts": int(restart_index),
        "first_recorded_seconds_by_HH": first_recorded,
        "archive": serializable_archive(archive),
        "trace": trace,
    }


def write_lightweight_model(model_json: Path, run_dir: Path) -> Path:
    data = json.loads(model_json.read_text(encoding="utf-8"))
    for key in ("linear", "quadratic", "ising"):
        data.pop(key, None)
    data["worker_lightweight_model"] = True
    destination = run_dir / f"model_FCC_HP_cluster_L{int(data['L'])}_worker_light.json"
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return destination


def prepare_model(row: BenchmarkRow, run_dir: Path, args: argparse.Namespace) -> Path:
    if args.model_json:
        source = Path(args.model_json)
        if not source.exists():
            raise FileNotFoundError(source)
        destination = run_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

    destination = run_dir / f"model_FCC_HP_cluster_L{int(args.l_size)}_worker_light.json"
    export_lightweight_model(
        row.hp_sequence, destination, int(args.l_size),
        float(args.pair_hh_reward), int(args.cluster_threshold),
        float(args.cluster_excess_reward), float(args.scale_target),
    )
    return destination


def generate_worker_starts(
    row: BenchmarkRow, model_json: Path, args: argparse.Namespace
) -> list[list[list[int]]]:
    _data, coords, _coord_to_site, nbr_sets, nbr_bool, hmask = model_context(model_json)
    pools: list[list[list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for worker_id in range(int(args.workers)):
        worker_starts: list[list[int]] = []
        for restart in range(int(args.fresh_starts_per_worker)):
            seed = int(args.seed) + worker_id * int(args.worker_seed_stride) + restart * int(args.restart_seed_stride)
            rng = random.Random(seed)
            pos = pairwise_contact_saw(
                row.hp_sequence, coords, nbr_sets, hmask, nbr_bool, rng,
                attempts=int(args.init_attempts), h_pool=int(args.init_h_pool),
                p_pool=int(args.init_p_pool), random_pool_prob=float(args.init_random_pool_prob),
            )
            if pos is None:
                raise RuntimeError(f"failed to generate SAW for worker {worker_id}, restart {restart}")
            key = tuple(int(x) for x in pos)
            if key in seen:
                # A second independent call is normally sufficient; exact
                # uniqueness is preferred but not allowed to block a run.
                retry = pairwise_contact_saw(
                    row.hp_sequence, coords, nbr_sets, hmask, nbr_bool, rng,
                    attempts=int(args.init_attempts), h_pool=max(3, int(args.init_h_pool) + 1),
                    p_pool=max(5, int(args.init_p_pool) + 1),
                    random_pool_prob=min(0.35, float(args.init_random_pool_prob) + 0.12),
                )
                if retry is not None:
                    pos = retry
                    key = tuple(int(x) for x in pos)
            seen.add(key)
            worker_starts.append([int(x) for x in pos])
        pools.append(worker_starts)
    return pools


def run_benchmark(row: BenchmarkRow, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.out_dir) / row.benchmark / args.run_name
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_json = prepare_model(row, run_dir, args)
    data, coords, _coord_to_site, nbr_sets, nbr_bool, _hmask = model_context(model_json)
    if int(data["S"]) < len(row.hp_sequence):
        raise ValueError(
            f"{ALGORITHM} lattice L={data['L']} has only {data['S']} sites for "
            f"the {len(row.hp_sequence)}-residue chain"
        )
    target = target_hh(row.target_energy)
    if target is None:
        raise ValueError(f"benchmark {row.benchmark} has no target energy")
    starts = generate_worker_starts(row, model_json, args)
    write_json(run_dir / "initial_saw_pool.json", {
        "algorithm": ALGORITHM,
        "worker_starts": starts,
    })

    cycle_steps = parse_int_tuple(args.cycle_steps, 4)
    cluster_multipliers = parse_float_tuple(args.cluster_multipliers, 4)
    stage_block_max = tuple(
        min(int(args.max_block_len_cap), int(args.max_block_len) + 4 * stage)
        for stage in range(4)
    )
    config = {
        "algorithm": ALGORITHM,
        "benchmark": row.benchmark,
        "N": len(row.hp_sequence),
        "seq": row.hp_sequence,
        "target_HH": target,
        "model_json": str(model_json),
        "workers": int(args.workers),
        "execution": "sequential independent workers" if args.serial_workers else "process-parallel independent workers",
        "wall_seconds": float(args.wall_seconds),
        "target_stop_enabled": not bool(args.disable_target_stop),
        "cycle_steps": list(cycle_steps),
        "discovery_cycle_rule": (
            f"advance through {', '.join(str(value) for value in cycle_steps)} proposals, "
            f"then double indefinitely after {int(args.cycles_without_improvement)} "
            "stagnant cycles at each level"
        ),
        "fresh_starts_per_worker": int(args.fresh_starts_per_worker),
        "restart_policy": "disabled; carry worker best into each larger cycle level",
        "closure_phase": "disabled",
        "archive_limit": int(args.archive_limit),
        "archive_target_gap": int(args.archive_target_gap),
        "archive_min_contact_difference": int(args.archive_min_contact_difference),
        "cycles_without_improvement": int(args.cycles_without_improvement),
        "temperature_policy": "uniform across all workers",
        "temperature_t0": float(args.t0),
        "temperature_t1": float(args.t1),
        "cluster_multipliers": list(cluster_multipliers),
        "stage_block_max": list(stage_block_max),
        "beam_width": int(args.beam_width),
        "near_target_beam_width": int(args.near_target_beam_width),
        "near_target_gap": int(args.near_target_gap),
        "selection": "Q-RBHA policy: 70% short blocks (length <= 8), 30% full configured range",
        "transition": "direct heat-bath transition; no second Metropolis filter",
        "pair_hh_reward": float(args.pair_hh_reward),
        "cluster_excess_reward": float(args.cluster_excess_reward),
        "cluster_threshold": int(args.cluster_threshold),
        "seed": int(args.seed),
        "l_size": int(data["L"]),
        "site_count": int(data["S"]),
        "lattice": "3d_fcc",
    }
    write_json(run_dir / "run_config.json", config)
    command = "Set-Location '" + str(SOURCE_DIR) + "'\n\npython '" + str(Path(__file__).resolve()) + "' " + " ".join(
        "'" + str(value).replace("'", "''") + "'" for value in sys.argv[1:]
    ) + "\n"
    (run_dir / "run_command.ps1").write_text(command, encoding="utf-8")

    manager = None if args.serial_workers else mp.Manager()
    stop_event = threading.Event() if args.serial_workers else manager.Event()
    tasks = []
    for worker_id in range(int(args.workers)):
        tasks.append({
            "worker_id": worker_id,
            "seed": int(args.seed) + worker_id * int(args.worker_seed_stride),
            "model_json": str(model_json),
            "starts": starts[worker_id],
            "target_hh": target,
            "target_stop_enabled": not bool(args.disable_target_stop),
            "cycle_steps": cycle_steps,
            "cluster_multipliers": cluster_multipliers,
            "stage_block_max": stage_block_max,
            "cluster_threshold": int(args.cluster_threshold),
            "pair_hh_reward": float(args.pair_hh_reward),
            "cluster_excess_reward": float(args.cluster_excess_reward),
            "wall_seconds": float(args.wall_seconds),
            "control_chunk_steps": int(args.control_chunk_steps),
            "cycles_without_improvement": int(args.cycles_without_improvement),
            "min_block_len": int(args.min_block_len),
            "max_block_len": int(args.max_block_len),
            "beam_width": int(args.beam_width),
            "near_target_beam_width": int(args.near_target_beam_width),
            "near_target_gap": int(args.near_target_gap),
            "t0": float(args.t0),
            "t1": float(args.t1),
            "archive_limit": int(args.archive_limit),
            "archive_target_gap": int(args.archive_target_gap),
            "archive_min_contact_difference": int(args.archive_min_contact_difference),
            "stop_event": stop_event,
            "progress_dir": str(run_dir),
        })

    run_started = time.perf_counter()
    worker_results: list[dict[str, Any]] = []
    def record_result(result: dict[str, Any]) -> None:
        worker_results.append(result)
        print(
            f"[worker {result['worker_id']:02d}] bestHH={result['best']['HH']} "
            f"cycles={result['cycles_completed']} steps={result['total_steps']} "
            f"wall={result['wall_seconds']:.1f}s hit={result['target_hit']}",
            flush=True,
        )

    if args.serial_workers:
        for task in tasks:
            result = worker_run(task)
            record_result(result)
            if result["target_hit"] and not bool(args.disable_target_stop):
                stop_event.set()
                break
    else:
        context = mp.get_context("spawn")
        with futures.ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as pool:
            pending = [pool.submit(worker_run, task) for task in tasks]
            for completed in futures.as_completed(pending):
                result = completed.result()
                record_result(result)
                if result["target_hit"] and not bool(args.disable_target_stop):
                    stop_event.set()

    wall = time.perf_counter() - run_started
    if manager is not None:
        manager.shutdown()
    worker_results.sort(key=lambda item: int(item["worker_id"]))
    best_result = max(
        worker_results,
        key=lambda item: (int(item["best"]["HH"]), int(item["best"]["cluster_excess"])),
    )
    best = best_result["best"]
    write_json(run_dir / "worker_results.json", worker_results)
    write_json(run_dir / "best.json", best)

    trace_rows = [row for result in worker_results for row in result["trace"]]
    write_json(run_dir / "cycle_trace.json", trace_rows)
    if trace_rows:
        with (run_dir / "cycle_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trace_rows)

    summary = {
        "algorithm": ALGORITHM,
        "benchmark": row.benchmark,
        "N": len(row.hp_sequence),
        "target_HH": target,
        "best_HH": int(best["HH"]),
        "best_energy": -float(best["HH"]),
        "target_hit": int(best["HH"]) >= target,
        "valid": bool(best["valid"]),
        "wall_seconds": float(wall),
        "winning_worker": int(best_result["worker_id"]),
        "winning_worker_first_recorded_seconds_by_HH": best_result["first_recorded_seconds_by_HH"],
        "total_worker_steps": int(sum(int(item["total_steps"]) for item in worker_results)),
        "total_transitions": int(sum(int(item["transitions"]) for item in worker_results)),
        "run_dir": str(run_dir),
        "config": config,
    }
    write_json(run_dir / "q_rbha_result.json", summary)
    return summary


def self_test_delta() -> None:
    lattice_check = validate_lattice(5)
    test_model = SOURCE_DIR / "_fcc_self_test_model.json"
    export_lightweight_model("HPHHPH", test_model, 5, -2.0, 2, -0.5, 1000.0)
    _data, coord_list, coord_to_site, nbr_sets, nbr_bool, _model_hmask = model_context(test_model)
    coords = np.asarray(coord_list, dtype=np.int16)
    coordinate_path = [(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0), (4, 0, 0), (3, 0, 1)]
    pos = np.asarray([coord_to_site[coordinate] for coordinate in coordinate_path], dtype=np.int32)
    hmask = np.asarray([1, 0, 1, 1, 0, 1], dtype=np.int8)
    hh, excess, degrees = state_metrics(pos, hmask, nbr_bool, 2)
    candidate_coords = [(2, 2, 0), (3, 1, 0)]
    candidate = np.asarray([coord_to_site[coordinate] for coordinate in candidate_coords], dtype=np.int32)
    new_hh, new_excess, _new_degrees = candidate_metrics_delta(
        pos, candidate, 2, 4, hmask, nbr_bool, hh, excess, degrees, 2
    )
    new_pos = pos.copy()
    new_pos[2:4] = candidate
    check_hh, check_excess, _ = state_metrics(new_pos, hmask, nbr_bool, 2)
    if int(new_hh) != int(check_hh) or int(new_excess) != int(check_excess):
        raise AssertionError((new_hh, new_excess, check_hh, check_excess))
    schedule = [expanding_cycle_steps(stage, DEFAULT_CYCLE_STEPS) for stage in range(8)]
    expected = [80000, 160000, 320000, 640000, 1280000, 2560000, 5120000, 10240000]
    if schedule != expected:
        raise AssertionError((schedule, expected))
    test_model.unlink(missing_ok=True)
    print(json.dumps({
        "self_test": "passed",
        "fcc_lattice": lattice_check,
        "HH": int(check_hh),
        "cluster_excess": int(check_excess),
        "expanding_cycle_steps": schedule,
        "progressive_cycle_steps": schedule,
        "closure_phase": "disabled",
    }))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {ALGORITHM} on 3D FCC HP benchmarks.")
    parser.add_argument("--seq", default="")
    parser.add_argument("--target_energy", type=float, default=0.0)
    parser.add_argument("--benchmark", default="direct")
    parser.add_argument(
        "--benchmarks_csv",
        default=str(REPO_ROOT / "data" / "benchmarks" / "3d_primary_benchmarks.csv"),
    )
    parser.add_argument("--only", default="")
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--run_name", default="UTLC_Q_RBHA_FCC_discovery_only")
    parser.add_argument("--model_json", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--serial_workers", action="store_true",
        help="Run independent worker seeds sequentially; useful on Windows when clean-process JIT startup dominates a short experiment.",
    )
    parser.add_argument("--wall_seconds", type=float, default=43200.0)
    parser.add_argument(
        "--disable_target_stop",
        action="store_true",
        help="Record target hits but let every worker consume its complete wall budget.",
    )
    parser.add_argument("--cycle_steps", default="80000,160000,320000,640000")
    parser.add_argument("--cycles_without_improvement", type=int, default=2)
    parser.add_argument("--control_chunk_steps", type=int, default=256)
    parser.add_argument("--fresh_starts_per_worker", type=int, default=1)
    parser.add_argument("--l_size", type=int, default=8)
    parser.add_argument("--init_attempts", type=int, default=8000)
    parser.add_argument("--init_h_pool", type=int, default=2)
    parser.add_argument("--init_p_pool", type=int, default=4)
    parser.add_argument("--init_random_pool_prob", type=float, default=0.10)
    parser.add_argument("--pair_hh_reward", type=float, default=-2.0)
    parser.add_argument("--cluster_excess_reward", type=float, default=-0.5)
    parser.add_argument("--cluster_threshold", type=int, default=2)
    parser.add_argument("--cluster_multipliers", default="1.0,0.66,0.33,0.0")
    parser.add_argument("--scale_target", type=float, default=1000.0)
    parser.add_argument("--min_block_len", type=int, default=2)
    parser.add_argument("--max_block_len", type=int, default=12)
    parser.add_argument("--max_block_len_cap", type=int, default=24)
    parser.add_argument("--beam_width", type=int, default=24)
    parser.add_argument(
        "--near_target_beam_width", type=int, default=48,
        help="Beam width used when worker-best HH is within near_target_gap of the target.",
    )
    parser.add_argument(
        "--near_target_gap", type=int, default=2,
        help="Activate the near-target beam at target_HH - this value.",
    )
    parser.add_argument("--t0", type=float, default=8.0)
    parser.add_argument("--t1", type=float, default=0.02)
    parser.add_argument("--archive_limit", type=int, default=8)
    parser.add_argument("--archive_target_gap", type=int, default=4)
    parser.add_argument("--archive_min_contact_difference", type=int, default=4)
    parser.add_argument("--worker_seed_stride", type=int, default=1000003)
    parser.add_argument("--restart_seed_stride", type=int, default=15485863)
    parser.add_argument("--seed", type=int, default=21013001)
    return parser.parse_args()


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    if args.near_target_gap < 0:
        raise ValueError("near_target_gap must be nonnegative")
    if args.near_target_beam_width < args.beam_width:
        raise ValueError("near_target_beam_width must be at least beam_width")
    if args.self_test:
        self_test_delta()
        return
    rows = [direct_row(args.seq, args.target_energy, args.benchmark)] if args.seq else load_rows(
        Path(args.benchmarks_csv), args.only
    )
    results = [run_benchmark(row, args) for row in rows]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
