#!/usr/bin/env python3
"""Export an auditable lightweight residue-site QUBO model on a 3D FCC lattice."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fcc_lattice import fcc_lattice_coords, fcc_neighbor_sets


def export_lightweight_model(
    sequence: str,
    output: Path,
    lattice_size: int,
    pair_hh_reward: float,
    cluster_threshold: int,
    cluster_excess_reward: float,
    scale_target: float,
) -> dict:
    sequence = sequence.strip().upper()
    if not sequence or set(sequence) - {"H", "P"}:
        raise ValueError("sequence must contain only H and P")

    coords = fcc_lattice_coords(int(lattice_size))
    neighbors = fcc_neighbor_sets(coords)
    n = len(sequence)
    sites = len(coords)
    if sites < n:
        raise ValueError(f"FCC L={lattice_size} has only {sites} sites for N={n}")

    matrix = {
        "H": {"H": float(pair_hh_reward), "P": 0.0},
        "P": {"H": 0.0, "P": 0.0},
    }
    max_abs = max(1.0, abs(float(pair_hh_reward)))
    local_bound = 12.0 * max(1, n - 1) * max_abs
    penalty = max(1.0, 6.0 * local_bound)
    max_coefficient = max(2.0 * penalty, abs(float(pair_hh_reward)))
    alpha = float(scale_target) / max_coefficient if max_coefficient else 1.0
    directed_edges = sum(len(values) for values in neighbors)
    hh_pairs = sum(
        1
        for first, residue in enumerate(sequence)
        if residue == "H"
        for second in range(first + 2, n)
        if sequence[second] == "H"
    )

    payload = {
        "format": "hp_cluster_pairwise_qubo_ising_3d_fcc_worker_light_v1",
        "model": "HP",
        "dimension": 3,
        "lattice": "3d_fcc",
        "coordinate_representation": "D3 integer triples with even coordinate sum",
        "seq": sequence,
        "alphabet": ["H", "P"],
        "N": n,
        "L": int(lattice_size),
        "S": sites,
        "V": n * sites,
        "contact_energy_formula": (
            "E_effective = pairwise_hh_reward * HH + "
            "cluster_excess_reward * sum_H max(0, d_H - cluster_threshold)"
        ),
        "contact_matrix": matrix,
        "matrix_source": "embedded",
        "penalty_mode": "auto_fcc_coordination_bound",
        "lambda_onehot": penalty,
        "lambda_site": penalty,
        "lambda_chain": penalty,
        "scale_target": float(scale_target),
        "scale_alpha": alpha,
        "scaled_penalties": {
            "lambda_onehot": penalty * alpha,
            "lambda_site": penalty * alpha,
            "lambda_chain": penalty * alpha,
        },
        "coords": [list(coordinate) for coordinate in coords],
        "neighbor_count": 12,
        "directed_neighbor_pairs": directed_edges,
        "term_counts": {
            "onehot_linear": n * sites,
            "onehot_quadratic": n * math.comb(sites, 2),
            "site_exclusivity": sites * math.comb(n, 2),
            "chain_continuity": (n - 1) * (sites * sites - directed_edges),
            "contact_terms": hh_pairs * directed_edges,
        },
        "scoring_model": "fcc_hp_pairwise_plus_h_neighbor_excess_v1",
        "pairwise_hh_reward": float(pair_hh_reward),
        "cluster_threshold": int(cluster_threshold),
        "cluster_excess_reward": float(cluster_excess_reward),
        "cluster_scoring": {
            "decoded_score": True,
            "degree_definition": "d_H counts nonconsecutive H residues adjacent on the FCC lattice",
            "quadratic_qubo_contains": "pairwise H-H contact reward only",
            "higher_order_note": "Compact-core shaping is evaluated on feasible decoded folds.",
        },
        "worker_lightweight_model": True,
        "explicit_term_arrays_materialized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l_size", type=int, default=8)
    parser.add_argument("--pair_hh_reward", type=float, default=-2.0)
    parser.add_argument("--cluster_threshold", type=int, default=2)
    parser.add_argument("--cluster_excess_reward", type=float, default=-0.5)
    parser.add_argument("--scale_target", type=float, default=1000.0)
    args = parser.parse_args()
    result = export_lightweight_model(
        args.seq, args.out, args.l_size, args.pair_hh_reward,
        args.cluster_threshold, args.cluster_excess_reward, args.scale_target,
    )
    print(json.dumps({key: result[key] for key in ("N", "L", "S", "V", "neighbor_count", "term_counts")}, indent=2))


if __name__ == "__main__":
    main()
