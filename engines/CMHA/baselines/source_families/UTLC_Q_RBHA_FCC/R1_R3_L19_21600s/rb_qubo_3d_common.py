#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared 3D lattice QUBO export and feasible-chain RB-QUBO SA utilities.

The public entry points are mode-specific scripts:

    export_3d_HP_qubo.py
    export_3d_HPNX_qubo.py
    export_3d_MJ_qubo.py
    solve_3d_HP_rb_qubo_sa.py
    solve_3d_HPNX_rb_qubo_sa.py
    solve_3d_MJ_rb_qubo_sa.py

This module keeps the implementations consistent while still providing
separate user-facing codes for HP, HPNX, and MJ validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path


HP_ALPHABET = list("HP")
HPNX_ALPHABET = list("HPNX")
MJ_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


HP_MATRIX = {
    "H": {"H": -1.0, "P": 0.0},
    "P": {"H": 0.0, "P": 0.0},
}

HPNX_MATRIX = {
    "H": {"H": -4.0, "P": 0.0, "N": 0.0, "X": 0.0},
    "P": {"H": 0.0, "P": 1.0, "N": -1.0, "X": 0.0},
    "N": {"H": 0.0, "P": -1.0, "N": 1.0, "X": 0.0},
    "X": {"H": 0.0, "P": 0.0, "N": 0.0, "X": 0.0},
}


def mj_validation_matrix() -> dict[str, dict[str, float]]:
    """Return a full 20-letter MJ-compatible validation contact matrix.

    This built-in matrix is meant for software smoke validation and uses a
    compact class-based approximation over the standard amino-acid alphabet.
    Publication experiments should pass the exact literature matrix with
    ``--matrix_json`` so the reported energies match the chosen MJ reference.
    """

    hydrophobic = set("ACFILMVWY")
    positive = set("KRH")
    negative = set("DE")
    polar = set("GNQSTP")

    matrix: dict[str, dict[str, float]] = {}
    for a in MJ_ALPHABET:
        matrix[a] = {}
        for b in MJ_ALPHABET:
            if a in hydrophobic and b in hydrophobic:
                val = -2.0
            elif (a in positive and b in negative) or (a in negative and b in positive):
                val = -1.2
            elif (a in positive and b in positive) or (a in negative and b in negative):
                val = 0.8
            elif (a in hydrophobic and b in polar) or (a in polar and b in hydrophobic):
                val = -0.3
            elif a in polar and b in polar:
                val = -0.1
            else:
                val = 0.0
            matrix[a][b] = float(val)
    return matrix


def builtin_config(mode: str) -> tuple[list[str], dict[str, dict[str, float]], str, str]:
    mode = mode.upper()
    if mode == "HP":
        return HP_ALPHABET, HP_MATRIX, "E = -HH", "builtin_hp_hh_minus_one"
    if mode == "HPNX":
        return HPNX_ALPHABET, HPNX_MATRIX, "E = -4*HH - PN + PP + NN", "builtin_hpnx_backofen_will"
    if mode == "MJ":
        return MJ_ALPHABET, mj_validation_matrix(), "E = sum MJ(a,b) over nonconsecutive contacts", "builtin_mj_validation_class_matrix"
    raise SystemExit(f"unsupported mode: {mode}")


def load_contact_matrix(path: str | Path, alphabet: list[str]) -> tuple[dict[str, dict[str, float]], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = data.get("contact_matrix", data)
    missing = [a for a in alphabet if a not in matrix]
    if missing:
        raise SystemExit(f"contact matrix missing rows for {missing}")
    out: dict[str, dict[str, float]] = {}
    for a in alphabet:
        out[a] = {}
        for b in alphabet:
            if b not in matrix[a]:
                raise SystemExit(f"contact matrix missing coefficient for {a},{b}")
            out[a][b] = float(matrix[a][b])
    return out, str(path)


def validate_seq(seq: str, alphabet: list[str]) -> str:
    seq = seq.strip().upper()
    allowed = set(alphabet)
    bad = sorted(set(seq) - allowed)
    if bad:
        raise SystemExit(f"invalid symbols {bad}; allowed alphabet is {''.join(alphabet)}")
    if not seq:
        raise SystemExit("sequence must not be empty")
    return seq


def choose_l_size(n: int, mode: str = "balanced", min_l: int = 3) -> int:
    factors = {"tight": 1.2, "balanced": 1.8, "roomy": 2.5}
    return max(min_l, int(math.ceil((factors[mode] * n) ** (1.0 / 3.0))))


def lattice_coords(l_size: int) -> list[tuple[int, int, int]]:
    return [(x, y, z) for x in range(l_size) for y in range(l_size) for z in range(l_size)]


def neighbor_sets_from_coords(coords: list[tuple[int, int, int]]) -> list[set[int]]:
    coord_to_site = {c: i for i, c in enumerate(coords)}
    deltas = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    nbr_sets: list[set[int]] = []
    for x, y, z in coords:
        ns = set()
        for dx, dy, dz in deltas:
            j = coord_to_site.get((x + dx, y + dy, z + dz))
            if j is not None:
                ns.add(j)
        nbr_sets.append(ns)
    return nbr_sets


def auto_penalties(seq: str, matrix: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    max_abs = max(abs(matrix[a][b]) for a in matrix for b in matrix[a])
    local_bound = 6.0 * max(1, len(seq) - 1) * max(1.0, max_abs)
    lam = max(1.0, 6.0 * local_bound)
    return lam, lam, lam


def add_ising_from_qubo(idx_list: list[int], coeff: float, h: dict[int, float], j2: dict[tuple[int, int], float]) -> None:
    if len(idx_list) == 1:
        i = idx_list[0]
        h[i] = h.get(i, 0.0) + coeff / 2.0
    elif len(idx_list) == 2:
        i, j = sorted(idx_list)
        h[i] = h.get(i, 0.0) + coeff / 4.0
        h[j] = h.get(j, 0.0) + coeff / 4.0
        j2[(i, j)] = j2.get((i, j), 0.0) + coeff / 4.0
    else:
        raise ValueError("pairwise exporter supports only linear/quadratic QUBO terms")


def export_model(
    mode: str,
    seq: str,
    out_path: str | Path,
    l_size: int | None = None,
    auto_mode: str = "balanced",
    penalty_mode: str = "auto",
    lambda_onehot: float | None = None,
    lambda_site: float | None = None,
    lambda_chain: float | None = None,
    scale_target: float = 1000.0,
    matrix_json: str = "",
) -> dict:
    mode = mode.upper()
    alphabet, matrix, formula, matrix_source = builtin_config(mode)
    if matrix_json:
        matrix, matrix_source = load_contact_matrix(matrix_json, alphabet)
    seq = validate_seq(seq, alphabet)

    n = len(seq)
    l_size = int(l_size) if l_size is not None else choose_l_size(n, auto_mode)
    coords = lattice_coords(l_size)
    nbr_sets = neighbor_sets_from_coords(coords)
    s_count = len(coords)

    if penalty_mode == "manual":
        if None in (lambda_onehot, lambda_site, lambda_chain):
            raise SystemExit("manual penalty mode requires all three lambda values")
        lam_onehot = float(lambda_onehot)
        lam_site = float(lambda_site)
        lam_chain = float(lambda_chain)
    else:
        lam_onehot, lam_site, lam_chain = auto_penalties(seq, matrix)

    def idx_of(r: int, site: int) -> int:
        return r * s_count + site

    linear: list[tuple[int, float]] = []
    quadratic: list[tuple[int, int, float]] = []
    term_counts = {
        "onehot_linear": 0,
        "onehot_quadratic": 0,
        "site_exclusivity": 0,
        "chain_continuity": 0,
        "contact_terms": 0,
    }

    for r in range(n):
        inds = [idx_of(r, s) for s in range(s_count)]
        for i in inds:
            linear.append((i, -lam_onehot))
            term_counts["onehot_linear"] += 1
        for a in range(len(inds)):
            for b in range(a + 1, len(inds)):
                quadratic.append((inds[a], inds[b], 2.0 * lam_onehot))
                term_counts["onehot_quadratic"] += 1

    for site in range(s_count):
        inds = [idx_of(r, site) for r in range(n)]
        for a in range(len(inds)):
            for b in range(a + 1, len(inds)):
                quadratic.append((inds[a], inds[b], lam_site))
                term_counts["site_exclusivity"] += 1

    all_sites = set(range(s_count))
    for r in range(n - 1):
        for s in range(s_count):
            illegal = all_sites - nbr_sets[s]
            for t in illegal:
                quadratic.append((idx_of(r, s), idx_of(r + 1, t), lam_chain))
                term_counts["chain_continuity"] += 1

    for i_res in range(n):
        ai = seq[i_res]
        for j_res in range(i_res + 2, n):
            aj = seq[j_res]
            coeff = float(matrix[ai][aj])
            if coeff == 0.0:
                continue
            for s in range(s_count):
                for t in nbr_sets[s]:
                    # Residue i and residue j are distinct variable blocks.
                    # Both site orderings are needed because a feasible fold
                    # may place residue i at the larger site index and residue
                    # j at the smaller one.  These are not duplicate QUBO
                    # variables: (i_res,s),(j_res,t) differs from
                    # (i_res,t),(j_res,s).
                    quadratic.append((idx_of(i_res, s), idx_of(j_res, t), coeff))
                    term_counts["contact_terms"] += 1

    max_coeff = 0.0
    for _, coeff in linear:
        max_coeff = max(max_coeff, abs(coeff))
    for _, _, coeff in quadratic:
        max_coeff = max(max_coeff, abs(coeff))
    alpha = 1.0
    if max_coeff > 0.0 and scale_target > 0.0:
        alpha = float(scale_target) / max_coeff
        linear = [(i, coeff * alpha) for i, coeff in linear]
        quadratic = [(i, j, coeff * alpha) for i, j, coeff in quadratic]

    h: dict[int, float] = {}
    j2: dict[tuple[int, int], float] = {}
    for i, coeff in linear:
        add_ising_from_qubo([i], coeff, h, j2)
    for i, j, coeff in quadratic:
        add_ising_from_qubo([i, j], coeff, h, j2)

    out = {
        "format": f"{mode.lower()}_pairwise_qubo_ising_3d_v1",
        "model": mode,
        "dimension": 3,
        "lattice": "3d_cubic",
        "seq": seq,
        "alphabet": alphabet,
        "N": n,
        "L": l_size,
        "S": s_count,
        "V": n * s_count,
        "contact_energy_formula": formula,
        "contact_matrix": matrix,
        "matrix_source": matrix_source,
        "penalty_mode": penalty_mode,
        "lambda_onehot": float(lam_onehot),
        "lambda_site": float(lam_site),
        "lambda_chain": float(lam_chain),
        "scale_target": float(scale_target),
        "scale_alpha": float(alpha),
        "scaled_penalties": {
            "lambda_onehot": float(lam_onehot * alpha),
            "lambda_site": float(lam_site * alpha),
            "lambda_chain": float(lam_chain * alpha),
        },
        "coords": coords,
        "neighbor_count": 6,
        "term_counts": term_counts,
        "linear": linear,
        "quadratic": quadratic,
        "ising": {
            "h": {str(k): v for k, v in h.items()},
            "J2": [(i, j, coeff) for (i, j), coeff in j2.items()],
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def export_cli(mode: str) -> None:
    p = argparse.ArgumentParser(description=f"Export 3D cubic {mode.upper()} pairwise QUBO + Ising JSON.")
    p.add_argument("--seq", required=True, help=f"{mode.upper()} sequence.")
    p.add_argument("--l_size", type=int, default=None, help="3D cubic lattice side length L.")
    p.add_argument("--auto_mode", default="balanced", choices=["tight", "balanced", "roomy"])
    p.add_argument("--penalty_mode", default="auto", choices=["auto", "manual"])
    p.add_argument("--lambda_onehot", type=float, default=None)
    p.add_argument("--lambda_site", type=float, default=None)
    p.add_argument("--lambda_chain", type=float, default=None)
    p.add_argument("--scale_target", type=float, default=1000.0)
    p.add_argument("--matrix_json", default="", help="Optional external contact_matrix JSON, especially for exact MJ.")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = export_model(
        mode=mode,
        seq=args.seq,
        out_path=args.out,
        l_size=args.l_size,
        auto_mode=args.auto_mode,
        penalty_mode=args.penalty_mode,
        lambda_onehot=args.lambda_onehot,
        lambda_site=args.lambda_site,
        lambda_chain=args.lambda_chain,
        scale_target=args.scale_target,
        matrix_json=args.matrix_json,
    )
    print(
        f"[ok] wrote {args.out} model={out['model']} dim=3 L={out['L']} "
        f"S={out['S']} V={out['V']} alpha={out['scale_alpha']:.6g}"
    )


def manhattan_coord(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def validate_pos(pos: list[int], nbr_sets: list[set[int]]) -> bool:
    if len(set(pos)) != len(pos):
        return False
    return all(pos[i + 1] in nbr_sets[pos[i]] for i in range(len(pos) - 1))


def pair_label(a: str, b: str) -> str:
    if {a, b} == {"P", "N"}:
        return "PN"
    return "".join(sorted((a, b)))


def fold_energy(
    pos: list[int],
    seq: str,
    nbr_sets: list[set[int]],
    matrix: dict[str, dict[str, float]],
) -> tuple[float, dict[str, int]]:
    counts: dict[str, int] = {"zero": 0}
    energy = 0.0
    n = len(seq)
    for i in range(n):
        ai = seq[i]
        for j in range(i + 2, n):
            if pos[j] not in nbr_sets[pos[i]]:
                continue
            aj = seq[j]
            coeff = float(matrix[ai][aj])
            energy += coeff
            if coeff == 0.0:
                counts["zero"] = counts.get("zero", 0) + 1
            else:
                key = pair_label(ai, aj)
                counts[key] = counts.get(key, 0) + 1
    return energy, counts


def local_energy_for_site(
    site: int,
    residue: int,
    pos: list[int],
    placed_count: int,
    seq: str,
    nbr_sets: list[set[int]],
    matrix: dict[str, dict[str, float]],
) -> float:
    e = 0.0
    for j in range(placed_count):
        if abs(residue - j) <= 1:
            continue
        if pos[j] in nbr_sets[site]:
            e += float(matrix[seq[residue]][seq[j]])
    return e


def random_saw(
    seq: str,
    coords: list[tuple[int, int, int]],
    nbr_sets: list[set[int]],
    matrix: dict[str, dict[str, float]],
    rng: random.Random,
    attempts: int,
) -> list[int] | None:
    s_count = len(coords)
    n = len(seq)
    for _ in range(attempts):
        pos = [rng.randrange(s_count)]
        used = {pos[0]}
        ok = True
        for r in range(1, n):
            candidates = [s for s in nbr_sets[pos[-1]] if s not in used]
            if not candidates:
                ok = False
                break
            scored = [
                (local_energy_for_site(s, r, pos, r, seq, nbr_sets, matrix), rng.random(), s)
                for s in candidates
            ]
            scored.sort()
            pool = scored[: max(1, min(4, len(scored)))]
            _, _, chosen = rng.choice(pool)
            pos.append(chosen)
            used.add(chosen)
        if ok and validate_pos(pos, nbr_sets):
            return pos
    return None


def local_move(pos: list[int], nbr_sets: list[set[int]], rng: random.Random) -> list[int] | None:
    n = len(pos)
    i = rng.randrange(n)
    occupied = set(pos)
    if i == 0:
        candidates = list(nbr_sets[pos[1]] - (occupied - {pos[0]}))
    elif i == n - 1:
        candidates = list(nbr_sets[pos[n - 2]] - (occupied - {pos[n - 1]}))
    else:
        candidates = list((nbr_sets[pos[i - 1]] & nbr_sets[pos[i + 1]]) - (occupied - {pos[i]}))
    if not candidates:
        return None
    new_pos = list(pos)
    new_pos[i] = rng.choice(candidates)
    return new_pos if validate_pos(new_pos, nbr_sets) else None


def determinant3(m: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def rotation_matrices() -> list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rows = []
            for row_idx in range(3):
                row = [0, 0, 0]
                row[perm[row_idx]] = signs[row_idx]
                rows.append(tuple(row))
            mat = tuple(rows)  # type: ignore[arg-type]
            if determinant3(mat) == 1:
                mats.append(mat)
    ident = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    return [m for m in mats if m != ident]


ROTATIONS = rotation_matrices()


def apply_rot(
    mat: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    vec: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(sum(mat[i][k] * vec[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def pivot_move(
    pos: list[int],
    coords: list[tuple[int, int, int]],
    coord_to_site: dict[tuple[int, int, int], int],
    nbr_sets: list[set[int]],
    rng: random.Random,
) -> list[int] | None:
    n = len(pos)
    if n < 4:
        return None
    pivot = rng.randrange(1, n - 1)
    rotate_suffix = rng.random() < 0.5
    mat = rng.choice(ROTATIONS)
    pivot_coord = coords[pos[pivot]]
    new_pos = list(pos)
    indices = range(pivot + 1, n) if rotate_suffix else range(0, pivot)
    for j in indices:
        old = coords[pos[j]]
        rel = (old[0] - pivot_coord[0], old[1] - pivot_coord[1], old[2] - pivot_coord[2])
        rr = apply_rot(mat, rel)
        new_coord = (pivot_coord[0] + rr[0], pivot_coord[1] + rr[1], pivot_coord[2] + rr[2])
        site = coord_to_site.get(new_coord)
        if site is None:
            return None
        new_pos[j] = site
    return new_pos if validate_pos(new_pos, nbr_sets) else None


def reptation_move(pos: list[int], nbr_sets: list[set[int]], rng: random.Random) -> list[int] | None:
    n = len(pos)
    if n < 3:
        return None
    if rng.random() < 0.5:
        occupied = set(pos[1:])
        candidates = list(nbr_sets[pos[-1]] - occupied)
        if not candidates:
            return None
        new_pos = pos[1:] + [rng.choice(candidates)]
    else:
        occupied = set(pos[:-1])
        candidates = list(nbr_sets[pos[0]] - occupied)
        if not candidates:
            return None
        new_pos = [rng.choice(candidates)] + pos[:-1]
    return new_pos if validate_pos(new_pos, nbr_sets) else None


def regrow_tail(pos: list[int], nbr_sets: list[set[int]], rng: random.Random, max_len: int) -> list[int] | None:
    n = len(pos)
    seg_len = rng.randint(1, min(max_len, max(1, n - 2)))
    grow_from_head = rng.random() < 0.5
    if grow_from_head:
        fixed = list(pos[:-seg_len])
        used = set(fixed)
        cur = fixed[-1]
        new_tail = []
        for _ in range(seg_len):
            candidates = list(nbr_sets[cur] - used)
            if not candidates:
                return None
            cur = rng.choice(candidates)
            used.add(cur)
            new_tail.append(cur)
        new_pos = fixed + new_tail
    else:
        fixed = list(pos[seg_len:])
        used = set(fixed)
        cur = fixed[0]
        new_head = []
        for _ in range(seg_len):
            candidates = list(nbr_sets[cur] - used)
            if not candidates:
                return None
            cur = rng.choice(candidates)
            used.add(cur)
            new_head.append(cur)
        new_head.reverse()
        new_pos = new_head + fixed
    return new_pos if validate_pos(new_pos, nbr_sets) else None


def bridge_regrow(pos: list[int], nbr_sets: list[set[int]], rng: random.Random, max_len: int) -> list[int] | None:
    n = len(pos)
    if n < 6:
        return None
    seg_len = rng.randint(1, min(max_len, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len - 1
    left = pos[start - 1]
    right = pos[end + 1]
    fixed = set(pos[:start] + pos[end + 1 :])
    path: list[int] = []

    def rec(prev: int, depth: int) -> bool:
        remaining = seg_len - depth
        if remaining == 0:
            return right in nbr_sets[prev]
        candidates = list(nbr_sets[prev] - fixed - set(path))
        rng.shuffle(candidates)
        for cand in candidates:
            if remaining == 1 and right not in nbr_sets[cand]:
                continue
            path.append(cand)
            if rec(cand, depth + 1):
                return True
            path.pop()
        return False

    if not rec(left, 0):
        return None
    new_pos = list(pos[:start]) + path + list(pos[end + 1 :])
    return new_pos if validate_pos(new_pos, nbr_sets) else None


def propose_move(
    pos: list[int],
    coords: list[tuple[int, int, int]],
    coord_to_site: dict[tuple[int, int, int], int],
    nbr_sets: list[set[int]],
    rng: random.Random,
    args: argparse.Namespace,
    move_attempts: dict[str, int],
) -> tuple[str, list[int] | None]:
    probs = [
        ("local", args.local_prob),
        ("pivot", args.pivot_prob),
        ("reptation", args.reptation_prob),
        ("end_regrow", args.end_regrow_prob),
        ("frag_regrow", args.frag_regrow_prob),
    ]
    total = sum(p for _, p in probs)
    u = rng.random() * total
    acc = 0.0
    move = probs[-1][0]
    for name, p in probs:
        acc += p
        if u <= acc:
            move = name
            break
    move_attempts[move] = move_attempts.get(move, 0) + 1
    if move == "local":
        return move, local_move(pos, nbr_sets, rng)
    if move == "pivot":
        return move, pivot_move(pos, coords, coord_to_site, nbr_sets, rng)
    if move == "reptation":
        return move, reptation_move(pos, nbr_sets, rng)
    if move == "end_regrow":
        return move, regrow_tail(pos, nbr_sets, rng, args.regrow_max_len)
    return move, bridge_regrow(pos, nbr_sets, rng, args.regrow_max_len)


def qubo_value_for_pos(data: dict, pos: list[int]) -> float:
    s_count = int(data["S"])
    active = {r * s_count + site for r, site in enumerate(pos)}
    q = 0.0
    for i, coeff in data["linear"]:
        if int(i) in active:
            q += float(coeff)
    for i, j, coeff in data["quadratic"]:
        if int(i) in active and int(j) in active:
            q += float(coeff)
    return q


def load_model(path: str | Path, expected_mode: str) -> tuple[dict, list[tuple[int, int, int]], dict[tuple[int, int, int], int], list[set[int]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("model") != expected_mode.upper() or int(data.get("dimension", 0)) != 3:
        raise SystemExit(f"input JSON must be a 3D {expected_mode.upper()} export")
    coords = [tuple(map(int, c)) for c in data["coords"]]
    coord_to_site = {c: i for i, c in enumerate(coords)}
    nbr_sets = neighbor_sets_from_coords(coords)
    return data, coords, coord_to_site, nbr_sets


def load_warm_start_pos(
    path: str | Path,
    seq: str,
    s_count: int,
    coord_to_site: dict[tuple[int, int, int], int],
    nbr_sets: list[set[int]],
) -> list[int] | None:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    best = raw.get("best", raw)
    warm_seq = raw.get("seq") or best.get("seq")
    if warm_seq and warm_seq != seq:
        raise SystemExit(f"warm-start sequence mismatch: expected {seq}, got {warm_seq}")

    # Prefer coordinates over raw site indices. Site indices are only stable
    # for the exact same lattice size/order, while coordinates remain valid
    # when a fold found on L=4 is used to warm-start an L=5 model.
    pos = None
    if "coords" in best:
        candidate = []
        for c in best["coords"]:
            site = coord_to_site.get(tuple(map(int, c)))
            if site is None:
                raise SystemExit(f"warm-start coordinate is outside this lattice: {c}")
            candidate.append(site)
        pos = candidate
    if pos is None and "sites" in best:
        candidate = [int(v) for v in best["sites"]]
        if len(candidate) == len(seq) and all(0 <= s < s_count for s in candidate):
            pos = candidate

    if pos is None or len(pos) != len(seq):
        raise SystemExit(f"warm-start file does not contain compatible sites/coords: {path}")
    if not validate_pos(pos, nbr_sets):
        raise SystemExit(f"warm-start fold is not a valid self-avoiding chain: {path}")
    return pos


def run_solver(args: argparse.Namespace, mode: str) -> dict:
    data, coords, coord_to_site, nbr_sets = load_model(args.ising, mode)
    seq = data["seq"]
    matrix = data["contact_matrix"]
    rng = random.Random(args.seed)
    start_time = time.perf_counter()
    warm_pos = None
    if getattr(args, "warm_start_json", ""):
        warm_pos = load_warm_start_pos(args.warm_start_json, seq, int(data["S"]), coord_to_site, nbr_sets)

    best_energy = float("inf")
    best_pos: list[int] | None = None
    best_counts: dict[str, int] | None = None
    trial_summaries = []
    move_attempts = {"local": 0, "pivot": 0, "reptation": 0, "end_regrow": 0, "frag_regrow": 0}
    move_accepts = {"local": 0, "pivot": 0, "reptation": 0, "end_regrow": 0, "frag_regrow": 0}
    warm_starts_used = 0

    for trial in range(args.trials):
        if warm_pos is not None and rng.random() < float(getattr(args, "warm_start_prob", 1.0)):
            pos = list(warm_pos)
            warm_starts_used += 1
        else:
            pos = random_saw(seq, coords, nbr_sets, matrix, rng, args.init_attempts)
        if pos is None:
            continue
        cur_energy, cur_counts = fold_energy(pos, seq, nbr_sets, matrix)
        trial_best_energy = cur_energy
        trial_best_pos = list(pos)
        trial_best_counts = dict(cur_counts)
        accepted = 0

        for step in range(args.steps):
            temp_frac = step / max(1, args.steps - 1)
            temp = args.t_init * ((args.t_final / args.t_init) ** temp_frac)
            move, cand = propose_move(pos, coords, coord_to_site, nbr_sets, rng, args, move_attempts)
            if cand is None:
                continue
            cand_energy, cand_counts = fold_energy(cand, seq, nbr_sets, matrix)
            delta = cand_energy - cur_energy
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                pos = cand
                cur_energy = cand_energy
                cur_counts = cand_counts
                accepted += 1
                move_accepts[move] = move_accepts.get(move, 0) + 1
                if cur_energy < trial_best_energy:
                    trial_best_energy = cur_energy
                    trial_best_pos = list(pos)
                    trial_best_counts = dict(cur_counts)

        if trial_best_energy < best_energy:
            best_energy = trial_best_energy
            best_pos = list(trial_best_pos)
            best_counts = dict(trial_best_counts)

        trial_summaries.append(
            {
                "trial": trial,
                "best_energy": trial_best_energy,
                "best_score": -trial_best_energy,
                "accepted": accepted,
                "accept_rate": accepted / float(args.steps),
            }
        )
        if args.stop_on_target and args.target_energy is not None and best_energy <= args.target_energy:
            break

    runtime = time.perf_counter() - start_time
    if best_pos is None or best_counts is None:
        raise SystemExit("no feasible initial chain was generated; increase L or --init_attempts")

    best_valid = validate_pos(best_pos, nbr_sets)
    best_coords = [list(coords[site]) for site in best_pos]
    qubo_check = None
    if args.check_qubo:
        q = qubo_value_for_pos(data, best_pos)
        expected = -float(data["scaled_penalties"]["lambda_onehot"]) * len(seq) + float(data["scale_alpha"]) * best_energy
        qubo_check = {
            "active_qubo_value": q,
            "expected_feasible_qubo_value": expected,
            "abs_error": abs(q - expected),
        }

    best = {
        "energy": best_energy,
        "score": -best_energy,
        "contact_counts": best_counts,
        "valid": best_valid,
        "sites": best_pos,
        "coords": best_coords,
    }
    result = {
        "format": f"{mode.lower()}_rb_qubo_sa_3d_result_v1",
        "model": mode.upper(),
        "source_ising": args.ising,
        "seq": seq,
        "alphabet": data.get("alphabet"),
        "N": len(seq),
        "L": data["L"],
        "S": data["S"],
        "dimension": 3,
        "lattice": "3d_cubic",
        "target_energy": args.target_energy,
        "hit_target": bool(args.target_energy is not None and best_energy <= args.target_energy),
        "trials_requested": args.trials,
        "trials_completed": len(trial_summaries),
        "steps": args.steps,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "move_attempts": move_attempts,
        "move_accepts": move_accepts,
        "warm_start_json": getattr(args, "warm_start_json", ""),
        "warm_start_prob": float(getattr(args, "warm_start_prob", 0.0)),
        "warm_starts_used": warm_starts_used,
        "qubo_check": qubo_check,
        "best": best,
        "trial_summaries": trial_summaries,
    }
    if args.save_best:
        out_path = Path(args.save_best)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[ok] wrote {args.save_best}")
    print(json.dumps({k: result[k] for k in ["model", "seq", "N", "L", "best", "runtime_seconds", "hit_target"]}, indent=2))
    return result


def solver_cli(mode: str) -> None:
    p = argparse.ArgumentParser(description=f"Run 3D {mode.upper()} feasible-chain RB-QUBO SA.")
    p.add_argument("--ising", required=True, help=f"Path to exported 3D {mode.upper()} QUBO/Ising JSON.")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--t_init", type=float, default=3.0)
    p.add_argument("--t_final", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_energy", type=float, default=None)
    p.add_argument("--stop_on_target", action="store_true")
    p.add_argument("--init_attempts", type=int, default=300)
    p.add_argument("--regrow_max_len", type=int, default=8)
    p.add_argument("--local_prob", type=float, default=0.35)
    p.add_argument("--pivot_prob", type=float, default=0.20)
    p.add_argument("--reptation_prob", type=float, default=0.15)
    p.add_argument("--end_regrow_prob", type=float, default=0.15)
    p.add_argument("--frag_regrow_prob", type=float, default=0.15)
    p.add_argument("--warm_start_json", default="", help="Optional best/result JSON used to seed trials.")
    p.add_argument("--warm_start_prob", type=float, default=1.0, help="Probability that a trial starts from the warm fold.")
    p.add_argument("--check_qubo", action="store_true")
    p.add_argument("--save_best", default="")
    args = p.parse_args()
    run_solver(args, mode)
