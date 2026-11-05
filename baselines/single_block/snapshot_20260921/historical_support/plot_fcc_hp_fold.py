#!/usr/bin/env python3
"""Render a validated 3D FCC HP fold with fixed lattice scaling."""

from __future__ import annotations

import argparse
import json
from itertools import permutations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


H_COLOR = "#d73027"
P_COLOR = "#2878b5"
CHAIN_COLOR = "#59636e"
CONTACT_COLOR = "#d99a00"


def fcc_displacements() -> set[tuple[int, int, int]]:
    values: set[tuple[int, int, int]] = set()
    for zero_axis in range(3):
        other = [axis for axis in range(3) if axis != zero_axis]
        for signs in product((-1, 1), repeat=2):
            delta = [0, 0, 0]
            delta[other[0]] = signs[0]
            delta[other[1]] = signs[1]
            values.add(tuple(delta))
    return values


FCC_DELTAS = fcc_displacements()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best_json", required=True)
    parser.add_argument("--out", required=True, help="PNG output path")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--l_size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--view_elev", type=float, default=24.0)
    parser.add_argument("--view_azim", type=float, default=38.0)
    parser.add_argument("--hide_contacts", action="store_true")
    return parser.parse_args()


def load_best(path: Path) -> tuple[dict, str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best", data)
    seq = str(best.get("seq") or data.get("seq") or "").upper()
    coords = np.asarray(best.get("coords"), dtype=float)
    if not seq or coords.shape != (len(seq), 3):
        raise SystemExit("best JSON must contain seq and an N x 3 coords array")
    return best, seq, coords


def is_fcc_neighbor(a: np.ndarray, b: np.ndarray) -> bool:
    delta = tuple(int(x) for x in (b - a))
    return delta in FCC_DELTAS


def hh_contacts(seq: str, coords: np.ndarray) -> list[tuple[int, int]]:
    contacts: list[tuple[int, int]] = []
    for i in range(len(seq)):
        if seq[i] != "H":
            continue
        for j in range(i + 2, len(seq)):
            if seq[j] == "H" and is_fcc_neighbor(coords[i], coords[j]):
                contacts.append((i, j))
    return contacts


def render(best: dict, seq: str, coords: np.ndarray, args: argparse.Namespace) -> None:
    contacts = hh_contacts(seq, coords)
    recorded = int(best.get("HH", best.get("score", len(contacts))))
    if len(contacts) != recorded:
        raise SystemExit(f"independent HH count {len(contacts)} != recorded {recorded}")

    fig = plt.figure(figsize=(9.0, 8.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.view_init(elev=args.view_elev, azim=args.view_azim)

    if not args.hide_contacts:
        for i, j in contacts:
            points = coords[[i, j]]
            ax.plot(
                points[:, 0], points[:, 1], points[:, 2],
                color=CONTACT_COLOR, linewidth=0.75, alpha=0.24,
                linestyle=(0, (2.0, 2.0)), zorder=1,
            )

    ax.plot(
        coords[:, 0], coords[:, 1], coords[:, 2],
        color=CHAIN_COLOR, linewidth=2.0, alpha=0.88, zorder=2,
    )

    hmask = np.asarray([residue == "H" for residue in seq])
    ax.scatter(
        coords[~hmask, 0], coords[~hmask, 1], coords[~hmask, 2],
        s=150, c=P_COLOR, edgecolors="white", linewidths=0.7,
        depthshade=True, alpha=0.96, zorder=3,
    )
    ax.scatter(
        coords[hmask, 0], coords[hmask, 1], coords[hmask, 2],
        s=205, c=H_COLOR, edgecolors="white", linewidths=0.8,
        depthshade=True, alpha=0.98, zorder=4,
    )

    lower, upper = -0.5, float(args.l_size) - 0.5
    ticks = np.arange(0, args.l_size, 1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_zlim(lower, upper)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(labelsize=8, pad=1)
    ax.set_xlabel("x", labelpad=7)
    ax.set_ylabel("y", labelpad=7)
    ax.set_zlabel("z", labelpad=7)
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.975, 0.98, 0.985, 1.0))
        axis.pane.set_edgecolor((0.78, 0.80, 0.82, 1.0))
    ax.grid(True, color="#cfd4da", linewidth=0.55, alpha=0.72)

    benchmark = args.benchmark or str(best.get("benchmark", "FCC benchmark"))
    ax.set_title(
        f"{benchmark}: 3D FCC HP ground-state fold\n"
        f"$C_{{HH}}={len(contacts)}$, $E_{{HP}}=-{len(contacts)}$, $L={args.l_size}$",
        fontsize=13, pad=16,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=H_COLOR,
               markeredgecolor="white", markersize=10, label="Hydrophobic (H)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=P_COLOR,
               markeredgecolor="white", markersize=9, label="Polar (P)"),
        Line2D([0], [0], color=CHAIN_COLOR, linewidth=2.0, label="Chain bond"),
    ]
    if not args.hide_contacts:
        handles.append(Line2D([0], [0], color=CONTACT_COLOR, linewidth=1.1,
                              linestyle=(0, (2, 2)), label="Nonbonded H-H contact"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
              frameon=True, framealpha=0.94, fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({
        "benchmark": benchmark,
        "HH": len(contacts),
        "png": str(out),
        "pdf": str(out.with_suffix('.pdf')),
    }))


def main() -> None:
    args = parse_args()
    best, seq, coords = load_best(Path(args.best_json))
    render(best, seq, coords, args)


if __name__ == "__main__":
    main()
