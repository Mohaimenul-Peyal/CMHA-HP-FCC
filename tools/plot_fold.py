#!/usr/bin/env python3
"""Render a validated saved FCC fold without rerunning optimization."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmha_results import validate_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choose = parser.add_mutually_exclusive_group(required=True)
    choose.add_argument("--benchmark")
    choose.add_argument("--fold", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.benchmark and (args.benchmark in (".", "..") or any(c in args.benchmark for c in "/\\:")):
        parser.error("Benchmark must be a single identifier")
    source = args.fold or root / "results/best_folds" / f"{args.benchmark}.json"
    fold = json.loads(source.read_text(encoding="utf-8-sig"))
    metric = validate_fold(fold["seq"], fold["coords"], fold["L"], fold["HH"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    points = np.asarray(fold["coords"])
    colors = np.array(["#d33450" if c == "H" else "#50a5c4" for c in fold["seq"]])
    fig = plt.figure(figsize=(7, 6.5))
    with plt.rc_context({"grid.color": "#d8dee4", "grid.alpha": .5}):
        ax = fig.add_axes((.04, .12, .80, .76), projection="3d")
    ax.plot(*points.T, color="#697983", linewidth=1.35, alpha=.95)
    ax.scatter(*points.T, c=colors, s=60, depthshade=True, edgecolors="white", linewidths=.45)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    span = max(3, int(np.ptp(points, axis=0).max()) + 2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axis.pane.fill = False
    ax.set_xlim(center[0]-span/2, center[0]+span/2)
    ax.set_ylim(center[1]-span/2, center[1]+span/2)
    ax.set_zlim(center[2]-span/2, center[2]+span/2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=19, azim=-53)
    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.text2D(1.04, .50, "z", transform=ax.transAxes)
    ax.set_title(f"{args.benchmark or source.stem}: N={len(points)}, L={fold['L']}, H-H={metric['HH']}", pad=18)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color="#d33450", label="H"),
                        Line2D([], [], marker="o", linestyle="", color="#50a5c4", label="P")],
               loc="lower center", ncol=2, frameon=False)
    output = args.output or root / "runs/figures" / f"{args.benchmark or source.stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
