"""Draw explanatory FCC diagrams; the synthetic example is not benchmark data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOCK_COLORS = ("#147e90", "#ba7c21")
H_COLOR, P_COLOR = "#c54258", "#6bb8cb"


def validate_example(example):
    seq, side, rows = example["sequence"], example["L"], example["pool"]
    moved = {i-1 for a, b in example["intervals_one_based"] for i in range(a, b+1)}
    fixed = [i for i in range(len(seq)) if i not in moved]
    base = np.asarray(rows[0]["coords"], dtype=int)
    scores = []
    for row in rows:
        xyz = np.asarray(row["coords"])
        assert xyz.shape == (len(seq), 3) and np.array_equal(xyz, xyz.astype(int))
        assert np.all((xyz >= 0) & (xyz < side)) and np.all(xyz.sum(axis=1) % 2 == 0)
        assert len({tuple(p) for p in xyz}) == len(seq)
        assert np.all(np.sum(np.diff(xyz, axis=0)**2, axis=1) == 2)
        assert np.array_equal(xyz[fixed], base[fixed])
        degrees = np.zeros(len(seq), dtype=int)
        hh = 0
        for i in range(len(seq)):
            for j in range(i+2, len(seq)):
                if seq[i] == seq[j] == "H" and np.sum((xyz[i]-xyz[j])**2) == 2:
                    hh += 1
                    degrees[i] += 1
                    degrees[j] += 1
        excess = sum(max(0, degrees[i]-2) for i, c in enumerate(seq) if c == "H")
        score = -2 * hh - .5 * example["cluster_multiplier"] * excess
        assert (hh, excess) == (row["HH"], row["excess"])
        assert np.isclose(score, row["effective_score"])
        scores.append(score)
    scores = np.asarray(scores)
    weights = np.exp(-(scores-scores.min()) / example["temperature"])
    weights /= weights.sum()
    assert np.allclose(weights, [r["probability"] for r in rows])
    draw = np.random.default_rng(example["categorical_draw_seed"]).choice(len(rows), p=weights)
    assert draw == example["selected_index"]
    return example


def draw_geometry(save):
    fig = plt.figure(figsize=(7.15, 4.25))
    with plt.rc_context({"grid.color": "#dde3e7", "grid.linewidth": .45}):
        ax = fig.add_axes((.12, .015, .76, .97), projection="3d", computed_zorder=False)
    points = np.asarray([(a,b,0) for a in (-1,1) for b in (-1,1)] +
                        [(a,0,b) for a in (-1,1) for b in (-1,1)] +
                        [(0,a,b) for a in (-1,1) for b in (-1,1)])
    assert len(points) == 12 and np.all(np.sum(points**2, axis=1) == 2)
    ax.add_collection3d(Line3DCollection([[np.zeros(3), p] for p in points],
        colors="#82929d", linewidths=.9, zorder=1))
    ax.scatter(*points.T, s=72, c="#2a91ad", edgecolors="white", linewidths=.65,
               depthshade=False, zorder=6)
    ax.scatter(0, 0, 0, s=125, c=H_COLOR, edgecolors="white", linewidths=.65,
               depthshade=False, zorder=7)
    ax.set(xlim=(-1.45,1.45), ylim=(-1.45,1.45), zlim=(-1.45,1.45))
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_zticks([-1,0,1])
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_xlabel("$x$", labelpad=-5, fontsize=9)
    ax.set_ylabel("$y$", labelpad=-5, fontsize=9)
    ax.set_zlabel("$z$", labelpad=-7, fontsize=9)
    ax.set_box_aspect((1,1,1))
    ax.view_init(19, -63)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_points = np.vstack((points, np.zeros(3, dtype=int)))
    projected = np.asarray([ax.transData.transform(proj3d.proj_transform(*p, ax.get_proj())[:2])
                            for p in all_points])
    placed = []
    # Place coordinates beside their own beads without callout lines.
    for index in sorted(range(13), key=lambda i: projected[i][1], reverse=True):
        point = all_points[index]
        px, py, _ = proj3d.proj_transform(*point, ax.get_proj())
        label = ax.annotate("(" + ",".join(map(str, point)) + ")", (px, py),
            xycoords="data", xytext=(0, 0), textcoords="offset points",
            fontsize=10, color="#a53043" if index == 12 else "#263842",
            bbox=dict(facecolor="white", edgecolor="none", pad=.3, alpha=.94), zorder=12)
        candidates = []
        for dx, dy in ((0,9),(0,-9),(8,6),(-8,6),(8,-6),(-8,-6),
                       (10,0),(-10,0),(0,15),(0,-15),(12,12),(-12,12),
                       (12,-12),(-12,-12),(18,3),(-18,3)):
            label.set_position((dx,dy))
            label.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
            label.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
            box = label.get_window_extent(renderer).expanded(1.05,1.12)
            collisions = sum(box.overlaps(other) for other in placed)
            collisions += sum(box.contains(*xy) for xy in projected)
            candidates.append((collisions * 1000 + abs(dx) + abs(dy), dx, dy))
        _, dx, dy = min(candidates)
        label.set_position((dx,dy))
        label.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
        label.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
        box = label.get_window_extent(renderer).expanded(1.05,1.12)
        assert not any(box.overlaps(other) for other in placed), "Overlapping FCC coordinate labels"
        placed.append(box)
    save(fig, "fig01_fcc_geometry")


def draw_sequence(ax, example):
    seq = example["sequence"]
    ax.set(xlim=(.1,len(seq)+.9), ylim=(-.8,1.65))
    ax.axis("off")
    for block, ((a, b), color) in enumerate(zip(example["intervals_one_based"], BLOCK_COLORS), 1):
        ax.add_patch(Rectangle((a-.48,-.30), b-a+.96, .92, facecolor=color, alpha=.08,
                               edgecolor="none", zorder=0))
        ax.plot([a-.35,b+.35],[-.38,-.38], color=color, lw=1.5)
        ax.plot([a-.35,a-.35],[-.38,-.23], color=color, lw=1.5)
        ax.plot([b+.35,b+.35],[-.38,-.23], color=color, lw=1.5)
        ax.text((a+b)/2, -.65, rf"$B_{block}=[{a},{b}]$", color=color,
                ha="center", va="center", fontsize=9)
    x = np.arange(1,len(seq)+1)
    ax.plot(x, np.zeros(len(seq)), color="#aab4bc", lw=1.05, zorder=1)
    for i, letter in enumerate(seq, 1):
        ax.scatter(i, 0, s=126, color=H_COLOR if letter == "H" else P_COLOR,
                   edgecolors="white", linewidths=.6, zorder=3)
        ax.text(i, 0, letter, color="white", fontsize=7.2, ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(i, .73, str(i), color="#58666f", fontsize=6.7, ha="center", va="center")
    boundaries = example["fixed_boundary_indices_one_based"]
    ax.scatter(boundaries, np.zeros(len(boundaries)), s=205, facecolors="none", edgecolors="#202c34",
               linewidths=1.1, zorder=5)


def draw_fold(ax, example, row, selected=False):
    points = np.asarray(row["coords"], dtype=float)
    moved = {i-1 for a,b in example["intervals_one_based"] for i in range(a,b+1)}
    fixed = np.asarray([i for i in range(len(points)) if i not in moved])
    segments = np.asarray([[points[i],points[i+1]] for i in range(len(points)-1)
                           if i not in moved and i+1 not in moved])
    ax.add_collection3d(Line3DCollection(segments, colors="#aeb6bd", linewidths=1.0, alpha=.8, zorder=1))
    ax.scatter(*points[fixed].T, s=17, color="#c1c8ce", alpha=.85, depthshade=False, zorder=3)
    for (a,b), color in zip(example["intervals_one_based"], BLOCK_COLORS):
        connection = points[a-2:b+1]
        ax.add_collection3d(Line3DCollection(np.stack([connection[:-1],connection[1:]],axis=1),
            colors=color, linewidths=1.8, linestyles="solid" if selected else "dashed", zorder=2))
        movable = points[a-1:b]
        hues = [H_COLOR if c == "H" else P_COLOR for c in example["sequence"][a-1:b]]
        ax.scatter(*movable.T, s=33, c=hues, depthshade=False, edgecolors="white", linewidths=.45, zorder=4)
    boundaries = np.asarray(example["fixed_boundary_indices_one_based"])-1
    ax.scatter(*points[boundaries].T, s=54, c="white", edgecolors="#263842", linewidths=1.0,
               depthshade=False, zorder=10)
    if selected:
        ax.add_collection3d(Line3DCollection([[points[first-1],points[second-1]]
            for first,second in row["new_interblock_contacts"]],
            colors="#b62a58", linewidths=2, linestyles="dotted", zorder=6))
    union = np.vstack([example["pool"][0]["coords"],example["pool"][example["selected_index"]]["coords"]])
    center = (union.min(axis=0)+union.max(axis=0))/2
    span = float(np.ptp(union,axis=0).max()) + 1.0
    for setter, middle in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
        setter(middle-span/2,middle+span/2)
    ax.set_box_aspect((1,1,1))
    ax.view_init(24,-57)
    for i in boundaries:
        px,py,_=proj3d.proj_transform(*points[i],ax.get_proj())
        ax.annotate(str(i+1),(px,py),xycoords="data",xytext=(5,5),textcoords="offset points",
            ha="left",va="bottom",color="#111111",fontsize=7.4,
            bbox=dict(facecolor="white",edgecolor="none",pad=.15),zorder=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
    ax.grid(False)


def draw_multiblock(save):
    example = validate_example(json.loads((ROOT / "data/schematic_multiblock_example.json").read_text()))
    chosen = example["pool"][example["selected_index"]]
    current = example["pool"][0]
    fig = plt.figure(figsize=(7.15,5.45))
    fig.text(.035,.975,"(a) Illustrative HP sequence and fixed boundaries",fontsize=10,fontweight="bold",va="top")
    draw_sequence(fig.add_axes((.025,.735,.95,.19)),example)
    handles = [Line2D([],[],marker="o",linestyle="",color=H_COLOR,label="H"),
               Line2D([],[],marker="o",linestyle="",color=P_COLOR,label="P"),
               Line2D([],[],marker="o",linestyle="",markerfacecolor="white",markeredgecolor="#263842",label="Fixed boundary"),
               Line2D([],[],color=BLOCK_COLORS[0],lw=2,label=r"Movable $B_1$"),
               Line2D([],[],color=BLOCK_COLORS[1],lw=2,label=r"Movable $B_2$")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.5,.715),ncol=5,frameon=False,
               fontsize=8,handlelength=1.4,columnspacing=1.0)
    fig.text(.035,.675,rf"(b) Current fold: $C_{{\rm HH}}={current['HH']}$",fontsize=9.4,fontweight="bold")
    fig.text(.65,.675,rf"(d) Shown draw: $C_{{\rm HH}}={chosen['HH']}$",fontsize=9.4,fontweight="bold")
    with plt.rc_context({"grid.color":"#e4e8eb"}):
        before=fig.add_axes((.005,.273,.355,.405),projection="3d",computed_zorder=False)
        after=fig.add_axes((.64,.273,.355,.405),projection="3d",computed_zorder=False)
    draw_fold(before,example,current)
    draw_fold(after,example,chosen,selected=True)
    ax=fig.add_axes((.385,.315,.23,.32))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.5,1.02,"(c) Joint reconstruction",ha="center",fontsize=9.2,fontweight="bold")
    ax.text(.5,.91,"One partial state (schematic)",ha="center",fontsize=7.5,color="#53616b")
    ax.text(.23,.81,r"$B_1$",ha="center",fontsize=9,color=BLOCK_COLORS[0])
    ax.text(.77,.81,r"$B_2$",ha="center",fontsize=9,color=BLOCK_COLORS[1])
    for row,(n1,n2) in enumerate(((1,0),(1,1),(2,1),(2,2))):
        y=.69-row*.14
        for n,start,color in ((n1,.035,BLOCK_COLORS[0]),(n2,.575,BLOCK_COLORS[1])):
            xs=start+np.arange(6)*.075
            ax.plot(xs,[y]*6,color="#cbd2d8",lw=.7,zorder=1)
            ax.scatter(xs,[y]*6,s=12,c=[color if j<n else "#e4e8eb" for j in range(6)],zorder=3)
    ax.text(.5,.15,r"$\vdots$",ha="center",fontsize=14)
    ax.text(.5,.02,"Alternate extensions; enforce\nboth endpoint links and no overlap",ha="center",
            va="top",fontsize=7.6,linespacing=1.45)
    for start,end in (((.345,.49),(.385,.49)),((.615,.49),(.655,.49))):
        fig.add_artist(FancyArrowPatch(start,end,transform=fig.transFigure,arrowstyle="-|>",
            mutation_scale=12,lw=1,color="#465b67"))
    fig.text(.03,.26,"Gray chain stays fixed; black rings identify the same four anchors in both folds.",
             fontsize=8,color="#46555f")
    fig.text(.03,.218,rf"(e) Direct heat-bath draw from all {len(example['pool'])} feasible pool members"
        + rf"  ($T={example['temperature']}$, $\lambda_s={example['cluster_multiplier']}$)",fontsize=9,fontweight="bold")
    table_ax=fig.add_axes((.03,.018,.94,.177));table_ax.axis("off")
    representative=next(i for i in range(1,len(example["pool"])) if i != example["selected_index"])
    indices=(0,example["selected_index"],representative)
    rows=[]
    for index,title in zip(indices,("Current fold","Shown draw","Another alternative")):
        row=example["pool"][index]
        probability = "<0.0001" if row['probability'] < .00005 else f"{row['probability']:.4f}"
        rows.append((title,str(row["HH"]),str(row["excess"]),f"{row['effective_score']:.2f}",probability))
    remainder=1-sum(example["pool"][i]["probability"] for i in indices)
    rows.append((f"Other {len(example['pool'])-3} alternatives","--","--","--",f"{remainder:.4f}"))
    table=table_ax.table(cellText=rows,colLabels=("Pool entry",r"$C_{\rm HH}$",r"$C_{\rm ex}$",r"$F_s$",r"$p_j$"),
        cellLoc="center",colWidths=(.40,.14,.14,.16,.16),bbox=(0,0,1,1),edges="horizontal")
    table.auto_set_font_size(False);table.set_fontsize(8)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#bac4cc");cell.set_linewidth(.45)
        if row==0:cell.set_text_props(weight="bold")
        if col==0:cell.set_text_props(ha="left")
        if row==2:cell.set_text_props(weight="bold",color="#086772")
    save(fig,"fig02_multiblock")


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42})
    def save(fig,name):
        fig.savefig(ROOT / f"figures/{name}.pdf",bbox_inches="tight",pad_inches=.04)
        fig.savefig(ROOT / f"figures/{name}.png",dpi=500,bbox_inches="tight",pad_inches=.04)
        plt.close(fig)
    draw_geometry(save)
    draw_multiblock(save)
    print("Rebuilt Figures 1 and 2; validated the complete synthetic candidate pool.")


if __name__=="__main__":
    main()
