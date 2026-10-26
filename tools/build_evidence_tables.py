"""Build supplementary diagnostics and seed tables from audited existing records."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(name, headings, rows, align):
    lines = [r"\begin{tabular}{@{}" + align + r"@{}}", r"\toprule",
             " & ".join(headings) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (ROOT / "tables" / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    confirmation = json.loads((ROOT / "data/confirmation_data.json").read_text())
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    fields = ('benchmark','role','base_seed','N','L','workers','wall_s','best_HH','mean_HH',
              'SD_HH','workers_at_best','first_best_s','process_wall_s','run_id')
    with (output/'SEED_CONFIRMATION.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(confirmation['runs'])
    summaries = confirmation['summaries']
    with (output/'SEED_SET_SUMMARY.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(summaries[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summaries)
    rows = []
    for row in confirmation["runs"]:
        if row["role"] == "original_seed_control":
            continue
        rows.append((row["benchmark"], row["L"], row["base_seed"], row["best_HH"],
                     rf"${row['mean_HH']:.4f}\pm{row['SD_HH']:.3f}$",
                     f"{row['workers_at_best']}/16", f"{row['first_best_s']/60:.2f}"))
    table("seed_confirmation", ("Instance", "$L$", "Base seed", "Best", r"$\overline C\pm s_C$",
                                "At best", "First (min)"), rows, "lrrrrrr")
    with (ROOT / "data/proposal_productivity.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    rows = []
    for row in metrics:
        if row["period"] != "full_budget":
            continue
        late = next(x for x in metrics if x["benchmark"] == row["benchmark"] and
                    x["family"] == row["family"] and x["period"] == "final_hour")
        proposals = int(row["proposals"])
        rows.append((row["benchmark"].replace("FCC", "").replace("_", r"\_"),
                     {"coupled": "Paired", "quadruple": "Four"}.get(row["family"], row["family"].title()),
                     f"{proposals/1e6:.1f}", f"{100*int(row['valid_pools'])/proposals:.1f}",
                     row["physical_best_improvements"], late["physical_best_improvements"],
                     f"{float(row['mean_sampled_latency_us']):.1f}",
                     f"{float(row['improvements_per_1000_estimated_construction_seconds']):.3f}"))
    table("proposal_productivity", ("Instance", "Family", "Attempts (M)", r"Valid (\%)",
          "New bests", "Last hour", r"Latency ($\mu$s)", r"Yield / $10^3$s"), rows, "llrrrrrr")
    print("Built four additional-run rows and 16 proposal-family diagnostic rows.")


if __name__ == "__main__":
    main()
