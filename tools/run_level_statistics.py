"""Run-level statistics and tables.

Every one of the sixteen workers in an execution uses a distinct seed, starts from
its own initial walk and never exchanges information with the others, so each
worker is treated as one independent run.  This script uses frozen package data
only (``data/paper_data.json``, ``data/single_block_baseline.json``,
``data/first_hit_records.json``, ``data/confirmation_data.json`` and, when present,
``data/validated_worker_folds.csv``) and writes:

* ``tables/matched_baseline.tex``  -- single-block baseline and its differences from CMHA;
* ``tables/ablation_combined.tex`` -- ablation results with adjusted p values;
* ``tables/anytime.tex``           -- times at which runs first exceed the IWL best;
* ``tables/full_statistics.tex``   -- every comparison with effect size, raw and
  adjusted p values and log-rank tests (supplement);
* ``data/revision_statistics.json`` -- all numbers quoted in the text.

Tests: two-sided permutation test of the difference in mean ranks (equivalent to an
exact Mann--Whitney test with ties), Holm adjustment within each family, the
Vargha--Delaney A12 effect size, an instance-stratified permutation test for the
four-instance ablation panel, a log-rank test of time to a common contact level,
and the Kruskal--Wallis test across seed sets.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'data/paper_data.json').read_text())
PRIMARY = {row['benchmark']: row for row in DATA['primary']}
ABLATIONS = DATA['ablations']
BASELINE = json.loads((ROOT / 'data/single_block_baseline.json').read_text())['instances']
HITS = json.loads((ROOT / 'data/first_hit_records.json').read_text())
CONFIRMATION = json.loads((ROOT / 'data/confirmation_data.json').read_text())
PANEL = ('F180_1FCC', 'R3FCC', '3mr7', '3no3')
LEVELS = {'F180_1FCC': 378, 'R3FCC': 385, '3mr7': 350, '3no3': 474}
VARIANT_LABELS = {
    'single_block': 'Single block only', 'single_paired_block': 'Single + paired (J2)',
    'without_p_run_windows': 'Without P-run windows',
    'without_contact_conditioning': 'Without contact conditioning',
    'without_compact_core_shaping': 'Without compact-core shaping',
    'without_direct_heat_bath': 'Without direct heat bath'}
PERMUTATIONS = 200_000
RNG = np.random.default_rng(20261106)


def label(name: str) -> str:
    return name.replace('FCC', '').replace('_', r'\_')


# Small exact replacements for scipy.stats, so that the analysis needs only the
# pinned NumPy of the solver environment.
def rankdata(values) -> np.ndarray:
    """Average ranks (1-based), ties receiving the mean of their positions."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ordered = values[order]
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    return ranks


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution for df = 1 or an even df."""
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    assert df % 2 == 0, 'only df = 1 or even df are needed'
    half = x / 2
    return math.exp(-half) * sum(half ** i / math.factorial(i) for i in range(df // 2))


def kruskal_p(*groups) -> float:
    """Kruskal--Wallis test with tie correction."""
    values = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    ranks = rankdata(values)
    n = len(values)
    statistic, offset = 0.0, 0
    for group in groups:
        statistic += ranks[offset:offset + len(group)].sum() ** 2 / len(group)
        offset += len(group)
    statistic = 12 / (n * (n + 1)) * statistic - 3 * (n + 1)
    _, counts = np.unique(values, return_counts=True)
    correction = 1 - float(np.sum(counts ** 3 - counts)) / (n ** 3 - n)
    return chi2_sf(statistic / correction, len(groups) - 1)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def permutation_p(first: list[int], second: list[int]) -> float:
    """Two-sided permutation test of the difference in mean ranks."""
    values = np.asarray(first + second, dtype=float)
    if np.all(values == values[0]):
        return 1.0
    ranks = rankdata(values)
    n = len(first)
    observed = ranks[:n].mean() - ranks[n:].mean()
    order = np.argsort(RNG.random((PERMUTATIONS, len(values))), axis=1)[:, :n]
    group = ranks[order].sum(axis=1)
    diffs = group / n - (ranks.sum() - group) / (len(values) - n)
    exceed = int(np.sum(np.abs(diffs) >= abs(observed) - 1e-12))
    return (exceed + 1) / (PERMUTATIONS + 1)


def a12(first: list[int], second: list[int]) -> float:
    greater = sum(x > y for x in first for y in second)
    equal = sum(x == y for x in first for y in second)
    return (greater + 0.5 * equal) / (len(first) * len(second))


def first_time(run: dict, level: int) -> float | None:
    times = [t for contacts, t in run['events'] if contacts >= level]
    return min(times) if times else None


def logrank_p(first: list[dict], second: list[dict], level: int) -> float:
    """Two-sided log-rank test of time to reach ``level`` (censored at the budget)."""
    data = []
    for group, runs in ((0, first), (1, second)):
        for run in runs:
            t = first_time(run, level)
            data.append((t if t is not None else run['wall_s'], t is not None, group))
    event_times = sorted({t for t, event, _ in data if event})
    if not event_times:
        return 1.0
    observed_minus_expected, variance = 0.0, 0.0
    for t in event_times:
        at_risk = [(e, g) for time, e, g in data if time >= t]
        n = len(at_risk)
        n1 = sum(1 for _, g in at_risk if g == 0)
        d = sum(1 for time, e, g in data if e and time == t)
        d1 = sum(1 for time, e, g in data if e and time == t and g == 0)
        if n > 1:
            observed_minus_expected += d1 - d * n1 / n
            variance += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if variance <= 0:
        return 1.0
    return chi2_sf(observed_minus_expected ** 2 / variance, 1)


def stratified_permutation(full: dict, variant: dict) -> float:
    blocks = [rankdata(full[name] + variant[name]) for name in PANEL]
    observed = sum(r[:16].mean() - r[16:].mean() for r in blocks)
    total = np.zeros(PERMUTATIONS)
    for ranks in blocks:
        order = np.argsort(RNG.random((PERMUTATIONS, 32)), axis=1)[:, :16]
        group = ranks[order].sum(axis=1)
        total += group / 16 - (ranks.sum() - group) / 16
    exceed = int(np.sum(np.abs(total) >= abs(observed) - 1e-12))
    return (exceed + 1) / (PERMUTATIONS + 1)


def fmt_p(value: float, math_mode: bool = False) -> str:
    text = r'<\!0.001' if value < 0.001 else f'{value:.3f}'
    return f'${text}$' if math_mode or value < 0.001 else text


def write(name: str, lines: list[str]) -> None:
    (ROOT / f'tables/{name}.tex').write_text('\n'.join(lines) + '\n', encoding='ascii', newline='\n')


def ablation_values(variant: str, name: str) -> list[int]:
    return next(r for r in ABLATIONS if r['variant'] == variant and r['benchmark'] == name)['worker_best_HH']


def matched_baseline(report: dict, rows_out: list) -> None:
    names = list(BASELINE)
    pairs = [(name, [w['best_HH'] for w in BASELINE[name]['workers']], PRIMARY[name]['worker_best_HH'])
             for name in names]
    raw = [permutation_p(cmha, base) for _, base, cmha in pairs]
    adjusted = holm(raw)
    lines = [r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule',
             r'& \multicolumn{3}{c}{Single-block baseline} & \multicolumn{4}{c}{\method{} relative to baseline} \\',
             r'\cmidrule(lr){2-4}\cmidrule(l){5-8}',
             r'Instance & Best & $\overline C\pm\mathrm{SD}$ & Runs at ref. & $\Delta$Best & $\Delta\overline C$ & $A_{12}$ & $p_{\mathrm{Holm}}$ \\',
             r'\midrule']
    entries = []
    for (name, base, cmha), p_raw, p_adj in zip(pairs, raw, adjusted):
        mb, mc = statistics.mean(base), statistics.mean(cmha)
        reference = PRIMARY[name]['native_HH']
        at_reference = f'{sum(v >= reference for v in base)}/16' if reference is not None else '--'
        gain = max(cmha) - max(base)
        effect = a12(cmha, base)
        lines.append(f'{label(name)} & {max(base)} & ${mb:.2f}\\pm{statistics.stdev(base):.2f}$ & {at_reference} & '
                     f'{gain:+d} & {mc - mb:+.2f} & {effect:.2f} & {fmt_p(p_adj)} \\\\'.replace('& +0 &', '& 0 &'))
        entry = dict(benchmark=name, baseline_best=max(base), baseline_mean=mb, cmha_best=max(cmha),
                     cmha_mean=mc, delta_best=gain, delta_mean=mc - mb, a12=effect, p_raw=p_raw, p_holm=p_adj,
                     baseline_runs_at_reference=None if reference is None else sum(v >= reference for v in base))
        entries.append(entry)
        rows_out.append(('Baseline', label(name), f'{mc - mb:+.2f}', f'{effect:.2f}', fmt_p(p_raw), fmt_p(p_adj), '--'))
    lines += [r'\bottomrule', r'\end{tabular}']
    write('matched_baseline', lines)
    report['matched_baseline'] = entries


def ablations(report: dict, rows_out: list) -> None:
    full = {name: ablation_values('full_cmha_repeat', name) for name in PANEL}
    full_hits = {name: HITS['ablation']['full_cmha_repeat'][name] for name in PANEL}
    cells = []
    for variant in VARIANT_LABELS:
        for name in PANEL:
            values = ablation_values(variant, name)
            cells.append(dict(variant=variant, benchmark=name, values=values,
                              delta_mean=statistics.mean(values) - statistics.mean(full[name]),
                              a12=a12(full[name], values), p_raw=permutation_p(full[name], values),
                              logrank_raw=logrank_p(full_hits[name], HITS['ablation'][variant][name],
                                                    LEVELS[name])))
    for key, target in (('p_raw', 'p_holm'), ('logrank_raw', 'logrank_holm')):
        for cell, adjusted in zip(cells, holm([c[key] for c in cells])):
            cell[target] = adjusted
    pooled = {v: dict(p_raw=stratified_permutation(full, {n: ablation_values(v, n) for n in PANEL}))
              for v in VARIANT_LABELS}
    for variant, adjusted in zip(VARIANT_LABELS, holm([pooled[v]['p_raw'] for v in VARIANT_LABELS])):
        pooled[variant]['p_holm'] = adjusted

    def cell_text(values, p=None, logrank=None):
        parts = [f'${max(values)}$', f'${statistics.mean(values):.2f}\\pm{statistics.stdev(values):.2f}$']
        if p is not None:
            value = r'p<0.001' if p < 0.001 else f'p={p:.3f}'
            parts.append(r'$\mathbf{' + value + '}$' if p < 0.05 else f'${value}$')
        if logrank is not None:
            value = r'p_{\mathrm{T}}<0.001' if logrank < 0.001 else f'p_{{\\mathrm{{T}}}}={logrank:.3f}'
            parts.append(r'$\mathbf{' + value + '}$' if logrank < 0.05 else f'${value}$')
        return r'\shortstack[c]{' + r'\\'.join(parts) + '}'

    lines = [r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
             r'Configuration & F180\_1 & R3 & 3mr7 & 3no3 & All four \\', r'\midrule',
             'Full \\method{} & ' + ' & '.join(cell_text(full[name]) for name in PANEL) + r' & -- \\[2pt]']
    for variant, label_text in VARIANT_LABELS.items():
        row = [label_text]
        for cell in [c for c in cells if c['variant'] == variant]:
            row.append(cell_text(cell['values'], cell['p_holm'], cell['logrank_holm']))
            rows_out.append((label_text, label(cell['benchmark']), f"{cell['delta_mean']:+.2f}", f"{cell['a12']:.2f}",
                             fmt_p(cell['p_raw']), fmt_p(cell['p_holm']), fmt_p(cell['logrank_holm'])))
        overall = pooled[variant]['p_holm']
        value = r'p<0.001' if overall < 0.001 else f'p={overall:.3f}'
        row.append(r'$\mathbf{' + value + '}$' if overall < 0.05 else f'${value}$')
        lines.append(' & '.join(row) + r' \\[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('ablation_combined', lines)
    report['ablation_tests'] = dict(
        cells=[{k: v for k, v in c.items() if k != 'values'} for c in cells], stratified=pooled,
        levels=LEVELS, permutations=PERMUTATIONS)


def anytime(report: dict) -> None:
    lines = [r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
             r'& & \multicolumn{4}{c}{\method{}} & \multicolumn{3}{c}{Single-block baseline} \\',
             r'\cmidrule(lr){3-6}\cmidrule(l){7-9}',
             r'Instance & \shortstack{IWL\\best} & 10 min & 60 min & End & \shortstack{Median\\(min)} & 60 min & End & \shortstack{Median\\(min)} \\',
             r'\midrule']
    entries = []
    for name in BASELINE:
        target = PRIMARY[name]['IWL_HH'] + 1
        summary = {}
        for key, runs in (('cmha', HITS['primary'][name]), ('baseline', HITS['baseline'][name])):
            times = [first_time(run, target) for run in runs]
            hit = sorted(t / 60 for t in times if t is not None)
            summary[key] = dict(within_10=sum(t <= 10 for t in hit), within_60=sum(t <= 60 for t in hit),
                                end=len(hit), median=statistics.median(hit) if len(hit) >= 8 else None)
        reference = PRIMARY[name]['native_HH']
        reference_times = None
        if reference is not None:
            reference_times = sorted(first_time(r, reference) / 60 for r in HITS['primary'][name]
                                     if first_time(r, reference) is not None)
        entries.append(dict(benchmark=name, iwl_best=PRIMARY[name]['IWL_HH'], **{
            f'{k}_{m}': v for k, d in summary.items() for m, v in d.items()},
            cmha_reference_minutes=reference_times))
        c, b = summary['cmha'], summary['baseline']
        med = lambda v: '--' if v is None else f'{v:.1f}'
        lines.append(f"{label(name)} & {PRIMARY[name]['IWL_HH']} & {c['within_10']} & {c['within_60']} & {c['end']} & "
                     f"{med(c['median'])} & {b['within_60']} & {b['end']} & {med(b['median'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write('anytime', lines)
    report['anytime_exceed_IWL_best'] = entries
    s_times = {}
    for name in ('S1FCC', 'S3FCC', 'S4FCC'):
        target = PRIMARY[name]['IWL_HH'] + 1
        s_times[name] = max(first_time(r, target) for r in HITS['primary'][name]) / 60
    report['S_suite_last_run_exceeds_IWL_minutes'] = s_times


def full_statistics(rows: list) -> None:
    lines = [r'\begin{tabular}{@{}llrrrrr@{}}', r'\toprule',
             r'Comparison & Instance & $\Delta\overline C$ & $A_{12}$ & $p$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{T,Holm}}$ \\',
             r'\midrule']
    previous = None
    for comparison, *rest in rows:
        if previous is not None and comparison != previous:
            lines.append(r'\addlinespace[2pt]')
        lines.append(' & '.join((comparison if comparison != previous else '', *rest)) + r' \\')
        previous = comparison
    lines += [r'\bottomrule', r'\end{tabular}']
    write('full_statistics', lines)


def seed_confirmation(report: dict) -> None:
    out = {}
    for name, level in (('3mr7', 350), ('3no3', 474)):
        groups = [r['worker_HH'] for r in CONFIRMATION['runs'] if r['benchmark'] == name]
        assert len(groups) == 3 and all(len(g) == 16 for g in groups)
        pooled = [v for g in groups for v in g]
        out[name] = dict(seed_sets=len(groups), runs=len(pooled), mean=statistics.mean(pooled),
                         sd=statistics.stdev(pooled), best=max(pooled),
                         runs_at_level=sum(v >= level for v in pooled), level=level,
                         set_means=[statistics.mean(g) for g in groups],
                         kruskal_p=kruskal_p(*groups))
    report['pooled_seed_confirmation'] = out


def literature_runs(report: dict) -> None:
    report['runs_versus_IWL_best'] = [
        dict(benchmark=name, iwl_best=row['IWL_HH'],
             above=sum(v > row['IWL_HH'] for v in row['worker_best_HH']),
             equal=sum(v == row['IWL_HH'] for v in row['worker_best_HH']),
             below=sum(v < row['IWL_HH'] for v in row['worker_best_HH']))
        for name, row in PRIMARY.items() if row['IWL_HH'] is not None]


def compute_budget(report: dict) -> None:
    source = ROOT / 'data/validated_worker_folds.csv'
    if not source.exists():
        return
    totals = {}
    with source.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['study'] == 'primary':
                totals[row['benchmark']] = totals.get(row['benchmark'], 0.0) + float(row['wall_s'])
    report['thread_hours_per_instance'] = {name: value / 3600 for name, value in totals.items()}
    report['thread_hours_primary_total'] = sum(totals.values()) / 3600


def main() -> None:
    report = {'statistical_unit': 'independent run: distinct seed, own initial walk, no communication'}
    rows: list = []
    matched_baseline(report, rows)
    ablations(report, rows)
    anytime(report)
    full_statistics(rows)
    seed_confirmation(report)
    literature_runs(report)
    compute_budget(report)
    (ROOT / 'data/revision_statistics.json').write_text(json.dumps(report, indent=1) + '\n', encoding='ascii',
                                                        newline='\n')
    print('Wrote matched_baseline, ablation_combined, anytime and full_statistics tables '
          'and data/revision_statistics.json')


if __name__ == '__main__':
    main()
