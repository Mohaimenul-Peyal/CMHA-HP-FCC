# Single-block baseline (matched comparison)

This folder archives the single-block predecessor of CMHA that is used as the
matched baseline in the article. It proposes only single residue intervals and
uses the same FCC lattice boxes, wall budgets, worker seeds, initial walks,
temperature schedule, cycle lengths, block caps, compact-core shaping and
finite-pool heat-bath selection as CMHA. It lacks the multi-block constructor,
the interval guides, contact conditioning, archive restarts, P-only refresh and
vacancy guidance. It also uses its own growth-boundary and beam-width rules.

The development name of this solver was Q-ABHA. That name appears only in the
original run identifiers and inside the archived source.

## Contents

- `snapshot_20260916/`: byte-exact source snapshot used for F180_1-3 (L=12) and
  R1-R3 (L=19), three-hour runs started on 16 September 2026.
- `snapshot_20260921/`: byte-exact source snapshot used for the six CASP-derived
  instances (L=12-19), six-hour runs started on 20-21 September 2026.
  - The two snapshots differ only in `improved/run_ra_q_rbha_fcc.py`, and only in
    target-stopping options (all-workers scope and runs without a reference).
  - With the adaptive beam controller enabled, as in all twelve runs, the
    reference energy is used only for stopping.
  - The snapshot under `historical_support/` contains an unused legacy helper
    with an absolute output path. It is kept unchanged so that every file
    matches the SHA-256 values in `source_sha256.json`, which were recorded when
    the runs were made.
- `run_configs/`: the twelve recorded run configurations, with machine-specific
  paths replaced by `<original-workspace>/`.
- `../../data/single_block_baseline.json`: all 192 validated worker-best folds
  (coordinates, contacts, seeds and wall times), plus run dates and software
  versions (Python 3.12.7, NumPy 1.26.4, Numba 0.60.0). For every instance, the
  initial walks are identical to those of the primary CMHA run.
- `../../data/first_hit_records.json`: first-hit times of every contact level for
  the baseline, primary and ablation runs.

## Analysis

`python tools/run_level_statistics.py` recomputes, from the frozen data:
- the matched-baseline comparison;
- the ablation tests (permutation Mann-Whitney with Holm adjustment, A12 effect
  sizes, log-rank time-to-level tests, and an instance-stratified permutation test);
- the anytime table;
- the seed-set statistics.

The outputs are written to `tables/` and `data/revision_statistics.json`.

## Running the snapshot

The snapshot is archived for inspection and provenance. Its entry point is
`improved/run_ra_q_rbha_fcc.py` with `--profile run_configs/<instance>.json`.
It needs the same pinned Python environment as CMHA. Wall-clock control means
that re-runs will not reproduce trajectories bitwise.
