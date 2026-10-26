# Data dictionary

`benchmarks.csv` is the canonical 31-instance index: identifier, collection,
literal HP string, N, H count, box side, Native/IWL reference contacts and sequence
SHA-256. Whitespace is not part of any sequence. Native is a published reference,
not a newly certified optimum. 3on7 has no Native target in the compared tables.

`results/PRIMARY_RESULTS.csv` contains one selected CMHA execution per benchmark.
Its `fold_file` and `profile` columns are real repository-relative artifact paths.
`original_run_id` identifies the original archived experiment, whose full logs
are not claimed to be present in this repository.

`results/RESULT_FCC_CMHA.csv` retains the detailed source-table schema: numerical
parameters, worker arrays, timings, stopping, first-best proposal information,
hashes and original-run identifiers. `results/RESULT_FCC_ablation_comparison.csv`
contains all 28 component-study cells. Machine-specific path prefixes have been
normalized; numbers and original experiment identities are unchanged.

`results/worker_best_folds.csv` has 944 rows: 496 primary and 448 ablation records.
Each includes study/variant, benchmark, worker ID/seed, best contacts, wall time,
stop reason and `coords_json`. It is not a collection of 944 independent full
batch repetitions. Worker IDs are zero based.

`data/paper_data.json` preserves analysis inputs and recorded convergence events.
`data/initial_saw_pools.json` preserves initial coordinates for all reported
workers, and `data/selected_folds.json` contains the 31 selected best folds.
`results/best_folds/*.json` exposes those same folds individually. `source` and
`run_directory` strings identify original archives rather than local dependencies.

The physical energy is `-best_HH`. Sample SD is calculated across worker-best
contacts and has the same magnitude for energies. `first_recorded_best_seconds`
means time to that row's own best, not necessarily to Native or a common threshold.
Missing values are unavailable/not applicable, not zero. Common-threshold
ablation statistics use explicitly stated target levels.

Examples are compact measured records, clearly separated from new `runs/`
outputs. Tests/smoke runs must never replace the stored publication results.
