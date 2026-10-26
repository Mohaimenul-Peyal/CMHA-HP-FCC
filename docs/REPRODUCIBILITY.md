# Reproduction protocol

The public profiles preserve the manuscript's numerical settings. Use the root
runner, not archival batch helpers under `engines/`:

```bash
python run_cmha.py --benchmark F180_1FCC
python run_cmha.py --benchmark 3mr7
python run_cmha.py --benchmark 3no3 --ablation without_direct_heat_bath
```

All benchmark positions start from fresh sequence-aware SAWs. The bundled saved
folds are validation/plotting inputs, not warm starts for these commands.
`--base-seed`, `--workers`, `--wall-seconds`, `--max-proposals` and `--target-stop`
are explicit experimental overrides, recorded in `cmha_invocation.json`.
Changing wall budget also changes controller duration limits and terminal cooling.

Use a custom profile by copying a primary JSON and changing sequence, identifier,
box and optional reference target consistently:

```bash
python run_cmha.py --profile my_profile.json --dry-run
```

## Timing and statistical unit

The configured wall allowance is per worker annealing process. Sixteen workers
run concurrently, not one after another. Initial SAW generation, JIT preparation
and final export are additional process time. First-recorded timestamps are
observed at enclosing eight-proposal chunks; proposal indices are exact.

H1-H10, F90 and S stop workers individually at their own Native reference, and
wait for all workers or their budgets. F180/R run for three hours and CASP-derived
cases for six hours without target stopping. A target-stopped zero SD is not an
unrestricted full-budget estimate. A missing target means no target-based stop.

Means and sample SDs use sixteen worker-best folds within one execution (SD
denominator 15). Whole-run variability is a different experimental quantity.
The later independent-seed confirmation is reported separately in release v1.1.0.
All four complete additional runs and the fresh controls are retained in
`data/confirmation_data.json`; the controls are not new executions. Reproduce
each benchmark/seed combination with the unchanged runner:

```bash
python run_cmha.py --benchmark 3mr7 --base-seed 42026001
python run_cmha.py --benchmark 3no3 --base-seed 42026001
python run_cmha.py --benchmark 3mr7 --base-seed 63039001
python run_cmha.py --benchmark 3no3 --base-seed 63039001
```

These are four six-hour experiments, each with sixteen parallel workers and no
target stopping. Compact archived folds are validation inputs, not warm starts.
The interrupted 3no3 attempt is excluded; the completed replacement is a fresh
full-budget execution, not checkpoint continuation.

## Repeatability and portability

Fixed seeds reproduce random inputs, but wall-time caps and terminal cooling can
change subsequent decisions under different loads or hardware. An exact replay
of a historical best or its first-hit time is not guaranteed. Fixed-proposal
software tests are useful but define a different budget from the reported runs.

Reference hardware: AMD Ryzen 7 9700X, 8 cores/16 logical processors, approximately
31.1 GiB physical RAM, Windows 11 Education. Reference software: Python 3.12.7,
NumPy 1.26.4, Numba 0.60.0 and Matplotlib 3.9.2. The included GitHub workflow tests
Python 3.12 on Linux and Windows; it does not imply that those hosted jobs were
already run before the repository was uploaded.

## Provenance

`frozen_source_manifest.json` records original and distributed hashes. Numerical
kernel, controller, initialization and transition implementations are unchanged.
Only an unused legacy helper's default output location is made relative. No
experimental values are changed. Original run IDs retain directory basenames
without requiring the original Windows machine or publishing its home paths.

The repository contains compact evidence, not gigabytes of duplicated effort
traces, checkpoints or every private experimental directory. `data/provenance.json`
preserves original evidence hashes; public-file hashes are separately recorded
in `MANIFEST.sha256`. Normalized public files need not have an original raw hash.
