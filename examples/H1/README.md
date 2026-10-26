# Measured H1 example

These files are a compact export of the selected original H1 CMHA experiment,
not a newly simulated result. All sixteen workers reached 69 nonconsecutive H-H
contacts (physical energy -69) under individual target stopping, at L=8.

`worker_results.json` preserves coordinates, IDs/seeds, stopping, wall times and
first-recorded milestones. It intentionally omits large repeated diagnostic
arrays. `initial_saw_pool.json` is the recorded initial site-index pool in the
original lexicographic even-parity FCC enumeration. `run_config.json` preserves
the recorded numerical configuration with machine paths normalized.

To reproduce the configuration from the repository root:

```bash
python run_cmha.py --benchmark H1
python tools/summarize_run.py --run-dir examples/H1 --check-only
```

The command starts new fresh SAWs and writes a new run. It does not use the
example best fold as initialization or promise identical first-hit times.
