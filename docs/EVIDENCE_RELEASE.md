# v1.1.0: Evidence and Figure Update

The numerical solver, initialization, reconstruction kernel, transition and
controller are unchanged from the initial release. The 31 primary observations,
28 ablation cells, original result tables and 944 original worker-best records
are unchanged. No new simulation was run to prepare this release.

## Complete-run confirmation

Two additional base seeds, 42026001 and 63039001, were previously run on each of
3mr7 (L=13) and 3no3 (L=17), with sixteen workers, six hours and no target stopping.
Source hashes and non-seed settings match the fresh original-seed ablation controls.

| Benchmark | Base seed | Run best | Worker mean | Worker sample SD |
| --- | ---: | ---: | ---: | ---: |
| 3mr7 | 21013001, reused control | 350 | 349.3125 | 0.4787 |
| 3mr7 | 42026001 | 350 | 349.1250 | 0.3416 |
| 3mr7 | 63039001 | 350 | 349.0625 | 0.2500 |
| 3no3 | 21013001, reused control | 474 | 472.3125 | 0.7932 |
| 3no3 | 42026001 | 473 | 471.8125 | 0.6551 |
| 3no3 | 63039001 | 473 | 471.9375 | 0.5737 |

The control rows are existing executions, counted once. The four additional
executions add 64 final folds and initial walks, yielding 1008 worker-best records
across the distributed studies. These are not 1008 independent complete runs.
All weaker completed confirmation outcomes are included. Historical directory
copies and interrupted attempts are excluded. The completed replacement of the
interrupted 3no3 run started afresh and completed six hours; it is not checkpoint
continuation. Three seeds on two hard instances do not establish suite-wide
reliability, and none reaches IWL's 353/477 contacts on those instances.

## Proposal diagnostics

The four fresh full-method ablation controls provide compact total and hourly
CSV logs in `data/proposal_diagnostics/`. Run `python tools/analyze_proposals.py`
to rebuild `data/proposal_productivity.csv` without the original machine.

Each family/worker/window has an attempt count and periodically sampled
construction latency. The estimated construction effort sums attempts times the
sampled mean latency. A sample-free short tail uses that worker/family's pooled
latency; the output records any imputed attempts. Timing is sampled once per 4096
ordinary proposal attempts, not measured exhaustively. The denominator sums
estimated construction seconds across concurrent workers, not benchmark elapsed
wall time, and excludes selection/scoring overhead. No exact CPU-time claim is made.

New-best counters count selected physical-contact improvements, not all generated
improvements or equal-contact tie breaks. Final-hour rows are subsets of the same
executions. Descriptive yield comparisons do not randomize matched states across
proposal families and do not attribute later benefits of temporary worsening moves.

## Figures and positioning

- Numeric FCC coordination shell centered at (0,0,0), with twelve correct neighbors.
- A validated synthetic 16-residue paired example with fixed anchors 1,8,9,16;
  all 25 pool states, scores and probabilities are preserved separately.
- A nine-panel gallery of Native-reference matches better than the compared IWL
  values: S1,S3,S4,F180_1-3,R1-3. It visualizes existing primary observations.

Matching a published Native reference is not a new optimality certificate. CMHA
is a finite-budget optimizer, not an exact solver or equilibrium sampler. Prior
constraint-programming work already jointly released several sequence intervals
(Dotu et al., 2011, Section 4.10, DOI: 10.1109/TCBB.2011.41). The method's technical
distinction concerns the shared stochastic beam, exact joint scoring and finite-pool
heat-bath integration, not the invention of multi-interval repair.

The manuscript, author portraits, private directories and internal reviewer notes
are not part of this code/data release.
