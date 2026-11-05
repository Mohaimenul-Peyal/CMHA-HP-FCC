# Changelog

## v1.1.1 - 2026-11-06

- Fix: the four `profiles/ablations/without_direct_heat_bath/*.json` profiles now
  set `"transition_rule": "uniform_metropolis"`. In v1.1.0 the key was missing, so
  `run_cmha.py --ablation without_direct_heat_bath` silently ran the direct
  heat-bath rule. The runner now also passes `--transition-rule uniform_metropolis`
  for that ablation and refuses a profile without the setting. A regression test
  covers both. The published ablation results are unaffected, because the original
  runs recorded `uniform_metropolis`.
- Fix: `cmha_runner.py` reports its own version. v1.1.0 still reported 1.0.0.
- Add `baselines/single_block/`, the byte-exact source of the single-block
  predecessor used in the article's matched baseline comparison. It includes its
  twelve run configurations, and `data/single_block_baseline.json` holds the 192
  validated folds.
- Add `data/first_hit_records.json`, with first-hit times of every contact level
  for the primary, ablation and baseline runs.
- Add `tools/run_level_statistics.py`, which runs the permutation, log-rank and
  stratified tests, computes the anytime table, and writes `tables/` and
  `data/revision_statistics.json`.
- The solver engines, primary profiles and all reported results are unchanged.

## v1.1.0 - 2026-10-27

- Add all four completed independent-seed confirmations and their 64 worker folds
  and initial walks; retain weaker outcomes and count controls only once.
- Add compact proposal counters, sampled timing windows and portable regeneration.
- Revise FCC coordinates, the short paired example and nine Native-reference folds.
- Extend independent validation and tests; refresh release integrity metadata.
- Preserve the frozen optimizer and every original primary/ablation result.

## v1.0.0

Initial CMHA FCC solver and reproducibility-data release, commit 183f893.
