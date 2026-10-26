# Matched component ablations

Profiles cover F180_1FCC (L=12, 3 h), R3FCC (L=19, 3 h), 3mr7 (L=13, 6 h)
and 3no3 (L=17, 6 h). All use 16 workers, original seed set 21013001 and no
target stopping. All four benchmarks must complete their configured budgets.

| Profile directory | Change from full CMHA |
| --- | --- |
| `full_cmha_repeat` | Fresh matched full-method control |
| `single_block` | Disable the whole multi-block subsystem and dependent selectors |
| `single_paired_block` | 75% single + 25% paired; disable triples/fours |
| `without_p_run_windows` | Remove P-run windows; retain ordinary contact guidance |
| `without_contact_conditioning` | Remove forced missing-pair adjacency only |
| `without_compact_core_shaping` | Zero explicit core reward; retain physical contacts and equal-contact tie-breaking |
| `without_direct_heat_bath` | Uniformly select a completed alternative, then use Metropolis acceptance |

The transition ablation uses the separate `engines/CMHA_no_heat_bath` snapshot.
Every other variant uses the unchanged full engine with its recorded profile.
There is no reverse-proposal correction in that ablation, so it is not a claim
that a generic exact Metropolis sampler has been compared.

```bash
python run_cmha.py --list-ablations
python run_cmha.py --benchmark R3FCC --ablation single_block --dry-run
python run_batch.py --ablation single_paired_block
```

The full control and selected primary runs are deliberately separate. Do not
substitute the higher primary mean into a matched control, count historical
copies twice, or interpret sixteen worker values as sixteen complete experiments.
Statistical effects are descriptive; lower SD alone does not establish improvement.
