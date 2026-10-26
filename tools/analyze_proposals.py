"""Regenerate sampled proposal-family productivity from bundled control logs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('single', 'coupled', 'triple', 'quadruple')
COUNTERS = ('proposals', 'valid_pools', 'distinct_candidates', 'physical_best_improvements',
            'pools_beating_previous_worker_best')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def analyze(root=ROOT):
    data = json.loads((root/'data/paper_data.json').read_text())
    controls = [r for r in data['ablations'] if r['variant']=='full_cmha_repeat']
    result = []
    for control in controls:
        benchmark = control['benchmark']
        folder = root/'data/proposal_diagnostics'/benchmark
        totals = read_csv(folder/'proposal_statistics.csv')
        windows = read_csv(folder/'proposal_windows.csv')
        for family in FAMILIES:
            cohort = [r for r in windows if r['proposal_type']==family]
            reference = [r for r in totals if r['proposal_type']==family]
            for field in COUNTERS:
                if sum(int(r[field]) for r in cohort) != sum(int(r[field]) for r in reference):
                    raise ValueError(f'Counter mismatch: {benchmark}/{family}/{field}')
            pooled = {}
            for worker in range(16):
                rows = [r for r in cohort if int(r['worker_id'])==worker]
                n = sum(int(r['timing_samples']) for r in rows)
                pooled[worker] = sum(float(r['sampled_construction_seconds']) for r in rows)/n if n else None
            for period in ('full_budget', 'final_hour'):
                selected = cohort if period=='full_budget' else [r for r in cohort
                    if float(r['wall_start']) >= control['wall_s']-3600-1]
                aggregate = {k:sum(int(r[k]) for r in selected) for k in COUNTERS+('timing_samples',)}
                estimate = 0.0
                imputed = 0
                for row in selected:
                    attempts, samples = int(row['proposals']), int(row['timing_samples'])
                    if not attempts:
                        continue
                    mean = float(row['sampled_construction_seconds'])/samples if samples else pooled[int(row['worker_id'])]
                    if mean is None or mean <= 0:
                        raise ValueError('No positive construction-latency estimate available')
                    estimate += attempts*mean
                    if not samples:
                        imputed += attempts
                result.append(dict(benchmark=benchmark,period=period,family=family,**aggregate,
                    estimated_construction_seconds=estimate,
                    proposals_with_pooled_latency_fallback=imputed,
                    mean_sampled_latency_us=1e6*estimate/aggregate['proposals'] if aggregate['proposals'] else None,
                    valid_pools_per_estimated_construction_second=aggregate['valid_pools']/estimate if estimate else None,
                    improvements_per_1000_estimated_construction_seconds=1000*aggregate['physical_best_improvements']/estimate if estimate else None,
                    interpretation='Periodic 1/4096 construction samples; not an exact CPU or end-to-end wall measurement'))
    return result


def main():
    rows = analyze()
    output = ROOT/'data/proposal_productivity.csv'
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Regenerated {len(rows)} full/final-hour diagnostic rows; no simulations run.')


if __name__=='__main__':
    main()
