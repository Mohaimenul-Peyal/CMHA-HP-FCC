"""Matched FCC ablations. Print the plan by default; --execute launches it."""
import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

from run_ra_q_rbha_fcc import VARIANTS, parse_config, write_json
from support import PACKAGE_ROOT, DEFAULT_RUNS
sys.path.insert(0,str(PACKAGE_ROOT/'tools'))
from replay_baseline import archived_argv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',type=Path,nargs='+',required=True)
    p.add_argument('--variants',nargs='+',choices=[*VARIANTS,'original','ra_v1'],
        default=['ra_v1','fast_only','full','no_bidirectional','no_budget_schedule','no_length_effort'])
    p.add_argument('--seeds',type=int,nargs='+',default=[21013001])
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--wall-seconds',type=float,default=3600.)
    p.add_argument('--max-proposals',type=int)
    p.add_argument('--out-dir',type=Path,default=DEFAULT_RUNS)
    p.add_argument('--batch-name',default='RA_Q_RBHA_FCC_v2_matched_pilot')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.variants))!=len(a.variants):p.error('duplicate seeds or variants')
    if 'original' in a.variants and a.max_proposals is not None:p.error('historical runner has no total proposal cap; use optimized_baseline for proposal-matched tests')
    if a.batch_name in ('.','..') or any(c in a.batch_name for c in '/\\:'):p.error('invalid batch-name')
    plan=[]
    for profile in a.profiles:
        data=json.loads(profile.read_text(encoding='utf-8-sig'))
        for seed in a.seeds:
            order=list(a.variants);random.Random(seed+len(plan)).shuffle(order)
            for variant in order:
                name=f"{a.batch_name}_{variant}_L{data['l_size']}_seed{seed}"
                if variant=='original':
                    command=archived_argv(profile.parent/'run_command.ps1',a.out_dir,name,a.wall_seconds,a.workers,seed,True)
                else:
                    runner=(Path(__file__).parent.parent/'RA_Q_RBHA_FCC'/'run_ra_q_rbha_fcc.py') if variant=='ra_v1' else Path(__file__).with_name('run_ra_q_rbha_fcc.py')
                    command=[sys.executable,str(runner),
                        '--profile',str(profile.resolve()),'--variant','full' if variant=='ra_v1' else variant,'--seed',str(seed),
                        '--workers',str(a.workers),'--wall-seconds',str(a.wall_seconds),'--no-target-stop',
                        '--out-dir',str(a.out_dir.resolve()),'--run-name',name]
                    if a.max_proposals is not None:command.extend(['--max-proposals',str(a.max_proposals)])
                    parse_config(command[2:])
                folder=a.out_dir.resolve()/data['benchmark']/name
                if folder.exists():raise FileExistsError(folder)
                if any(j['run_dir'] == str(folder) for j in plan):p.error('duplicate benchmark/L/seed job; use separate batch names for distinct profiles')
                plan.append({'benchmark':data['benchmark'],'variant':variant,'seed':seed,'L':data['l_size'],
                    'profile':str(profile.resolve()),'argv':command,'run_dir':str(folder),'status':'pending'})
    print(json.dumps(plan,indent=2))
    if not a.execute:return
    control=a.out_dir.resolve()/a.batch_name
    control.mkdir(parents=True,exist_ok=False)
    write_json(control/'batch_plan.json',plan)
    rows=[]
    for job in plan:
        job['status']='running';write_json(control/'batch_status.json',plan)
        log=control/f"{job['benchmark']}_{job['variant']}_{job['seed']}.log"
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(job['argv'],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            job['status']='failed';write_json(control/'batch_status.json',plan)
            raise RuntimeError(f'Run failed; see {log}')
        folder=Path(job['run_dir'])
        summary=json.loads((folder/'q_rbha_result.json').read_text())
        # Original files use a different summary schema; never relabel worker SD as batch SD.
        workers=json.loads((folder/'worker_results.json').read_text())
        hh=[w.get('best_HH',w.get('best',{}).get('HH')) for w in workers]
        import statistics
        row={k:job[k] for k in ('benchmark','variant','seed','L','run_dir')}
        row.update(best_HH=max(hh),mean_worker_best_HH=statistics.mean(hh),
            sample_SD_worker_best_HH=statistics.stdev(hh) if len(hh)>1 else None,
            workers=len(hh),wall_budget_per_worker=a.wall_seconds,
            target_HH=summary.get('target_HH'),
            worker_target_hit_rate_pct=100*sum(c >= summary['target_HH'] for c in hh)/len(hh),
            mean_worker_wall_seconds=statistics.mean(w['wall_seconds'] for w in workers),
            statistical_unit='worker-best; row is one independently seeded batch')
        row.update(mean_proposals_per_second=statistics.mean(w.get('steps',w.get('total_steps',0))/max(w['wall_seconds'],1.e-12) for w in workers),
            earliest_best_seconds=summary.get('earliest_best_observed_wall_seconds'),
            all_workers_valid=all(w.get('valid',w.get('best',{}).get('valid',False)) for w in workers))
        rows.append(row)
        job['status']='completed';write_json(control/'batch_status.json',plan)
        write_json(control/'ablation_results.json',rows)
        with (control/'ablation_results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
