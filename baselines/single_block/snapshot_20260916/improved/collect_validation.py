"""Consolidate matched pilots, keeping the original experiments and selected CSV intact."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from validate_saved_run import validate


IWL={'S1FCC':355,'S2FCC':360,'S3FCC':366,'S4FCC':366,'R1FCC':369,'R2FCC':371,'R3FCC':373}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--refresh-control-summaries',action='store_true',help='Rebuild derived summary timings from worker records, not folds')
    a=p.parse_args();rows=[];checks=[];starts={}
    for control in a.controls:
        for job in json.loads((control/'batch_status.json').read_text()):
            if job['status']!='completed':raise RuntimeError(f'Incomplete experiment: {job}')
            folder=Path(job['run_dir'])
            cfg=json.loads((folder/'run_config.json').read_text())
            w=json.loads((folder/'worker_results.json').read_text())
            checks.append(validate(folder))
            hh=[x.get('best_HH',x.get('best',{}).get('HH')) for x in w]
            best=max(hh);times=[];events=[];pure_steps=0;steps=0;cycles=0;completed=0
            for x in w:
                if 'first_recorded' in x:
                    for count,event in x['first_recorded'].items():
                        if int(count)>=best:times.append(event['recorded_wall_seconds']);events.append(dict(event,worker=x['worker_id'],seed=x['seed']))
                else:
                    times.extend(t for c,t in x['first_recorded_seconds_by_HH'].items() if int(c)>=best)
                for c in x.get('cycle_trace',x.get('trace',[])):
                    n=c.get('steps_completed',c.get('cycle_steps_completed',0));steps+=n;cycles+=1
                    completed+=bool(c.get('complete',c.get('cycle_steps_requested')==n))
                    if c.get('cluster_multiplier',c.get('stage_cluster_multiplier'))==0:pure_steps+=n
            initial=json.loads((folder/'initial_saw_pool.json').read_text())['worker_starts']
            key=(cfg['benchmark'],cfg['l_size'],cfg['seed'])
            if key in starts:assert initial==starts[key],f'Initial SAWs differ: {folder}'
            starts[key]=initial
            row={k:job[k] for k in ('benchmark','variant','seed','L','run_dir')}
            row.update(wall_budget=cfg['wall_seconds'],workers=len(w),best_HH=best,best_HP_energy=-best,
                mean_worker_best_HH=statistics.mean(hh),sample_SD_worker_best_HH=statistics.stdev(hh) if len(hh)>1 else None,
                first_best_seconds=min(times) if times else None,
                first_best_event=min(events,key=lambda e:e['recorded_wall_seconds']) if events else None,
                IWL_best_HH=IWL.get(cfg['benchmark']),contacts_above_IWL=best-IWL[cfg['benchmark']] if cfg['benchmark'] in IWL else None,
                mean_proposals_per_second=statistics.mean(x.get('steps',x.get('total_steps',0))/x['wall_seconds'] for x in w),
                mean_annealing_wall_seconds=statistics.mean(x['wall_seconds'] for x in w),
                cycles_logged=cycles,complete_cycles=completed,pure_contact_proposal_fraction=pure_steps/steps,
                all_folds_independently_valid=True,matched_initial_SAWs=True,
                seq_sha256=hashlib.sha256(cfg['seq'].encode()).hexdigest(),
                statistical_unit='worker-best, one 16-worker batch per variant; no repeat-batch inference')
            rows.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in (('validation_comparison.json',rows),('independent_validation.json',checks)):
        (a.output_dir/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    with (a.output_dir/'validation_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows(dict(r,first_best_event=json.dumps(r['first_best_event'])) for r in rows)
    if a.refresh_control_summaries:
        by_folder={r['run_dir']:r for r in rows}
        for control in a.controls:
            path=control/'ablation_results.json';old_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            data=json.loads(path.read_text())
            for r in data:
                correct=by_folder[r['run_dir']]
                r['mean_proposals_per_second']=correct['mean_proposals_per_second']
                r['earliest_best_seconds']=correct['first_best_seconds']
            path.write_text(json.dumps(data,indent=2),encoding='utf-8')
            with (control/'ablation_results.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
            (control/'summary_rebuild_note.json').write_text(json.dumps({'reason':'Read original total_steps or RA steps; recover first-best time from worker records',
                'previous_summary_sha256':old_hash,'raw_worker_records_modified':False},indent=2),encoding='utf-8')
    for r in rows:print(r['benchmark'],r['variant'],r['best_HH'],r['mean_worker_best_HH'],r['mean_proposals_per_second'],r['first_best_seconds'])


if __name__=='__main__':main()
