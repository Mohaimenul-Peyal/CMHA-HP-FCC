"""Second saved-fold check using pairwise Cartesian differences, without solver imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate(folder):
    cfg=json.loads((folder/'run_config.json').read_text(encoding='utf-8-sig'))
    records=json.loads((folder/'worker_results.json').read_text(encoding='utf-8-sig'))
    reports=[]
    for w in records:
        entry=w.get('best',w)
        xyz=entry['coords'];seq=cfg['seq'];n=len(seq)
        assert len(xyz)==n
        assert all(len(p)==3 and all(isinstance(v,(int,float)) and math.isfinite(v) and int(v)==v and 0<=v<cfg['l_size'] for v in p) and sum(p)%2==0 for p in xyz)
        assert len(set(map(tuple,xyz)))==n
        def adjacent(i,j):return sorted(abs(xyz[i][k]-xyz[j][k]) for k in range(3))==[0,1,1]
        assert all(adjacent(i,i+1) for i in range(n-1))
        contacts=[(i,j) for i in range(n) if seq[i]=='H' for j in range(i+2,n) if seq[j]=='H' and adjacent(i,j)]
        assert len(contacts)==w.get('best_HH',entry.get('HH'))
        reports.append({'worker_id':w['worker_id'],'HH':len(contacts),'valid':True})
    assert len(records)==cfg['workers']
    best=json.loads((folder/'best.json').read_text(encoding='utf-8-sig'))
    best_hh=best.get('best_HH',best.get('HH'))
    matching=[w for w in records if w.get('best',w)['coords']==best['coords'] and w.get('best_HH',w.get('best',{}).get('HH'))==best_hh]
    assert matching and best_hh==max(r['HH'] for r in reports)
    return {'run_dir':str(folder),'seq_sha256':hashlib.sha256(cfg['seq'].encode()).hexdigest(),
        'worker_results_sha256':hashlib.sha256((folder/'worker_results.json').read_bytes()).hexdigest(),
        'all_valid':True,'workers':reports}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();report=validate(a.run_dir)
    if a.output:a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
