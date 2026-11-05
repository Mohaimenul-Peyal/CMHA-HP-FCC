"""Matched warmed-kernel timing, not an IWL quality benchmark or a fresh-start experiment."""
import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from support import PACKAGE_ROOT,fcc_lattice_coords,fcc_neighbor_sets,validate_coordinates
from run_ra_q_rbha_fcc import neighbors_array
import kernel as new
sys.path.insert(0,str(PACKAGE_ROOT))
old=importlib.import_module('improved.RA_Q_RBHA_FCC.kernel')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args()
    cfg=json.loads((a.run_dir/'run_config.json').read_text(encoding='utf-8-sig'))
    saved=json.loads((a.run_dir/'best.json').read_text(encoding='utf-8-sig'))
    coords=np.array(fcc_lattice_coords(cfg['l_size']),np.int32)
    index={tuple(map(int,c)):i for i,c in enumerate(coords)}
    pos=np.array([index[tuple(c)] for c in saved['coords']],np.int32)
    h=np.array([c=='H' for c in cfg['seq']],np.int8)
    neighbors=neighbors_array(fcc_neighbor_sets([tuple(c) for c in coords]))
    rows=[]
    def execute(module,cluster,length,steps):
        cur=pos.copy();best=cur.copy()
        hh,ex,deg=module.full_metrics(cur,h,neighbors,2)
        module.seed_kernel(21013001)
        t=time.perf_counter()
        out=module.anneal_chunk(cur,best,hh,ex,deg,hh,ex,h,coords,neighbors,
            2,-2.,cluster,2,length,24,8.,.02,80000,40000,steps,10000,True,True,.4,.15,
            *((False,0) if module is new else ()))
        return time.perf_counter()-t,cur,best,out
    for length in (12,24):
        for cluster in (-.5,0.):
            for module in (old,new):execute(module,cluster,length,1)
            timings={'RA_v1':[],'v2_fast_only':[]}
            for repeat in range(a.repeats):
                modules=(('RA_v1',old),('v2_fast_only',new))
                if repeat%2:modules=modules[::-1]
                runs={}
                for name,module in modules:
                    dt,cur,best,out=execute(module,cluster,length,a.steps)
                    timings[name].append(dt);runs[name]=(cur,best,out)
                    check=validate_coordinates(cfg['seq'],coords[best],cfg['l_size'])
                    if not check['valid'] or check['HH']!=out[3]:raise AssertionError('invalid benchmark fold')
                for i in (0,1):np.testing.assert_array_equal(runs['RA_v1'][i],runs['v2_fast_only'][i])
                for i in (0,1,2,3,4):np.testing.assert_array_equal(runs['RA_v1'][2][i],runs['v2_fast_only'][2][i])
            row={'block_cap':length,'cluster_reward':cluster,'steps':a.steps,'repeats':a.repeats,
                'seconds':timings,'median_speedup':statistics.median(timings['RA_v1'])/statistics.median(timings['v2_fast_only']),
                'identical_trajectories':True}
            rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'benchmark':cfg['benchmark'],'L':cfg['l_size'],'source':str(a.run_dir),
        'purpose':'microbenchmark from a saved fold, not a new recovery result','measurements':rows},indent=2),encoding='utf-8')


if __name__=='__main__':main()
