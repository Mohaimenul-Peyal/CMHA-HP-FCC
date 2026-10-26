"""Bounded neutral P-interval reconstruction, not an HP-energy modification."""
import time
import numpy as np
from numba import njit

from .kernel import occupancy, beam_candidates_scored, reconstruction_reverse, full_metrics
from .diversity_restart import contact_map, map_id
from .route_diversity import polar_environment


def polar_intervals(hmask, maximum=24):
    """Partition P runs into bounded intervals; outside residues remain fixed."""
    intervals=[]
    i=0
    while i<len(hmask):
        if hmask[i]:
            i+=1
            continue
        end=i+1
        while end<len(hmask) and not hmask[end]:
            end+=1
        for a in range(i,end,maximum):
            b=min(a+maximum,end)
            if a>0 or b<len(hmask):
                intervals.append((a,b))
        i=end
    return np.asarray(intervals,dtype=np.int32).reshape((-1,2))


@njit(cache=True)
def refresh_chunk(pos,hmask,coords,neighbors,intervals,attempts,width,expansion_cap,
                  construction_floor,routing_weight):
    """Uniform finite-pool heat bath: all completed P-only states have equal E."""
    counts=np.zeros(5,np.int64)  # attempts, pools, distinct alternatives, transitions, expansions
    if len(intervals)==0:
        return counts
    occ=occupancy(pos,len(neighbors))
    for _ in range(attempts):
        q=np.random.randint(len(intervals))
        a,b=intervals[q,0],intervals[q,1]
        reverse=reconstruction_reverse(pos,a,b,coords,neighbors,occ,1)
        paths,count,expanded,_,_,_=beam_candidates_scored(
            pos,a,b,hmask,coords,neighbors,occ,-2.,construction_floor,width,True,True,
            construction_floor,routing_weight,reverse,0,0,expansion_cap)
        counts[0]+=1
        counts[4]+=expanded
        unique=np.empty(count,np.int32)
        take=0
        for k in range(count):
            same=True
            for j in range(b-a):
                if paths[k,j]!=pos[a+j]:
                    same=False
                    break
            if same:
                continue
            duplicate=False
            for prior in range(take):
                equal=True
                for j in range(b-a):
                    if paths[k,j]!=paths[unique[prior],j]:
                        equal=False
                        break
                if equal:
                    duplicate=True
                    break
            if not duplicate:
                unique[take]=k
                take+=1
        if take:
            counts[1]+=1
            counts[2]+=take
            choice=np.random.randint(take+1)  # zero is the unchanged current fold
            if choice:
                for j in range(a,b):
                    occ[pos[j]]=-1
                for j in range(a,b):
                    pos[j]=paths[unique[choice-1],j-a]
                    occ[pos[j]]=j
                counts[3]+=1
    return counts


def due(elapsed,last_improvement,last_refresh,minimum_stage,stage,patience,cooldown):
    return stage>=minimum_stage and elapsed-last_improvement>=patience and elapsed-last_refresh>=cooldown


def perform_refresh(current,hmask,coords,neighbors,intervals,cfg,deadline):
    """Validate exact core preservation; record route changes even if none occur."""
    started=time.perf_counter()
    before=current.copy()
    before_contacts=contact_map(before,hmask,neighbors)
    before_route=polar_environment(before,hmask,neighbors)
    before_metrics=full_metrics(before,hmask,neighbors,cfg['cluster_threshold'])
    stats=np.zeros(5,np.int64)
    while stats[0]<cfg['polar_refresh_attempts'] and time.perf_counter()<deadline and len(intervals):
        chunk=min(8,cfg['polar_refresh_attempts']-int(stats[0]))
        stats+=refresh_chunk(current,hmask,coords,neighbors,intervals,chunk,
            cfg['beam_width'],cfg['polar_refresh_expansion_cap'],
            cfg['construction_floor'],cfg['routing_weight'])
    after_contacts=contact_map(current,hmask,neighbors)
    hh,excess,degrees=full_metrics(current,hmask,neighbors,cfg['cluster_threshold'])
    after_route=polar_environment(current,hmask,neighbors)
    assert np.array_equal(before[hmask.astype(bool)],current[hmask.astype(bool)])
    assert before_contacts==after_contacts and hh==before_metrics[0] and excess==before_metrics[1]
    points=coords[current]
    assert len(set(map(int,current)))==len(current)
    assert np.all(np.sum(np.diff(points,axis=0)**2,axis=1)==2)
    event=dict(**dict(zip(('attempts','valid_pools','distinct_candidates','transitions','expansions'),map(int,stats))),
        refresh_wall_seconds=time.perf_counter()-started,HH_before=int(before_metrics[0]),HH_after=int(hh),
        H_sites_preserved=True,H_contact_map_preserved=True,
        contact_map_sha256=map_id(after_contacts),
        polar_environment_before_sha256=map_id(before_route),
        polar_environment_after_sha256=map_id(after_route),
        polar_pairs_changed=len(before_route^after_route),
        polar_residues_displaced=int(np.sum(before!=current)),
        positions_before=before.tolist(),positions_after=current.tolist())
    return int(hh),int(excess),degrees,event


def warm_refresh(coords,neighbors,hmask,initial):
    # Compile before starting/reseeding a worker's annealing RNG and clock.
    refresh_chunk(initial.copy(),hmask,coords,neighbors,polar_intervals(hmask),0,24,12000,.4,.15)
