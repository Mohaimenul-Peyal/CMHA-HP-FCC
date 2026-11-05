#!/usr/bin/env python3
"""Q-ABHA-FCC: QUBO-guided Adaptive Beam Heat-bath Annealing.

The validated RA-Q-RBHA-FCC-v2 numerical implementation is retained unchanged.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(env, '1')

import numpy as np
import numba
if __package__:
    from .controller import EffortController, budget_stage, capped_cycle
    from .support import (PACKAGE_ROOT, DEFAULT_RUNS, HISTORICAL_SOURCE, export_lightweight_model,
                          fcc_lattice_coords, fcc_neighbor_sets, model_context,
                          pairwise_contact_saw, validate_coordinates)
else:
    from controller import EffortController, budget_stage, capped_cycle
    from support import (PACKAGE_ROOT, DEFAULT_RUNS, HISTORICAL_SOURCE, export_lightweight_model,
                         fcc_lattice_coords, fcc_neighbor_sets, model_context,
                         pairwise_contact_saw, validate_coordinates)

VARIANTS = {
    'full': (True, True, True),
    'optimized_baseline': (False, False, False),
    'no_routing': (False, True, True),
    'no_diversity': (True, False, True),
    'no_adaptive_effort': (True, True, False),
    'routing_only': (True, False, False),
    'diversity_only': (False, True, False),
    'adaptive_effort_only': (False, False, True),
    'fast_only': (True, True, True),
    'no_bidirectional': (True, True, True),
    'no_budget_schedule': (True, True, True),
    'no_length_effort': (True, True, True),
}
DEFAULTS = dict(workers=16, wall_seconds=3600.0, seed=21013001, worker_seed_stride=1000003,
                cycle_steps=[80000,160000,320000,640000], cycles_without_improvement=2,
                temperature_t0=8.0, temperature_t1=0.02, stage_block_max=[12,16,20,24],
                cluster_multipliers=[1.0,0.66,0.33,0.0], cluster_threshold=2,
                pair_hh_reward=-2.0, cluster_excess_reward=-0.5, min_block_len=2,
                beam_width=24, near_target_beam_width=48, near_target_gap=4,
                init_attempts=8000, init_h_pool=2, init_p_pool=4, init_random_pool_prob=0.1,
                control_chunk_steps=8, target_stop_enabled=False,
                routing_weight=0.15, construction_floor=0.4, adaptive_window_steps=8000,
                max_adaptive_beam=96, max_proposals_per_worker=None)


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def neighbors_array(sets):
    result = np.full((len(sets), 12), -1, np.int32)
    for i, values in enumerate(sets):
        for j, v in enumerate(sorted(values)):
            result[i, j] = v
    return result


def load_kernel():
    import importlib
    return importlib.import_module('.kernel',__package__) if __package__ else importlib.import_module('kernel')


def warm_kernel():
    k=load_kernel()
    coords = np.asarray(fcc_lattice_coords(3), np.int32)
    neighbors = neighbors_array(fcc_neighbor_sets([tuple(p) for p in coords]))
    pos = np.array([0, 1, 2, 3], np.int32)
    # Find a tiny valid path instead of assuming a particular site ordering.
    path = [0]
    while len(path) < 4:
        path.append(next(int(v) for v in neighbors[path[-1]] if v >= 0 and v not in path))
    pos[:] = path
    h = np.ones(4, np.int8)
    hh, ex, deg = k.full_metrics(pos, h, neighbors, 2)
    k.seed_kernel(1)
    k.anneal_chunk(pos, pos.copy(), hh, ex, deg, hh, ex, h, coords, neighbors,
                 2, -2., -.5, 2, 3, 4, 8., .02, 4, 0, 1, 100, True, True, .4, .15, True, 4)


def worker(task):
    k=load_kernel()
    full_metrics,seed_kernel,anneal_chunk=k.full_metrics,k.seed_kernel,k.anneal_chunk
    cfg, wid = task['config'], task['worker_id']
    folder = Path(task['folder'])
    seed = cfg['seed'] + wid*cfg['worker_seed_stride']
    prep = time.perf_counter()
    coords = np.asarray(task['coords'], np.int32)
    neighbors = neighbors_array(fcc_neighbor_sets([tuple(p) for p in coords]))
    hmask = np.asarray([c == 'H' for c in cfg['seq']], np.int8)
    current = np.asarray(task['initial'], np.int32)
    hh, excess, degrees = full_metrics(current, hmask, neighbors, cfg['cluster_threshold'])
    best, best_hh, best_excess = current.copy(), int(hh), int(excess)
    warm_kernel()
    seed_kernel(seed)
    prep_seconds = time.perf_counter()-prep
    routing, diversity, adaptive = VARIANTS[cfg['variant']]
    ctl = EffortController(width=cfg['beam_width'], base=cfg['beam_width'],
                           maximum=cfg['max_adaptive_beam'], window_steps=cfg['adaptive_window_steps'],
                           next_check=cfg['adaptive_window_steps'])
    totals, window = np.zeros(9, np.int64), np.zeros(9, np.int64)
    window_capacity = 0
    first = {str(best_hh): {'proposal': 0, 'recorded_wall_seconds': 0.0, 'stage': 0, 'cycle': 0}}
    cycle_rows, effort_rows = [], []
    stage, cycle, stagnant, total_steps = 0, 0, 0, 0
    rate = None
    stop_hh = cfg['target_HH'] if cfg['target_stop_enabled'] else np.iinfo(np.int32).max
    event = task.get('stop_event')
    abort_event = task.get('abort_event')
    last_abort_check = -1.0
    reason = 'wall_budget'
    started = time.perf_counter()
    last_progress = -10.0
    done = cfg['target_stop_enabled'] and best_hh >= cfg['target_HH']
    if done:
        reason = 'target_hit'
        if event is not None:
            event.set()
    while not done:
        elapsed = time.perf_counter()-started
        if cfg['budget_schedule']:
            required = budget_stage(elapsed,cfg['wall_seconds'],total_steps,cfg['max_proposals_per_worker'])
            if required>stage:
                stage,stagnant = required,0
                current = best.copy()
                hh,excess,degrees = full_metrics(current,hmask,neighbors,cfg['cluster_threshold'])
        requested_length = (cfg['cycle_steps'][stage] if stage < len(cfg['cycle_steps']) else
                  cfg['cycle_steps'][-1] * 2**(stage-len(cfg['cycle_steps'])+1))
        length = capped_cycle(requested_length,stage,elapsed,cfg['wall_seconds'],total_steps,
                              cfg['max_proposals_per_worker'],rate) if cfg['budget_schedule'] else requested_length
        cluster = cfg['cluster_excess_reward'] * cfg['cluster_multipliers'][min(stage,3)]
        max_block = min(len(current)-1, cfg['stage_block_max'][min(stage,3)])
        start_hh, offset = best_hh, 0
        cycle_started = time.perf_counter()-started
        budget_interrupted = False
        while offset < length:
            elapsed = time.perf_counter()-started
            if elapsed >= cfg['wall_seconds']:
                done, reason = True, 'wall_budget'
                break
            if abort_event is not None and elapsed-last_abort_check>=1.0:
                last_abort_check=elapsed
                if abort_event.is_set():
                    done,reason=True,'peer_worker_failed'
                    break
            if cfg['target_stop_enabled'] and event is not None and event.is_set():
                done, reason = True, 'global_target_stop'
                break
            remaining = cfg['max_proposals_per_worker']
            if remaining is not None and total_steps >= remaining:
                done, reason = True, 'proposal_budget'
                break
            if cfg['budget_schedule'] and budget_stage(elapsed,cfg['wall_seconds'],total_steps,remaining)>stage:
                budget_interrupted = True
                break
            chunk = min(cfg['control_chunk_steps'], length-offset)
            if remaining is not None:
                chunk = min(chunk, remaining-total_steps)
                if cfg['budget_schedule'] and stage<3:
                    chunk = min(chunk,max(1,math.ceil((stage+1)*remaining/4)-total_steps))
            width = ctl.width if adaptive else (
                cfg['near_target_beam_width'] if best_hh >= cfg['target_HH']-cfg['near_target_gap'] else cfg['beam_width'])
            previous_best = best_hh
            hh, excess, degrees, best_hh, best_excess, stats, milestones = anneal_chunk(
                current, best, hh, excess, degrees, best_hh, best_excess, hmask, coords, neighbors,
                cfg['cluster_threshold'], cfg['pair_hh_reward'], cluster, cfg['min_block_len'],
                max_block, width, cfg['temperature_t0'], cfg['temperature_t1'], length, offset,
                chunk, stop_hh, routing, diversity, cfg['construction_floor'], cfg['routing_weight'],
                cfg['bidirectional'],cfg['beam_width'] if cfg['length_effort'] else 0)
            elapsed = time.perf_counter()-started
            for proposal, contacts in milestones:
                first.setdefault(str(contacts), {'proposal': total_steps+int(proposal),
                    'recorded_wall_seconds': elapsed, 'stage': stage, 'cycle': cycle})
            total_steps += int(stats[0])
            offset += int(stats[0])
            totals += stats
            window += stats
            window_capacity += int(stats[8])
            if adaptive:
                due = total_steps >= ctl.next_check or best_hh > previous_best
                old_width = ctl.width
                why = ctl.update(total_steps, best_hh > previous_best,
                    float(window[1])/max(1,int(window[0])), float(window[3])/max(1,int(window[2])),
                    float(window[2])/max(1,window_capacity))
                if due:
                    effort_rows.append({'proposal': total_steps, 'wall_seconds': elapsed,
                        'best_HH': int(best_hh), 'old_beam': old_width, 'new_beam': ctl.width,
                        'reason': why, 'valid_pool_fraction': float(window[1])/max(1,int(window[0])),
                        'signature_diversity': float(window[3])/max(1,int(window[2]))})
                    window[:] = 0
                    window_capacity = 0
            if elapsed-last_progress >= 5.0:
                write_json(folder / f'worker_{wid:02d}_progress.json', {
                    'worker_id': wid, 'seed': seed, 'best_HH': int(best_hh), 'current_HH': int(hh),
                    'elapsed_seconds': elapsed, 'stage': stage, 'cycle': cycle,
                    'cycle_steps': length, 'cycle_offset': offset, 'steps': total_steps,
                    'beam_width': width, 'best_positions': best.tolist(), 'first_recorded': first})
                last_progress = elapsed
            if cfg['target_stop_enabled'] and best_hh >= cfg['target_HH']:
                done, reason = True, 'target_hit'
                if event is not None:
                    event.set()
                break
        improved = best_hh > start_hh
        complete = offset == length
        cycle_wall = time.perf_counter()-started-cycle_started
        cycle_rows.append({'worker_id':wid, 'stage':stage, 'cycle':cycle, 'cycle_steps':length,
            'requested_cycle_steps':requested_length,'budget_stage_interrupted':budget_interrupted,
            'rate_used_for_cycle_cap':rate,'measured_proposals_per_second':offset/max(cycle_wall,1.e-12),
            'steps_completed':offset, 'complete':complete, 'start_HH':int(start_hh), 'best_HH':int(best_hh),
            'wall_start':cycle_started, 'wall_end':time.perf_counter()-started,
            'cluster_multiplier':cfg['cluster_multipliers'][min(stage,3)], 'max_block_len':max_block})
        cycle += 1
        if offset and cycle_wall>0:
            measured = offset/cycle_wall
            rate = measured if rate is None else .5*rate+.5*measured
        if complete:
            stagnant = 0 if improved else stagnant+1
            if stagnant >= cfg['cycles_without_improvement']:
                stage += 1
                stagnant = 0
                current = best.copy()
                hh, excess, degrees = full_metrics(current, hmask, neighbors, cfg['cluster_threshold'])
    wall = time.perf_counter()-started
    coordinates = coords[best].tolist()
    validation = validate_coordinates(cfg['seq'], coordinates, cfg['l_size'])
    if not validation['valid'] or validation['HH'] != best_hh:
        raise AssertionError(f'Worker {wid}: independent fold validation failed')
    result = {'worker_id':wid, 'seed':seed, 'best_HH':int(best_hh), 'best_energy':-int(best_hh),
        'best_excess':int(best_excess), 'best_positions':best.tolist(), 'coords':coordinates,
        'valid':True, 'target_hit':bool(best_hh >= cfg['target_HH']), 'stop_reason':reason,
        'wall_seconds':wall, 'preparation_seconds':prep_seconds, 'steps':total_steps,
        'first_recorded':first, 'statistics':dict(zip(
            ['proposals','valid_pools','candidates','distinct_signature_sum','transitions','expansions','geometry_prunes','duplicates','pool_capacity'],
            map(int,totals))), 'cycle_trace':cycle_rows, 'effort_trace':effort_rows}
    write_json(folder / f'worker_{wid:02d}_result.json', result)
    write_json(folder / f'worker_{wid:02d}_validation.json', validation)
    return result


def parse_config(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, help='Historical run_config.json, copied into the new run for provenance')
    ap.add_argument('--variant', choices=VARIANTS, default='full')
    ap.add_argument('--seq')
    ap.add_argument('--benchmark')
    ap.add_argument('--target-hh', type=int)
    ap.add_argument('--l-size', type=int)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--wall-seconds', type=float)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--max-proposals', type=int)
    ap.add_argument('--cycle-steps', help='Four comma-separated positive cycle lengths')
    ap.add_argument('--adaptive-window-steps', type=int)
    ap.add_argument('--beam-width', type=int)
    ap.add_argument('--max-adaptive-beam',type=int)
    ap.add_argument('--t0',type=float)
    ap.add_argument('--t1',type=float)
    ap.add_argument('--control-chunk-steps', type=int)
    stop = ap.add_mutually_exclusive_group()
    stop.add_argument('--target-stop', dest='target_stop', action='store_true')
    stop.add_argument('--no-target-stop', dest='target_stop', action='store_false')
    ap.set_defaults(target_stop=None)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_RUNS)
    ap.add_argument('--run-name', default='Q_ABHA_FCC')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    inherited = {}
    if args.profile:
        inherited = json.loads(args.profile.read_text(encoding='utf-8-sig'))
        cfg.update({k:v for k,v in inherited.items() if k in DEFAULTS or k in ('seq','benchmark','target_HH','l_size')})
        record = args.profile.parent / 'run_command.ps1'
        if record.is_file():
            sys.path.insert(0, str(PACKAGE_ROOT/'tools'))
            from replay_baseline import read_recorded_argv
            tokens = read_recorded_argv(record)[2:]
            aliases = {'t0':'temperature_t0','t1':'temperature_t1'}
            recorded = {}
            for key, default in DEFAULTS.items():
                cli_key = next((a for a,b in aliases.items() if b == key), key)
                flag = '--'+cli_key
                if flag in tokens and isinstance(default,(int,float)) and not isinstance(default,bool):
                    recorded[key] = type(default)(tokens[tokens.index(flag)+1])
            cfg.update(recorded)
            cfg['inherited_recorded_cli_parameters'] = recorded
        cfg['source_profile'] = str(args.profile.resolve())
    for key in ('seq','benchmark','l_size','workers','wall_seconds','seed','adaptive_window_steps','beam_width','control_chunk_steps','max_adaptive_beam'):
        value = getattr(args,key)
        if value is not None:
            cfg[key] = value
    if args.t0 is not None: cfg['temperature_t0']=args.t0
    if args.t1 is not None: cfg['temperature_t1']=args.t1
    if args.target_hh is not None:
        cfg['target_HH'] = args.target_hh
    if args.max_proposals is not None:
        cfg['max_proposals_per_worker'] = args.max_proposals
    if args.cycle_steps:
        cfg['cycle_steps'] = [int(x) for x in args.cycle_steps.split(',')]
    if args.target_stop is not None:
        cfg['target_stop_enabled'] = args.target_stop
    cfg['seq'] = cfg.get('seq','').strip().upper()
    cfg.setdefault('benchmark','direct')
    cfg['variant'] = args.variant
    cfg['algorithm'] = 'Q-ABHA-FCC'
    cfg['implementation_ancestor'] = 'RA-Q-RBHA-FCC-v2, validated 2026-09-15'
    cfg['bidirectional'] = args.variant not in ('optimized_baseline','fast_only','no_bidirectional')
    cfg['budget_schedule'] = args.variant not in ('optimized_baseline','fast_only','no_budget_schedule')
    cfg['length_effort'] = args.variant not in ('optimized_baseline','fast_only','no_length_effort')
    cfg['components'] = dict(zip(('routing','diversity','adaptive_effort'), VARIANTS[args.variant]))
    cfg['components'].update({k:cfg[k] for k in ('bidirectional','budget_schedule','length_effort')})
    cfg['schedule_definition'] = ('Proposal cycles; two stagnant cycles grow requested length. Budget quarters impose minimum '
        'objective stages 0/1/2/3; final quarter uses zero cluster shaping. Cycle length capped at boundaries using '
        'observed throughput, never timed candidate construction. Proposal-capped tests use proposal quarters instead.') if cfg['budget_schedule'] else 'Unchanged RA-v1 proposal-cycle stage controller'
    if len(cfg['seq']) < 4 or set(cfg['seq'])-{'H','P'}:
        ap.error('provide an HP sequence of at least four residues or a profile containing one')
    for key in ('l_size','target_HH','workers','control_chunk_steps','beam_width','adaptive_window_steps','cycles_without_improvement'):
        if cfg.get(key,0) <= 0:
            ap.error(f'{key} must be positive')
    if cfg['workers'] > 256 or cfg['beam_width'] > cfg['max_adaptive_beam']:
        ap.error('worker or beam limit exceeded')
    if cfg['seed'] < 0 or cfg['worker_seed_stride'] <= 0 or cfg['seed']+(cfg['workers']-1)*cfg['worker_seed_stride'] >= 2**32:
        ap.error('each worker seed must fit the NumPy/Numba uint32 seed range')
    for key in ('wall_seconds','temperature_t0','temperature_t1'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            ap.error(f'{key} must be finite and positive')
    if len(cfg['cycle_steps']) != 4 or min(cfg['cycle_steps']) <= 0:
        ap.error('cycle_steps requires four positive integers')
    if len(cfg['stage_block_max']) != 4 or min(cfg['stage_block_max']) < cfg['min_block_len']:
        ap.error('stage_block_max requires four lengths >= min_block_len')
    if cfg['min_block_len'] >= len(cfg['seq']) or cfg['min_block_len'] < 1:
        ap.error('min_block_len must be between 1 and N-1')
    if len(cfg['cluster_multipliers']) != 4 or any(not math.isfinite(x) for x in cfg['cluster_multipliers']):
        ap.error('cluster_multipliers requires four finite values')
    for key in ('pair_hh_reward','cluster_excess_reward','construction_floor','routing_weight'):
        if not math.isfinite(cfg[key]):ap.error(f'{key} must be finite')
    if cfg['construction_floor']<0 or cfg['routing_weight']<0:
        ap.error('construction_floor and routing_weight must be nonnegative')
    if cfg['max_adaptive_beam']<cfg['beam_width'] or cfg['max_adaptive_beam']>192:
        ap.error('max_adaptive_beam must be between base width and 192')
    if cfg['budget_schedule'] and cfg['cluster_multipliers'][-1]!=0:
        ap.error('budget-aware final phase requires zero cluster multiplier')
    if len(cfg['seq'])>(cfg['l_size']**3+1)//2:
        ap.error('chain exceeds the number of FCC sites')
    if cfg['temperature_t1'] > cfg['temperature_t0']:
        ap.error('cooling requires T1 <= T0')
    if cfg['near_target_beam_width'] < cfg['beam_width']:
        ap.error('near-target beam width must be >= base width')
    if cfg['max_proposals_per_worker'] is not None and cfg['max_proposals_per_worker'] <= 0:
        ap.error('max-proposals must be positive')
    for label in (cfg['benchmark'], args.run_name):
        if not label or label in ('.','..') or any(c in label for c in '/\\:'):
            ap.error('benchmark and run-name must be single safe directory names')
    return args,cfg,inherited


def run(args,cfg,inherited):
    folder = (args.out_dir / cfg['benchmark'] / args.run_name).resolve()
    if args.dry_run:
        print(json.dumps({'folder':str(folder),'config':cfg},indent=2))
        return
    folder.mkdir(parents=True,exist_ok=False)
    total_started = time.perf_counter()
    cfg['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    cfg['runtime_versions'] = {'python':platform.python_version(),'numpy':np.__version__,'numba':numba.__version__}
    cfg['timing_definition'] = 'Per-worker annealing wall time excludes initialization and JIT preparation; total_process_wall_seconds includes setup.'
    cfg['first_hit_definition'] = 'Exact proposal index; wall time first observed at the end of the enclosing kernel chunk.'
    write_json(folder / 'run_config.json',cfg)
    if inherited:
        write_json(folder / 'inherited_profile.json',inherited)
    write_json(folder / 'run_command.json',{'argv':[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]})
    def quote(value):
        return "'"+str(value).replace("'","''")+"'"
    (folder / 'run_command.ps1').write_text('& '+ ' '.join(map(quote,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]))+'\n',encoding='utf-8')
    source_files = list(Path(__file__).parent.glob('*.py')) + list(HISTORICAL_SOURCE.glob('*.py'))
    snapshot = folder / 'source_snapshot'
    snapshot.mkdir()
    source_hashes = {}
    import shutil
    for path in source_files:
        group = 'improved' if path.parent == Path(__file__).parent else 'historical_support'
        target = snapshot / group / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target)
        source_hashes[target.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(folder / 'source_sha256.json', source_hashes)
    model = folder / 'model_FCC_HP_cluster_worker_light.json'
    export_lightweight_model(cfg['seq'],model,cfg['l_size'],cfg['pair_hh_reward'],cfg['cluster_threshold'],cfg['cluster_excess_reward'],1000.)
    _,coords,_,sets,nbr_bool,hmask = model_context(model)
    starts, seen = [],set()
    for wid in range(cfg['workers']):
        rng = random.Random(cfg['seed']+wid*cfg['worker_seed_stride'])
        pos = pairwise_contact_saw(cfg['seq'],coords,sets,hmask,nbr_bool,rng,
            attempts=cfg['init_attempts'],h_pool=cfg['init_h_pool'],p_pool=cfg['init_p_pool'],random_pool_prob=cfg['init_random_pool_prob'])
        if pos is None:
            raise RuntimeError(f'Fresh SAW generation failed for worker {wid}')
        if tuple(pos) in seen:
            retry = pairwise_contact_saw(cfg['seq'],coords,sets,hmask,nbr_bool,rng,
                attempts=cfg['init_attempts'],h_pool=max(3,cfg['init_h_pool']+1),p_pool=max(5,cfg['init_p_pool']+1),
                random_pool_prob=min(.35,cfg['init_random_pool_prob']+.12))
            if retry is not None:
                pos = retry
        seen.add(tuple(pos))
        check = validate_coordinates(cfg['seq'],[coords[i] for i in pos],cfg['l_size'])
        if not check['valid']:
            raise AssertionError('Invalid initializer output')
        starts.append(list(map(int,pos)))
    del nbr_bool
    write_json(folder / 'initial_saw_pool.json',{'seq':cfg['seq'],'L':cfg['l_size'],
        'base_seed':cfg['seed'],'worker_starts':[[p] for p in starts]})
    warm_kernel()
    context = mp.get_context('spawn')
    results=[]
    with context.Manager() as manager:
        event = manager.Event()
        abort_event = manager.Event()
        tasks=[{'config':cfg,'worker_id':i,'folder':str(folder),'coords':coords,'initial':p,'stop_event':event,'abort_event':abort_event}
               for i,p in enumerate(starts)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=context) as pool:
            jobs=[pool.submit(worker,t) for t in tasks]
            try:
                for future in concurrent.futures.as_completed(jobs):
                    result=future.result()
                    results.append(result)
                    print(f"worker {result['worker_id']:02d}: HH={result['best_HH']} wall={result['wall_seconds']:.3f}s stop={result['stop_reason']}",flush=True)
            except BaseException as exc:
                abort_event.set()
                write_json(folder/'run_failure.json',{'error_type':type(exc).__name__,'error':str(exc),
                    'completed_workers':len(results),'status':'failed; partial results are not a complete experiment'})
                for job in jobs:job.cancel()
                raise
    results.sort(key=lambda x:x['worker_id'])
    best=max(results,key=lambda x:(x['best_HH'],x['best_excess']))
    contacts=[r['best_HH'] for r in results]
    def first_observation(threshold):
        times=[v['recorded_wall_seconds'] for r in results for c,v in r['first_recorded'].items()
               if int(c)>=threshold]
        return min(times) if times else None
    summary={'algorithm':cfg['algorithm'],'variant':cfg['variant'],'benchmark':cfg['benchmark'],'N':len(cfg['seq']),
        'L':cfg['l_size'],'target_HH':cfg['target_HH'],'best_HH':max(contacts),'best_energy':-max(contacts),
        'mean_worker_best_HH':statistics.mean(contacts),
        'sample_SD_worker_best_HH':statistics.stdev(contacts) if len(contacts)>1 else None,
        'workers_recorded':len(results),'worker_target_hit_rate_pct':100*sum(r['target_hit'] for r in results)/len(results),
        'mean_worker_wall_seconds':statistics.mean(r['wall_seconds'] for r in results),
        'earliest_best_observed_wall_seconds':first_observation(max(contacts)),
        'earliest_target_observed_wall_seconds':first_observation(cfg['target_HH']),
        'mean_valid_pools_per_second':statistics.mean(r['statistics']['valid_pools']/max(r['wall_seconds'],1.e-12) for r in results),
        'mean_candidates_per_second':statistics.mean(r['statistics']['candidates']/max(r['wall_seconds'],1.e-12) for r in results),
        'total_process_wall_seconds':time.perf_counter()-total_started,
        'statistical_unit':'independent worker-best fold; not repeated 16-worker batch-best',
        'full_annealing_budgets_completed':all(r['stop_reason']=='wall_budget' for r in results),
        'run_dir':str(folder),'valid':all(r['valid'] for r in results)}
    write_json(folder/'worker_results.json',results)
    write_json(folder/'best.json',dict(best,seq=cfg['seq'],L=cfg['l_size'],HH=best['best_HH']))
    write_json(folder/'validation_report.json',{'all_valid':True,'validated_workers':len(results),'method':'independent coordinate-distance contact count'})
    write_json(folder/'q_rbha_result.json',summary)
    for name,rows in [('cycle_trace',[x for r in results for x in r['cycle_trace']]),
                      ('effort_trace',[dict(x,worker_id=r['worker_id']) for r in results for x in r['effort_trace']])]:
        write_json(folder/f'{name}.json',rows)
        if rows:
            with (folder/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (folder/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary));w.writeheader();w.writerow(summary)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    mp.freeze_support()
    args,cfg,inherited=parse_config()
    run(args,cfg,inherited)


if __name__=='__main__':
    main()
