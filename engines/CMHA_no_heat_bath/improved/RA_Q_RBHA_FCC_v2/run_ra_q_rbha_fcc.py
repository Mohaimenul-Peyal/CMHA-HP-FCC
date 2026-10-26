#!/usr/bin/env python3
"""PGR-J4-Q-ABHA-FCC: sequence-derived paired selection on the J4 baseline."""
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
    from .p_run_windows import build_catalog, describe_catalog
    from .vacancy_focus import vacated_mask, pulse_remaining, counts as focus_counts
    from .polar_refresh import polar_intervals, perform_refresh, warm_refresh, due as refresh_due
    from .route_diversity import RouteDiversityArchive
    from .diversity_restart import DiversityArchive, contact_map
    from .controller import EffortController, budget_stage, capped_cycle
    from .resource_control import ResourceEffortController, terminal_cycle_plan, terminal_temperature
    from .support import (PACKAGE_ROOT, DEFAULT_RUNS, HISTORICAL_SOURCE, export_lightweight_model,
                          fcc_lattice_coords, fcc_neighbor_sets, model_context,
                          pairwise_contact_saw, validate_coordinates)
else:
    # The CLI is normally imported as a package by run_q_abha_fcc.py.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from improved.RA_Q_RBHA_FCC_v2.p_run_windows import build_catalog, describe_catalog
    from improved.RA_Q_RBHA_FCC_v2.vacancy_focus import vacated_mask, pulse_remaining, counts as focus_counts
    from improved.RA_Q_RBHA_FCC_v2.polar_refresh import polar_intervals, perform_refresh, warm_refresh, due as refresh_due
    from improved.RA_Q_RBHA_FCC_v2.route_diversity import RouteDiversityArchive
    from diversity_restart import DiversityArchive, contact_map
    from controller import EffortController, budget_stage, capped_cycle
    from resource_control import ResourceEffortController, terminal_cycle_plan, terminal_temperature
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
                control_chunk_steps=8, target_stop_enabled=False, target_stop_scope='any_worker',
                routing_weight=0.15, construction_floor=0.4, adaptive_window_steps=8000,
                max_adaptive_beam=96, max_proposals_per_worker=None,
                joint_probability=0.25, joint_beam_width=24, joint_parents=1,
                joint_expansion_budget=12000, late_cycle_cap_fraction=0.25,
                paired_beam_selection='baseline', boundary_policy='constrained_first',
                unguided_probability=0.0, effort_policy='resource', cycle_policy='terminal',
                guided_coupled_probability=0.5, guided_anchor_attempts=4,
                diversity_restart=True, archive_capacity=16, archive_max_gap=2,
                archive_min_distance=0.02, archive_sample_steps=4096,
                archive_min_stage=3, route_diversity=False,
                archive_route_weight=0.25, archive_routes_per_map=2,
                polar_refresh=False, polar_refresh_attempts=128,
                polar_refresh_expansion_cap=12000, polar_refresh_patience_seconds=1200.,
                polar_refresh_cooldown_seconds=600., vacancy_responsive=False,
                vacancy_focus_steps=256000, vacancy_boost=3.0, triple_probability=0.0,
                quadruple_probability=0.0, p_run_probability=0.0,
                contact_condition_probability=0.0)

CONDITION_FIELDS = ('proposals','valid_pools','completed_candidates','expansions',
    'forced_reachability_prunes','target_closing_candidates','target_closed_pools',
    'transitions','selected_target_closures','physical_best_improvements','contacts_gained',
    'generated_gain_over_best','generated_gain_over_current','selected_net_contact_gain',
    'construction_seconds','best_generated_HH')


def condition_gate(stage, stagnant_cycles):
    return stage >= 3 and stagnant_cycles >= 2


def condition_rows(values, wid):
    rows = []
    for arm, name in enumerate(('ordinary_eligible_pair','conditioned_pair')):
        row = dict(worker_id=wid, constructor=name, **dict(zip(CONDITION_FIELDS,values[arm].tolist())))
        seconds = row['construction_seconds']
        row.update(best_improvements_per_construction_second=(row['physical_best_improvements']/seconds if seconds else None),
                   contacts_gained_per_construction_second=(row['contacts_gained']/seconds if seconds else None),
                   best_improvements_per_proposal=(row['physical_best_improvements']/row['proposals'] if row['proposals'] else None),
                   timing_definition='Every eligible construction timed, including timing bridge overhead; scoring/selection excluded')
        rows.append(row)
    return rows

GUIDANCE_MODES = ('single', 'random_coupled', 'guided_coupled', 'guided_fallback',
                  'blocking_residue_triple', 'triple_no_eligible_intervals',
                  'routing_blocker_quadruple', 'quadruple_no_eligible_intervals',
                  'p_run_guided_pair', 'p_run_fallback_inherited')
PROPOSAL_TYPES = ('single','coupled','triple','quadruple')
GUIDANCE_FIELDS = ('proposals', 'valid_pools', 'expansions', 'pools_beating_worker_best',
                   'physical_best_improvements', 'contacts_gained')


def guidance_rows(values, wid):
    return [dict(worker_id=wid, selection_mode=name,
                 **dict(zip(GUIDANCE_FIELDS, map(int, values[i]))))
            for i, name in enumerate(GUIDANCE_MODES[:len(values)])]


def detail_record(row):
    return dict(block_intervals_zero_based_half_open=[[int(row[0]), int(row[1])]] +
                ([[int(row[2]), int(row[3])]] if row[6] else []) +
                ([[int(row[11]),int(row[12])]] if len(row)>11 and row[6] in (4,6) else []) +
                ([[int(row[14]),int(row[15])]] if len(row)>15 and row[6]==6 else []),
                blocking_residue_index=int(row[13]) if len(row)>13 and row[13]>=0 else None,
                beam_width=int(row[4]), temperature=float(row[5]),
                interval_selection=GUIDANCE_MODES[int(row[6])],
                selected_missing_H_pair=[int(row[7]), int(row[8])] if row[7] >= 0 else None,
                contacts_created_vs_previous_best=int(row[9]) if row[9] >= 0 else None,
                contacts_broken_vs_previous_best=int(row[10]) if row[10] >= 0 else None,
                second_blocking_residue_index=int(row[16]) if len(row)>16 and row[16]>=0 else None,
                contact_conditioned=bool(row[17]>0) if len(row)>17 else False,
                contact_condition_eligible=bool(row[18]>0) if len(row)>18 else False)

SELECTION_FIELDS = ['construction_depths', 'retained_gap0', 'retained_gap1',
                    'retained_gap2', 'retained_gap3plus', 'available_gap1', 'available_gap2']

BLOCK_FIELDS = ['proposals', 'valid_pools', 'candidates', 'distinct_signature_sum',
                'transitions', 'expansions', 'geometry_prunes', 'duplicates', 'pool_capacity',
                'physical_best_improvements', 'contacts_gained', 'timing_samples',
                'sampled_construction_seconds']

PROPOSAL_FIELDS = ['proposals', 'valid_pools', 'distinct_candidates', 'transitions',
    'both_intervals_changed', 'pools_beating_previous_worker_best', 'physical_best_improvements',
    'contacts_gained', 'expansions', 'geometry_prunes', 'best_generated_HH', 'no_eligible_pair']


def proposal_rows(values, worker_id):
    return [dict(worker_id=worker_id, proposal_type=name,
                 **{('all_intervals_changed' if key=='both_intervals_changed' else key): int(v)
                    for key, v in zip(PROPOSAL_FIELDS, values[k])})
            for k, name in enumerate(PROPOSAL_TYPES[:len(values)])]


def window_rows(values, old, timings, old_timings, wid, start, end):
    delta = values-old
    delta[:, PROPOSAL_FIELDS.index('best_generated_HH')] = values[:, PROPOSAL_FIELDS.index('best_generated_HH')]
    rows = proposal_rows(delta, wid)
    for k, row in enumerate(rows):
        samples, seconds = timings[k]-old_timings[k]
        row.update(wall_start=start, wall_end=end, duration_seconds=end-start,
                   timing_samples=int(samples), sampled_construction_seconds=float(seconds),
                   mean_sampled_construction_seconds=float(seconds/samples) if samples else None,
                   generated_HH_definition='cumulative maximum at window end, not a difference')
    return rows


def block_rows(values, worker_id):
    rows = []
    for length in range(len(values)):
        if not values[length, 0]:
            continue
        row = {'worker_id': worker_id, 'block_length': length}
        row.update({key: float(value) if key == 'sampled_construction_seconds' else int(value)
                    for key, value in zip(BLOCK_FIELDS, values[length])})
        row['mean_sampled_construction_seconds'] = (row['sampled_construction_seconds'] / row['timing_samples']
                                                   if row['timing_samples'] else None)
        rows.append(row)
    return rows


def resource_rows(values, worker_id):
    fields = ['proposals','valid_pools','expansions','generated_gain_over_current',
              'generated_gain_over_worker_best','accepted_best_contacts_gained']
    rows = []
    for kind in range(len(values)):
        for arm,width in enumerate((24,48,96)):
            for policy in range(2):
                if values[kind,arm,policy,0]:
                    rows.append(dict(worker_id=worker_id,
                        proposal_type=PROPOSAL_TYPES[kind], beam_width=width,
                        partial_preference=('guided','unguided')[policy],
                        **dict(zip(fields,map(int,values[kind,arm,policy])))))
    return rows


def interval_rows(values, worker_id):
    rows = []
    for a,b in np.argwhere(values[:,:,0] > 0):
        rows.append(dict(worker_id=worker_id, proposal_type='coupled' if b else 'single',
            first_length=int(a), second_length=int(b),
            **dict(zip(('proposals','valid_pools','contacts_gained','expansions'),
                       map(int,values[a,b])))))
    return rows


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacement of an open file.
    # Retry only access/sharing errors; preserve the old file and pending JSON
    # if a permanent failure remains after the bounded retry window.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc,'winerror',None) in (5,32,33)
            if not transient or attempt==9:
                raise
            time.sleep(min(.25,.01*2**attempt))


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
                 2, -2., -.5, 2, 3, 4, 8., .02, 4, 0, 1, 100, True, True, .4, .15, True, 4,
                 np.zeros((5, len(BLOCK_FIELDS)), np.float64), 0.25, 24, 1, 12000,
                 np.zeros((4, len(PROPOSAL_FIELDS)), np.int64), np.full((1, 4), -1, np.int64),
                 np.zeros((4,2), np.float64), True, np.zeros(7, np.int64),
                 1,.25,np.zeros((3,5),np.int64),np.zeros((4,3,2,6),np.int64),
                 np.zeros((25,25,4),np.int64),-1., .5, 4,
                 np.zeros((8,6),np.int64),np.full((1,17),-1.,np.float64),
                 np.zeros(len(coords),np.bool_),3.,np.zeros(8,np.int64),0.1,0.05)
    k.contact_map_fingerprint(pos, h, neighbors)


def warm_p_run_kernel(coords, neighbors, hmask, pos, cfg):
    k = load_kernel()
    _, tables = build_catalog(hmask, cfg['stage_block_max'])
    windows, offsets, members = tables[min(12, max(cfg['stage_block_max'])//2)]
    hh, ex, deg = k.full_metrics(pos, hmask, neighbors, cfg['cluster_threshold'])
    k.anneal_chunk(pos.copy(), pos.copy(), hh, ex, deg, hh, ex, hmask, coords, neighbors,
        cfg['cluster_threshold'], cfg['pair_hh_reward'], cfg['cluster_excess_reward'],
        cfg['min_block_len'], 24, cfg['beam_width'], cfg['temperature_t0'], cfg['temperature_t1'],
        8, 0, 8, np.iinfo(np.int32).max, True, True, cfg['construction_floor'],
        cfg['routing_weight'], True, cfg['beam_width'],
        np.zeros((len(pos)+1, len(BLOCK_FIELDS)), np.float64),
        cfg['joint_probability'], cfg['joint_beam_width'], cfg['joint_parents'],
        cfg['joint_expansion_budget'], np.zeros((4,12), np.int64), np.full((8,4),-1,np.int64),
        np.zeros((4,2), np.float64), False, np.zeros(7,np.int64), 1, 0.,
        np.zeros((3,5),np.int64), np.zeros((4,3,2,6),np.int64),
        np.zeros((25,25,4),np.int64), -1., cfg['guided_coupled_probability'],
        cfg['guided_anchor_attempts'], np.zeros((10,6),np.int64), np.full((8,17),-1.,np.float64),
        np.zeros(len(coords),np.bool_), 0., np.zeros(8,np.int64),
        cfg['triple_probability'],cfg['quadruple_probability'],cfg['p_run_probability'],
        windows,offsets,members,cfg['contact_condition_probability'],True,
        np.zeros((2,len(CONDITION_FIELDS)),np.float64),
        int(cfg.get('transition_rule', 'direct_heat_bath') == 'uniform_metropolis'),
        np.zeros(8,np.int64))


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
    archive = (DiversityArchive(seed, cfg['archive_capacity'], cfg['archive_max_gap'],
               cfg['archive_min_distance']) if cfg['diversity_restart'] else None)
    if archive is not None and cfg.get('route_diversity',False):
        archive=RouteDiversityArchive(seed,hmask,neighbors,cfg['archive_capacity'],
            cfg['archive_max_gap'],cfg['archive_min_distance'],cfg['archive_route_weight'],
            cfg['archive_routes_per_map'])
    restart_rows = []
    next_archive = cfg['archive_sample_steps']
    if archive is not None:
        archive.observe(best, contact_map(best,hmask,neighbors), best_hh, 0, 0.)
    warm_kernel()
    p_run_tables = None
    if cfg['p_run_probability']:
        _, p_run_tables = build_catalog(hmask, cfg['stage_block_max'])
        warm_p_run_kernel(coords, neighbors, hmask, current, cfg)
    intervals = polar_intervals(hmask)
    if cfg['polar_refresh']:
        warm_refresh(coords,neighbors,hmask,current)
    seed_kernel(seed)
    prep_seconds = time.perf_counter()-prep
    routing, diversity, adaptive = VARIANTS[cfg['variant']]
    ctl = EffortController(width=cfg['beam_width'], base=cfg['beam_width'],
                           maximum=cfg['max_adaptive_beam'], window_steps=cfg['adaptive_window_steps'],
                           next_check=cfg['adaptive_window_steps'])
    resource = cfg['effort_policy'] == 'resource'
    if resource:
        ctl = ResourceEffortController(base=cfg['beam_width'],width=cfg['beam_width'],
            maximum=cfg['max_adaptive_beam'],window_steps=cfg['adaptive_window_steps'],
            next_check=cfg['adaptive_window_steps'])
    effort_stats = np.zeros((3,5),np.int64)
    proposal_count = 4 if cfg.get('quadruple_probability',0.0) else 3
    decision_stats = np.zeros((proposal_count,3,2,6),np.int64)
    interval_stats = np.zeros((max(cfg['stage_block_max'])+1,)*2+(4,),np.int64)
    totals, window = np.zeros(9, np.int64), np.zeros(9, np.int64)
    transition_stats = np.zeros(8, np.int64)
    transition_rule = int(cfg.get('transition_rule', 'direct_heat_bath') == 'uniform_metropolis')
    by_length = np.zeros((len(current)+1, len(BLOCK_FIELDS)), np.float64)
    by_proposal = np.zeros((proposal_count, len(PROPOSAL_FIELDS)), np.int64)
    construction_timings = np.zeros((proposal_count,2), np.float64)
    selection_stats = np.zeros(7, np.int64)
    guidance_stats = np.zeros((10 if cfg['p_run_probability'] else
                               (8 if proposal_count==4 else 6),6), np.int64)
    contact_condition_stats = np.zeros((2,len(CONDITION_FIELDS)),np.float64)
    contact_stagnant_cycles = 0
    condition_windows = []
    prior_condition = contact_condition_stats.copy()
    novelty_seen, novelty_rows, novelty_samples = set(), [], 0
    next_novelty = 4096
    prior_proposals, prior_timings = by_proposal.copy(), construction_timings.copy()
    proposal_windows = []
    window_start, next_window = 0., min(3600., cfg['wall_seconds']/3)
    window_seconds = next_window
    generated_best = best_hh
    generated_first = {str(best_hh): {'proposal': 0, 'recorded_wall_seconds': 0.0,
                                     'proposal_type': 'initial_SAW'}}
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
    refresh_rows=[]
    focus_mask=np.zeros(len(coords),np.bool_)
    focus_end=focus_steps=0
    focus_stats=np.zeros(8,np.int64)
    focus_rows=[]
    last_improvement=last_refresh=0.
    next_checkpoint=3600.

    def apply_refresh(trigger):
        nonlocal hh,excess,degrees,last_refresh,focus_mask,focus_end
        if refresh_rows and refresh_rows[-1]['best_HH_at_next_refresh'] is None:
            refresh_rows[-1]['best_HH_at_next_refresh']=int(best_hh)
        hh,excess,degrees,record=perform_refresh(current,hmask,coords,neighbors,intervals,cfg,
                                               started+cfg['wall_seconds'])
        last_refresh=time.perf_counter()-started
        if cfg['vacancy_responsive']:
            focus_mask=vacated_mask(record['positions_before'],current,len(coords))
            focus_end=total_steps+cfg['vacancy_focus_steps'] if np.any(focus_mask) else total_steps
            occ=k.occupancy(current,len(coords))
            exposed=[i for i in range(len(current)) if hmask[i] and
                     k.vacancy_exposure(current[i],occ,neighbors,focus_mask)>0]
            focus_rows.append(dict(refresh_index=len(refresh_rows),proposal=total_steps,
                end_proposal=focus_end,wall_seconds=last_refresh,
                vacated_sites=np.flatnonzero(focus_mask).tolist(),exposed_H_indices=exposed,
                best_HH_at_activation=int(best_hh)))
            write_json(folder/f'worker_{wid:02d}_vacancy_focus.json',focus_rows)
        refresh_rows.append(dict(worker_id=wid,seed=seed,trigger=trigger,stage=stage,cycle=cycle,
            proposal=total_steps,wall_seconds=last_refresh,best_HH_at_refresh=int(best_hh),
            best_HH_at_next_refresh=None,**record))
        write_json(folder/f'worker_{wid:02d}_polar_refresh.json',refresh_rows)
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
                if restart_rows and restart_rows[-1]['end_wall_seconds'] is None:
                    restart_rows[-1].update(end_wall_seconds=elapsed,best_HH_at_end=int(best_hh),
                        improvement_after_restart=int(best_hh)-restart_rows[-1]['best_HH_before'])
                stage,stagnant = required,0
                current = best.copy()
                hh,excess,degrees = full_metrics(current,hmask,neighbors,cfg['cluster_threshold'])
        requested_length = (cfg['cycle_steps'][stage] if stage < len(cfg['cycle_steps']) else
                  cfg['cycle_steps'][-1] * 2**(stage-len(cfg['cycle_steps'])+1))
        length = capped_cycle(requested_length,stage,elapsed,cfg['wall_seconds'],total_steps,
                              cfg['max_proposals_per_worker'],rate,
                              cfg['late_cycle_cap_fraction']) if cfg['budget_schedule'] else requested_length
        terminal = False
        if cfg['cycle_policy']=='terminal':
            length,terminal = terminal_cycle_plan(requested_length,stage,elapsed,
                cfg['wall_seconds'],total_steps,cfg['max_proposals_per_worker'],rate,
                cfg['late_cycle_cap_fraction'],length)
        terminal_start_steps = total_steps
        if resource:
            ctl.new_cycle()
        cap_elapsed = elapsed
        cluster = cfg['cluster_excess_reward'] * cfg['cluster_multipliers'][min(stage,3)]
        max_block = min(len(current)-1, cfg['stage_block_max'][min(stage,3)])
        p_windows, p_offsets, p_members = (p_run_tables[min(12,max_block//2)]
                                           if p_run_tables is not None else (None,None,None))
        start_hh, offset = best_hh, 0
        cycle_started = time.perf_counter()-started
        budget_interrupted = False
        while offset < length or terminal:
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
            if cfg['polar_refresh'] and refresh_due(elapsed,last_improvement,last_refresh,
                    cfg['archive_min_stage'],stage,cfg['polar_refresh_patience_seconds'],
                    cfg['polar_refresh_cooldown_seconds']):
                # Stay on the existing cooling curve. Restore best only if current
                # is outside the permitted near-best contact gap.
                if hh < best_hh-cfg['archive_max_gap']:
                    current=best.copy()
                apply_refresh('wall_stagnation')
                elapsed=time.perf_counter()-started
                if elapsed>=cfg['wall_seconds']:
                    done,reason=True,'wall_budget'
                    break
            if not terminal and cfg['budget_schedule'] and budget_stage(elapsed,cfg['wall_seconds'],total_steps,remaining)>stage:
                budget_interrupted = True
                break
            chunk = cfg['control_chunk_steps'] if terminal else min(cfg['control_chunk_steps'], length-offset)
            focus_active=cfg['vacancy_responsive'] and pulse_remaining(total_steps,focus_end)>0
            if focus_active:
                chunk=min(chunk,pulse_remaining(total_steps,focus_end))
            if remaining is not None:
                chunk = min(chunk, remaining-total_steps)
                if not terminal and cfg['budget_schedule'] and stage<3:
                    chunk = min(chunk,max(1,math.ceil((stage+1)*remaining/4)-total_steps))
            width = ctl.width if adaptive else (
                cfg['near_target_beam_width'] if cfg['target_HH'] is not None and
                best_hh >= cfg['target_HH']-cfg['near_target_gap'] else cfg['beam_width'])
            previous_best = best_hh
            terminal_t = terminal_temperature(cfg['temperature_t0'],cfg['temperature_t1'],
                elapsed,cycle_started,cfg['wall_seconds'],total_steps,terminal_start_steps,
                cfg['max_proposals_per_worker']) if terminal else -1.
            candidate_records = np.full((chunk, 4), -1, np.int64)
            proposal_details = np.full((chunk, 19), -1., np.float64)
            hh, excess, degrees, best_hh, best_excess, stats, milestones = anneal_chunk(
                current, best, hh, excess, degrees, best_hh, best_excess, hmask, coords, neighbors,
                cfg['cluster_threshold'], cfg['pair_hh_reward'], cluster, cfg['min_block_len'],
                max_block, width, cfg['temperature_t0'], cfg['temperature_t1'], length, offset,
                chunk, stop_hh, routing, diversity, cfg['construction_floor'], cfg['routing_weight'],
                cfg['bidirectional'],cfg['beam_width'] if cfg['length_effort'] else 0, by_length,
                cfg['joint_probability'], cfg['joint_beam_width'], cfg['joint_parents'],
                cfg['joint_expansion_budget'], by_proposal, candidate_records, construction_timings,
                cfg['paired_beam_selection']=='contact_stratified', selection_stats,
                int(cfg['boundary_policy']=='constrained_first'),cfg['unguided_probability'],
                effort_stats,decision_stats,interval_stats,terminal_t,
                cfg['guided_coupled_probability'],cfg['guided_anchor_attempts'],
                guidance_stats,proposal_details,focus_mask,
                cfg['vacancy_boost'] if focus_active else 0.,focus_stats,cfg['triple_probability'],
                cfg.get('quadruple_probability',0.0),cfg['p_run_probability'],
                p_windows,p_offsets,p_members,cfg['contact_condition_probability'],
                condition_gate(stage,contact_stagnant_cycles),contact_condition_stats,
                transition_rule,transition_stats)
            if focus_active:
                focus_steps+=int(stats[0])
            elapsed = time.perf_counter()-started
            if best_hh>previous_best:
                last_improvement=elapsed
                contact_stagnant_cycles = 0
            if elapsed >= next_window:
                delta_condition = contact_condition_stats-prior_condition
                delta_condition[:,15] = contact_condition_stats[:,15]
                condition_windows.extend(dict(wall_start=window_start,wall_end=elapsed,
                    generated_HH_definition='cumulative maximum at window end',**r)
                    for r in condition_rows(delta_condition,wid))
                prior_condition = contact_condition_stats.copy()
                proposal_windows.extend(window_rows(by_proposal, prior_proposals,
                    construction_timings, prior_timings, wid, window_start, elapsed))
                prior_proposals, prior_timings = by_proposal.copy(), construction_timings.copy()
                window_start, next_window = elapsed, next_window+window_seconds
            if np.max(candidate_records[:, 1]) > generated_best:
                for index, (kind, pool_hh, selected_hh, previous_hh) in enumerate(candidate_records):
                    if pool_hh > generated_best:
                        generated_best = int(pool_hh)
                        generated_first[str(generated_best)] = dict(proposal=total_steps+index+1,
                            vacancy_focus_active=focus_active,
                            recorded_wall_seconds=elapsed, stage=stage, cycle=cycle,
                            proposal_type=PROPOSAL_TYPES[kind], selected_HH=int(selected_hh),
                            previous_worker_best_HH=int(previous_hh),
                            **detail_record(proposal_details[index]))
            for proposal, contacts in milestones:
                first.setdefault(str(contacts), {'proposal': total_steps+int(proposal),
                    'vacancy_focus_active':focus_active,
                    'recorded_wall_seconds': elapsed, 'stage': stage, 'cycle': cycle,
                    'proposal_type': PROPOSAL_TYPES[candidate_records[int(proposal)-1,0]],
                    **detail_record(proposal_details[int(proposal)-1])})
            total_steps += int(stats[0])
            if cfg['polar_refresh'] and elapsed>=next_checkpoint:
                check=validate_coordinates(cfg['seq'],coords[best].tolist(),cfg['l_size'])
                assert check['valid'] and check['HH']==best_hh
                write_json(folder/f'worker_{wid:02d}_checkpoint_{int(next_checkpoint):05d}s.json',
                    dict(worker_id=wid,seed=seed,elapsed_seconds=elapsed,best_HH=int(best_hh),
                        current_HH=int(hh),coords=coords[best].tolist(),steps=total_steps,
                        first_recorded=first,refresh_count=len(refresh_rows),valid=True))
                next_checkpoint+=3600.
            if archive is not None:
                if best_hh > previous_best:
                    archive.observe(best, contact_map(best,hmask,neighbors), best_hh,
                                    total_steps, elapsed)
                if total_steps >= next_archive:
                    next_archive = (total_steps//cfg['archive_sample_steps']+1)*cfg['archive_sample_steps']
                    if hh >= best_hh-cfg['archive_max_gap']:
                        contacts = contact_map(current,hmask,neighbors)
                        if len(contacts) != hh:
                            raise AssertionError('Archive/contact-score mismatch')
                        archive.observe(current,contacts,best_hh,total_steps,elapsed)
            if total_steps >= next_novelty:
                next_novelty = (total_steps//4096+1)*4096
                if hh >= best_hh-2:
                    novelty_samples += 1
                    fingerprint = int(k.contact_map_fingerprint(current,hmask,neighbors))
                    if fingerprint not in novelty_seen:
                        novelty_seen.add(fingerprint)
                        novelty_rows.append(dict(proposal=total_steps,wall_seconds=elapsed,
                            HH=int(hh),worker_best_HH=int(best_hh),stage=stage,cycle=cycle,
                            contact_map_fingerprint=f'{fingerprint:016x}'))
            offset += int(stats[0])
            totals += stats
            window += stats
            window_capacity += int(stats[8])
            if adaptive:
                due = total_steps >= ctl.next_check or best_hh > previous_best
                old_width = ctl.width
                if resource:
                    due = total_steps >= ctl.next_check
                    why = ctl.update(total_steps,effort_stats)
                else:
                    why = ctl.update(total_steps, best_hh > previous_best,
                        float(window[1])/max(1,int(window[0])), float(window[3])/max(1,int(window[2])),
                        float(window[2])/max(1,window_capacity))
                if due:
                    effort_rows.append({'proposal': total_steps, 'wall_seconds': elapsed,
                        'best_HH': int(best_hh), 'old_beam': old_width, 'new_beam': ctl.width,
                        'reason': why, 'valid_pool_fraction': float(window[1])/max(1,int(window[0])),
                        'signature_diversity': float(window[3])/max(1,int(window[2])),
                        'stage':stage,'cycle':cycle,
                        'long_single_effort_evidence':effort_stats.tolist() if resource else None})
                    window[:] = 0
                    window_capacity = 0
            if elapsed-last_progress >= 5.0:
                write_json(folder / f'worker_{wid:02d}_progress.json', {
                    'worker_id': wid, 'seed': seed, 'best_HH': int(best_hh), 'current_HH': int(hh),
                    'elapsed_seconds': elapsed, 'stage': stage, 'cycle': cycle,
                    'cycle_steps': length, 'cycle_offset': offset, 'steps': total_steps,
                    'beam_width': width, 'best_positions': best.tolist(), 'first_recorded': first,
                    'terminal_cycle':terminal,
                    'transition_rule':cfg.get('transition_rule', 'direct_heat_bath'),
                    'transition_statistics':dict(zip(
                        ('scored_pools','uniform_draws','nonworse_accepted','uphill_proposed',
                         'uphill_accepted','uphill_rejected','empty_distinct_pools','empty_generated_pools'),
                        map(int,transition_stats))),
                    'completed_contact_stagnant_cycles':contact_stagnant_cycles,
                    'contact_condition_gate':condition_gate(stage,contact_stagnant_cycles),
                    'contact_condition_statistics':condition_rows(contact_condition_stats,wid),
                    'contact_condition_windows':condition_windows,
                    'polar_refresh_count':len(refresh_rows),
                    'last_polar_refresh_wall_seconds':last_refresh,
                    'interval_guidance_statistics':guidance_rows(guidance_stats,wid),
                    'vacancy_focus_steps':focus_steps,'vacancy_focus_statistics':focus_counts(focus_stats),
                    'near_best_contact_map_samples':novelty_samples,
                    'near_best_distinct_contact_map_fingerprints':len(novelty_seen),
                    'resource_statistics':resource_rows(decision_stats,wid),
                    'proposal_statistics': proposal_rows(by_proposal, wid),
                    'proposal_windows': proposal_windows,
                    'paired_beam_selection_statistics': dict(zip(SELECTION_FIELDS,map(int,selection_stats))),
                    'best_generated_HH': generated_best, 'first_generated': generated_first})
                if archive is not None:
                    write_json(folder / f'worker_{wid:02d}_restart_archive.json',
                        dict(worker_id=wid,seed=seed,**archive.snapshot()))
                    write_json(folder / f'worker_{wid:02d}_restarts.json',restart_rows)
                write_json(folder / f'worker_{wid:02d}_block_statistics.json', block_rows(by_length, wid))
                last_progress = elapsed
            if cfg['target_stop_enabled'] and best_hh >= cfg['target_HH']:
                done, reason = True, 'target_hit'
                if event is not None:
                    event.set()
                break
        improved = best_hh > start_hh
        complete = offset == length and not terminal and not budget_interrupted
        if complete:
            contact_stagnant_cycles = 0 if improved else contact_stagnant_cycles+1
        cycle_wall = time.perf_counter()-started-cycle_started
        cycle_rows.append({'worker_id':wid, 'stage':stage, 'cycle':cycle, 'cycle_steps':length,
            'terminal_cycle':terminal,'terminal_temperature_clock':('proposals' if cfg['max_proposals_per_worker'] else 'wall') if terminal else None,
            'requested_cycle_steps':requested_length,'budget_stage_interrupted':budget_interrupted,
            'cycle_cap_elapsed_seconds':cap_elapsed, 'cycle_start_proposal':total_steps-offset,
            'cycle_cap_fraction':cfg['late_cycle_cap_fraction'] if stage >= 3 else 0.125,
            'nominal_cycle_cap_seconds':cfg['wall_seconds']*(cfg['late_cycle_cap_fraction'] if stage >= 3 else 0.125),
            'rate_used_for_cycle_cap':rate,'measured_proposals_per_second':offset/max(cycle_wall,1.e-12),
            'steps_completed':offset, 'complete':complete, 'start_HH':int(start_hh), 'best_HH':int(best_hh),
            'wall_start':cycle_started, 'wall_end':time.perf_counter()-started,
            'cluster_multiplier':cfg['cluster_multipliers'][min(stage,3)], 'max_block_len':max_block,
            'completed_contact_stagnant_cycles':contact_stagnant_cycles})
        cycle += 1
        if offset and cycle_wall>0:
            measured = offset/cycle_wall
            rate = measured if rate is None else .5*rate+.5*measured
        if complete:
            stagnant = 0 if improved else stagnant+1
            if stagnant >= cfg['cycles_without_improvement']:
                stage += 1
                stagnant = 0
                restart = None
                if archive is not None and stage >= cfg['archive_min_stage'] and not done:
                    if restart_rows and restart_rows[-1]['end_wall_seconds'] is None:
                        restart_rows[-1].update(end_wall_seconds=time.perf_counter()-started,
                            best_HH_at_end=int(best_hh),
                            improvement_after_restart=int(best_hh)-restart_rows[-1]['best_HH_before'])
                    best_contacts = contact_map(best,hmask,neighbors)
                    archive.observe(best,best_contacts,best_hh,total_steps,time.perf_counter()-started)
                    chosen,restart = archive.choose(best,best_contacts,best_hh,current)
                    current = np.asarray(chosen,np.int32).copy()
                    if restart is not None:
                        restart_rows.append(dict(worker_id=wid,proposal=total_steps,
                            wall_seconds=time.perf_counter()-started,stage=stage,next_cycle=cycle,
                            best_HH_before=int(best_hh),**restart,
                            end_wall_seconds=None,best_HH_at_end=None,improvement_after_restart=None))
                else:
                    current = best.copy()
                hh, excess, degrees = full_metrics(current, hmask, neighbors, cfg['cluster_threshold'])
                if restart is not None and int(hh) != restart['source_HH']:
                    raise AssertionError('Restored archive fold score mismatch')
                if (cfg['polar_refresh'] and stage>=cfg['archive_min_stage'] and not done
                    and time.perf_counter()-started-last_refresh>=cfg['polar_refresh_cooldown_seconds']):
                    apply_refresh('stagnant_cycle_advancement')
    wall = time.perf_counter()-started
    if refresh_rows:
        refresh_rows[-1]['best_HH_at_next_refresh']=int(best_hh)
        write_json(folder/f'worker_{wid:02d}_polar_refresh.json',refresh_rows)
    if restart_rows and restart_rows[-1]['end_wall_seconds'] is None:
        restart_rows[-1].update(end_wall_seconds=wall,best_HH_at_end=int(best_hh),
            improvement_after_restart=int(best_hh)-restart_rows[-1]['best_HH_before'])
    if archive is not None:
        write_json(folder / f'worker_{wid:02d}_restart_archive.json',
            dict(worker_id=wid,seed=seed,**archive.snapshot()))
        write_json(folder / f'worker_{wid:02d}_restarts.json',restart_rows)
    if np.any(by_proposal[:,0] != prior_proposals[:,0]):
        proposal_windows.extend(window_rows(by_proposal, prior_proposals,
            construction_timings, prior_timings, wid, window_start, wall))
    coordinates = coords[best].tolist()
    delta_condition = contact_condition_stats-prior_condition
    delta_condition[:,15] = contact_condition_stats[:,15]
    condition_windows.extend(dict(wall_start=window_start,wall_end=wall,
        generated_HH_definition='cumulative maximum at window end',**r)
        for r in condition_rows(delta_condition,wid))
    validation = validate_coordinates(cfg['seq'], coordinates, cfg['l_size'])
    if not validation['valid'] or validation['HH'] != best_hh:
        raise AssertionError(f'Worker {wid}: independent fold validation failed')
    result = {'worker_id':wid, 'seed':seed, 'best_HH':int(best_hh), 'best_energy':-int(best_hh),
        'best_excess':int(best_excess), 'best_positions':best.tolist(), 'coords':coordinates,
        'valid':True, 'target_hit':bool(best_hh >= cfg['target_HH']) if cfg['target_HH'] is not None else None, 'stop_reason':reason,
        'wall_seconds':wall, 'preparation_seconds':prep_seconds, 'steps':total_steps,
        'first_recorded':first, 'statistics':dict(zip(
            ['proposals','valid_pools','candidates','distinct_signature_sum','transitions','expansions','geometry_prunes','duplicates','pool_capacity'],
            map(int,totals))), 'cycle_trace':cycle_rows, 'effort_trace':effort_rows,
        'block_statistics':block_rows(by_length, wid),
        'proposal_statistics':proposal_rows(by_proposal, wid),
        'proposal_windows':proposal_windows,
        'construction_timings':construction_timings.tolist(),
        'resource_statistics':resource_rows(decision_stats,wid),
        'interval_statistics':interval_rows(interval_stats,wid),
        'interval_guidance_statistics':guidance_rows(guidance_stats,wid),
        'near_best_contact_map_samples':novelty_samples,
        'near_best_distinct_contact_map_fingerprints':len(novelty_seen),
        'near_best_contact_map_novelty':novelty_rows,
        'novelty_definition':'Sample current folds within two contacts of worker best every 4096 proposals; '
            '64-bit contact-map fingerprints, not an exact exhaustive diversity count; diagnostic only',
        'paired_beam_selection_statistics':dict(zip(SELECTION_FIELDS,map(int,selection_stats))),
        'best_generated_HH':generated_best, 'first_generated':generated_first,
        'contact_condition_statistics':condition_rows(contact_condition_stats,wid),
        'contact_condition_windows':condition_windows}
    result['diversity_restart_count'] = archive.restarts if archive is not None else 0
    result['transition_rule'] = cfg.get('transition_rule', 'direct_heat_bath')
    result['transition_statistics'] = dict(zip(
        ('scored_pools','uniform_draws','nonworse_accepted','uphill_proposed',
         'uphill_accepted','uphill_rejected','empty_distinct_pools','empty_generated_pools'),
        map(int,transition_stats)))
    if archive is not None and cfg.get('route_diversity',False):
        result['route_archive_statistics']={key:value for key,value in archive.snapshot().items()
                                            if key!='entries'}
    result['restart_trace'] = restart_rows
    result['polar_refresh_trace']=refresh_rows
    result['polar_refresh_count']=len(refresh_rows)
    result['polar_refresh_proposals']=sum(e['attempts'] for e in refresh_rows)
    result['vacancy_focus_trace']=focus_rows
    result['vacancy_focus_steps']=focus_steps
    result['vacancy_focus_statistics']=focus_counts(focus_stats)
    write_json(folder / f'worker_{wid:02d}_result.json', result)
    write_json(folder / f'worker_{wid:02d}_validation.json', validation)
    return result


def parse_config(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--transition-rule', choices=('direct_heat_bath', 'uniform_metropolis'))
    ap.add_argument('--contact-condition-probability',type=float,
                    help='Condition this fraction of eligible near-best guided paired reconstructions')
    ap.add_argument('--profile', type=Path, help='Historical run_config.json, copied into the new run for provenance')
    ap.add_argument('--variant', choices=VARIANTS, default='full')
    ap.add_argument('--seq')
    ap.add_argument('--benchmark')
    target = ap.add_mutually_exclusive_group()
    target.add_argument('--target-hh', type=int)
    target.add_argument('--no-target-hh', action='store_true',
                        help='No reference target; disable target stopping and leave target statistics null')
    ap.add_argument('--l-size', type=int)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--wall-seconds', type=float)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--max-proposals', type=int)
    ap.add_argument('--cycle-steps', help='Four comma-separated positive cycle lengths')
    ap.add_argument('--stage-block-max', help='Four comma-separated maximum block lengths')
    ap.add_argument('--adaptive-window-steps', type=int)
    ap.add_argument('--beam-width', type=int)
    ap.add_argument('--max-adaptive-beam',type=int)
    ap.add_argument('--t0',type=float)
    ap.add_argument('--t1',type=float)
    ap.add_argument('--control-chunk-steps', type=int)
    ap.add_argument('--joint-probability', type=float)
    ap.add_argument('--triple-probability',type=float,
                    help='Overall probability at block cap >=24, carved out of joint_probability')
    ap.add_argument('--quadruple-probability',type=float,
                    help='Additional probability at block cap >=24, carved out of single-block proposals')
    ap.add_argument('--guided-coupled-probability', type=float)
    ap.add_argument('--p-run-probability', type=float,
                    help='Fraction of paired proposals attempting sequence-derived P-run windows')
    ap.add_argument('--guided-anchor-attempts', type=int)
    ap.add_argument('--no-diversity-restart', action='store_true',
                    help='Restore the inherited CG policy; archive disabled')
    ap.add_argument('--route-diversity', action='store_true',
                    help='Distinguish HP/PP routing environments in the DR restart archive')
    ap.add_argument('--polar-refresh',action='store_true',
                    help='Reconstruct P-only intervals on stagnation without altering H coordinates or scores')
    ap.add_argument('--vacancy-responsive',action='store_true',
                    help='Temporarily bias existing guided intervals toward vacancies opened by polar refresh')
    ap.add_argument('--boundary-policy',choices=('random','constrained_first'))
    ap.add_argument('--unguided-probability',type=float)
    ap.add_argument('--effort-policy',choices=('legacy','resource'))
    ap.add_argument('--cycle-policy',choices=('legacy','terminal'))
    ap.add_argument('--paired-beam-selection',choices=('baseline','contact_stratified'))
    ap.add_argument('--joint-beam-width', type=int)
    ap.add_argument('--joint-parents', type=int)
    ap.add_argument('--joint-expansion-budget', type=int)
    ap.add_argument('--late-cycle-cap-fraction', type=float,
                    help='Budget fraction limiting estimated cycle duration at stage >=3; default 0.125')
    stop = ap.add_mutually_exclusive_group()
    stop.add_argument('--target-stop', dest='target_stop', action='store_true')
    stop.add_argument('--no-target-stop', dest='target_stop', action='store_false')
    stop.add_argument('--all-workers-target-stop', action='store_true',
                      help='Stop each worker at its own target; wait for all workers or their wall budgets')
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
        if 'transition_rule' in inherited:
            cfg['transition_rule'] = inherited['transition_rule']
        record = args.profile.parent / 'run_command.ps1'
        if record.is_file():
            structured = args.profile.parent / 'run_command.json'
            if structured.is_file():
                argv_record = json.loads(structured.read_text(encoding='utf-8-sig'))['argv']
                if not isinstance(argv_record,list) or not all(isinstance(x,str) for x in argv_record):
                    ap.error('recorded argv must be a string array')
                tokens = argv_record[2:]
            else:
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
    if args.transition_rule is not None:
        cfg['transition_rule'] = args.transition_rule
    if cfg.get('transition_rule', 'direct_heat_bath') not in ('direct_heat_bath', 'uniform_metropolis'):
        ap.error('invalid transition_rule in profile')
    for key in ('seq','benchmark','l_size','workers','wall_seconds','seed','adaptive_window_steps','beam_width','control_chunk_steps','max_adaptive_beam',
                'joint_probability','joint_beam_width','joint_parents','joint_expansion_budget','late_cycle_cap_fraction','paired_beam_selection',
                'boundary_policy','unguided_probability','effort_policy','cycle_policy',
                'guided_coupled_probability','guided_anchor_attempts','triple_probability','quadruple_probability',
                'p_run_probability','contact_condition_probability'):
        value = getattr(args,key)
        if value is not None:
            cfg[key] = value
    if args.t0 is not None: cfg['temperature_t0']=args.t0
    if args.no_diversity_restart: cfg['diversity_restart']=False
    if args.route_diversity: cfg['route_diversity']=True
    if args.polar_refresh: cfg['polar_refresh']=True
    if args.vacancy_responsive: cfg['vacancy_responsive']=True
    if args.t1 is not None: cfg['temperature_t1']=args.t1
    if args.target_hh is not None:
        cfg['target_HH'] = args.target_hh
    if args.max_proposals is not None:
        cfg['max_proposals_per_worker'] = args.max_proposals
    if args.cycle_steps:
        cfg['cycle_steps'] = [int(x) for x in args.cycle_steps.split(',')]
    if args.stage_block_max:
        cfg['stage_block_max'] = [int(x) for x in args.stage_block_max.split(',')]
    if args.target_stop is not None:
        cfg['target_stop_enabled'] = args.target_stop
        cfg['target_stop_scope'] = 'any_worker'
    if args.all_workers_target_stop:
        cfg['target_stop_enabled'] = True
        cfg['target_stop_scope'] = 'all_workers'
    if args.no_target_hh:
        if args.target_stop is True or args.all_workers_target_stop:
            ap.error('target stopping requires a reference target')
        cfg['target_HH'] = None
        cfg['target_stop_enabled'] = False
    if cfg['target_stop_scope'] not in ('any_worker', 'all_workers'):
        ap.error('target_stop_scope must be any_worker or all_workers')
    cfg['seq'] = cfg.get('seq','').strip().upper()
    cfg.setdefault('benchmark','direct')
    cfg['variant'] = args.variant
    cfg['algorithm'] = 'RD-DR-Q-ABHA-FCC' if cfg['route_diversity'] else 'DR-Q-ABHA-FCC'
    if cfg['polar_refresh']:
        cfg['algorithm']='PR-DR-Q-ABHA-FCC'
        cfg['polar_refresh_definition']=(
            'At stagnant cycle advancement (minimum stage 3) or 1200 seconds without a new worker-best contact, '
            'with 600-second cooldown, run up to 128 P-only block reconstructions. '
            'All H sites and the exact H-H map are fixed; choose uniformly among unique completed assignments '
            'and current, the equal-energy finite-pool heat bath. Retain the existing cooling position. '
            'If a wall-triggered current fold is more than two contacts below best, restore best first. '
            'Best remains untouched. Refresh consumes the same wall budget, with separate proposal counters. '
            'No external warm start, no pull/pivot move, no HP or PP energy; no equilibrium-sampling claim.')
    cfg['implementation_ancestor'] = 'DR-Q-ABHA-FCC, completed 3no3 L17 run, 2026-09-26/27'
    if cfg['vacancy_responsive']:
        if not cfg['polar_refresh']:
            ap.error('Vacancy-responsive selection requires polar_refresh')
        if (not isinstance(cfg['vacancy_focus_steps'],int) or cfg['vacancy_focus_steps']<=0
                or not math.isfinite(cfg['vacancy_boost']) or cfg['vacancy_boost']<0):
            ap.error('Vacancy focus requires positive integer steps and finite nonnegative boost')
        cfg['algorithm']='VR-PR-Q-ABHA-FCC'
        cfg['implementation_ancestor']='PR-DR-Q-ABHA-FCC, 473-contact 3no3 L17 six-hour run, 2026-09-28'
        cfg['vacancy_focus_definition']=(
            'After each unchanged neutral polar refresh, mark sites occupied before but empty after. '
            'For vacancy_focus_steps ordinary proposals, multiply existing H-anchor and partner weights '
            'by 1+vacancy_boost*min(v,2)/2, where v counts marked neighbors still empty in the current fold. '
            'No H residue is excluded; only the existing guided coupled branch is biased. '
            'No forced contact, extra proposal, altered beam, score bonus or acceptance rule. '
            'Pulse expires by proposal count; a new refresh replaces it. All workers start from fresh SAWs.')
    if not math.isfinite(cfg['triple_probability']) or not 0<=cfg['triple_probability']<=cfg['joint_probability']:
        ap.error('triple_probability must be between zero and joint_probability')
    if cfg['triple_probability']:
        if max(cfg['stage_block_max'])!=24:
            ap.error('J3 controlled profile requires a 24-residue maximum')
        cfg['algorithm']='J3-PR-Q-ABHA-FCC'
        cfg['implementation_ancestor']='VR-PR-Q-ABHA-FCC seed A, six-hour 473-contact run'
        cfg['triple_definition']=(
            'At block cap >=24 replace 10 percentage points of the 25% joint branch with '
            'three disjoint 6-8 residue intervals (total <=24). Select two via the inherited '
            'low-degree missing-H-pair policy with cap16, and a third around an H or P '
            'occupying a common-neighbor site outside both intervals and boundaries. '
            'Uniform eligible blocker/interval selection, bounded attempts; no eligible '
            'selection is a null proposal. Reconstruct all intervals in one alternating '
            'beam of width24 with shared 12000-expansion limit. No forced new contact. '
            'Score the full completed assignment exactly, retain current in heat-bath pool. '
            'All original controller, scoring, refresh and archive policies are retained.')
    if (not math.isfinite(cfg['quadruple_probability']) or
            not 0<=cfg['quadruple_probability']<=1-cfg['joint_probability']):
        ap.error('quadruple_probability must be finite and between zero and 1-joint_probability')
    if cfg['quadruple_probability']:
        if max(cfg['stage_block_max'])!=24:
            ap.error('J4 retains the 24-residue cap for existing proposal families')
        cfg['algorithm']='J4-PR-Q-ABHA-FCC'
        cfg['implementation_ancestor']='Original J3-PR-Q-ABHA-FCC six-hour fresh-start profile'
        cfg['quadruple_definition']=(
            'At existing block cap >=24 carve quadruple_probability from the single branch, '
            'leaving paired and triple probabilities unchanged. Select an inherited J3 '
            'missing-pair/blocker triple, then a fourth 6-8 interval around an occupied '
            'alternative common-neighbor site of the H pair or neighbor of the first '
            'blocker. All intervals have fixed external boundaries and disjoint interiors '
            'with at least one fixed residue separating them. Total movable length 24-32. '
            'Uniform eligible fourth blocker/start after a uniform 6-8 length draw; '
            'bounded attempts, null proposal on failure, no forced contact. Release all '
            'four intervals together and alternate their growth in the same joint beam '
            'of width24 and shared 12000-expansion cap. Complete valid unique assignments '
            'and current enter unchanged direct heat-bath selection. Existing scoring, '
            'controller, SAWs, archive and polar-refresh settings are retained.')
    if not math.isfinite(cfg['p_run_probability']) or not 0<=cfg['p_run_probability']<=1:
        ap.error('p_run_probability must be finite and between zero and one')
    if cfg['p_run_probability']:
        if cfg['joint_probability']<=cfg['triple_probability']:
            ap.error('P-run selection requires a nonzero paired proposal branch')
        cfg['algorithm']='PGR-J4-Q-ABHA-FCC' if cfg['quadruple_probability'] else 'PGR-Q-ABHA-FCC'
        cfg['implementation_ancestor']='J4-PR-Q-ABHA-FCC, completed six-hour profiles of 2026-10-02'
        cfg['p_run_definition']=(
            'Only paired selection changes. With p_run_probability attempt precomputed whole-P-run '
            'windows including every available adjacent H as movable residues; minimally pad to '
            'six residues. Each length <=min(12,stage_cap//2); oversized runs are skipped. '
            'Select a movable H flank with inherited low-degree/vacancy weights and a '
            'nonconsecutive, noncontacting H flank exactly two FCC edges away. Uniformly '
            'sample eligible window pairs from different runs with a fixed intervening residue. '
            'After bounded failure use the inherited 50/50 spatial/contact-guided selector. '
            'The remaining paired attempts directly retain that selector. Triple/four-block '
            'selectors, boundaries, constructor, expansion limits, heat bath, scores, '
            'initialization, archive, refresh and controller remain unchanged. '
            'No contact constraint or extra energy term.')
    if not math.isfinite(cfg['contact_condition_probability']) or not 0 <= cfg['contact_condition_probability'] <= 1:
        ap.error('contact_condition_probability must be finite and between zero and one')
    if cfg['contact_condition_probability']:
        cfg['algorithm'] = 'CCP-PGR-Q-ABHA-FCC'
        cfg['implementation_ancestor'] = 'Original PGR-J4 six-hour fresh-start profiles, 2026-10-02'
        cfg['contact_condition_definition'] = (
            'After two completed cooling cycles without a worker-best physical HH improvement, '
            'at stage >=3 and current HH >= best-1, condition 25% of guided paired attempts '
            '(modes 2 and 8) on closing the selected missing nonconsecutive HH pair. '
            'No selector change. Optimistic contact-reachability pruning in the alternating joint '
            'beam; completed conditioned assignments must close the target pair. Width and total '
            'expansion cap unchanged. No extra fallback on failure; null proposal. Physical and '
            'compact-core energies, direct heat bath and all other proposal families unchanged. '
            'Reset eligibility on new physical best, including within a chunk. Interrupted/terminal '
            'cycles are not counted as completed stagnant cycles. Not exact enumeration or an '
            'equilibrium-sampling claim. Fresh SAWs and worker-local archives retained.')
    cfg['archive_definition'] = ('Worker-local visited-fold archive, capacity 16, physical-contact '
        'gap <=2, exact nonconsecutive H-H contact maps with Jaccard diversity. Sample every '
        '4096 proposals and on best-contact improvements. Quality-weighted farthest-point '
        'retention; restart weights exp(-contact_gap)*distance_to_best/(1+restart_visits). '
        'At existing two-stagnant-cycle stage advancement, new stage >=3, choose a different '
        'near-best contact map instead of always restoring best; otherwise inherit best restore. '
        'No extra restart during terminal cooling; worker best is never overwritten by a worse '
        'fold. Independent archive RNG; fresh SAWs, no external warm starts or cross-worker sharing. '
        'Direct heat-bath proposal kernel and temperatures unchanged. Restart is an explicit '
        'optimization mechanism, not an equilibrium sampling claim.')
    if cfg['route_diversity']:
        cfg['archive_definition'] = (
            'Same worker-local visited-fold capacity, contact gap, observation and restart clocks '
            'as DR. Distinguish the noncovalent HP/PP adjacency environment as a route proxy. '
            'Archive distance=max(HH-map Jaccard,archive_route_weight*HP/PP-map Jaccard). '
            'Retain at most archive_routes_per_map states per HH map within the same total '
            'capacity. Restart may select a different routing environment with the same HH map '
            'as best; independent archive RNG and physical-best preservation are unchanged. '
            'No HP/PP attraction or energy bonus, no kernel change, no archive sharing or '
            'external warm start. This is an optimization memory mechanism, not Boltzmann sampling.')
    cfg['coupled_definition'] = (f"{100*(1-cfg['joint_probability']):g}% historical single-block proposals, "
        f"{100*cfg['joint_probability']:g}% paired-beam joint reconstruction; "
        'two disjoint 6-12-residue intervals with a fixed sequence gap, spatial separation <=2 FCC edges; '
        'total movable residues <= current stage cap. Mixture of historical random interval selection '
        'and contact-guided interval selection as defined below. Both released before construction; '
        'alternating interval growth in one joint beam. '
        f"Width {cfg['joint_beam_width']}, shared {cfg['joint_expansion_budget']}-expansion bound. "
        'Contacts to fixed residues and both partial intervals scored once on extension; '
        'only complete joint assignments enter unchanged direct heat-bath selection. '
        'Diverse near-best archive restart at stagnant stage advancement when enabled.')
    cfg['joint_constructor'] = 'alternating_paired_beam'
    cfg['guided_interval_definition'] = (
        'Conditional on a coupled proposal, use guided_coupled_probability for guided selection. '
        'Draw each interval length uniformly from 6..min(12,stage_cap//2), unchanged. '
        'Sample an H anchor with weight 1/(1+nonbonded_H_degree)^2; consider nonconsecutive '
        'H partners at exactly two FCC graph edges that can occupy separate intervals with a fixed gap. '
        'Sample partner with the same degree weight. Select compatible starts containing the H pair '
        'with weight 1+P_count/total_block_length. After guided_anchor_attempts unsuccessful anchor '
        'draws, fall back to the original random selector with the already drawn lengths. '
        'No forced contact, target-dependent rule, scoring change, or additional feasibility rejection.')
    cfg['paired_beam_retention_definition'] = (
        'At each common construction depth retain width/2, width/3, remainder for partial physical '
        'contact deficits 0,1,2 relative to the highest count among that depth expansions. Width 24 '
        'gives quotas 12/8/4. Within each quota prefer distinct contact/P-route/tip signatures, '
        'then fill by existing noisy construction score. Unused quota fills globally from all '
        'remaining candidates by the same score, with diversity preferred; gaps >2 are allowed '
        'only through redistribution. No BKV-dependent scoring, archive addition or heat-bath change.'
        if cfg['paired_beam_selection']=='contact_stratified' else 'Original score-quality/diversity beam retention')
    cfg['joint_parents_definition'] = 'Must be 1: one paired beam, no sequential first-block parent selection'
    cfg['bidirectional'] = args.variant not in ('optimized_baseline','fast_only','no_bidirectional')
    cfg['budget_schedule'] = args.variant not in ('optimized_baseline','fast_only','no_budget_schedule')
    cfg['length_effort'] = args.variant not in ('optimized_baseline','fast_only','no_length_effort')
    cfg['components'] = dict(zip(('routing','diversity','adaptive_effort'), VARIANTS[args.variant]))
    cfg['components'].update({k:cfg[k] for k in ('bidirectional','budget_schedule','length_effort')})
    cfg['schedule_definition'] = ('Proposal cycles; two stagnant cycles grow requested length. Budget quarters impose minimum '
        'objective stages 0/1/2/3; final quarter uses zero cluster shaping. Cycle length capped at boundaries using '
        'observed throughput, never timed candidate construction. Proposal-capped tests use proposal quarters instead.') if cfg['budget_schedule'] else 'Unchanged RA-v1 proposal-cycle stage controller'
    cfg['cycle_cap_definition'] = ('Stages 0-2 retain W/8. From stage 3, use '
        'W*late_cycle_cap_fraction, also bounded by remaining stage/budget time. '
        'This is an estimated duration converted to proposals at the cycle boundary, '
        'not a guaranteed completed-cycle duration. The global wall limit is unchanged.')
    cfg['boundary_policy_definition'] = ('For internal intervals prefer the boundary with fewer '
        'optimistically feasible first sites with probability 0.75; retain 0.25 opposite-direction '
        'exploration. Ties are 50/50. Both movable intervals are released before checking coupled '
        'boundaries. This preference does not add a feasibility rejection.')
    cfg['partial_preference_definition'] = ('Each proposal independently disables partial contact, '
        'distance and continuation score bonuses with unguided_probability. Stochastic ranking '
        'and signature diversity remain; this is not uniform path sampling. Such proposals bypass '
        'contact-stratified quotas. Completed folds always retain the same physical/shaping energy.')
    cfg['resource_effort_definition'] = ('Resource mode probes 24/48/96 in three of every eight '
        '8000-proposal windows, exploiting in five. Evidence only from single blocks longer than 8: '
        '(sum positive pool gain over current + 100*gain over worker best + 0.1*near-best pools) '
        '/ expansions. Expansion count is a cost proxy. Discount history by 0.8/window and 0.5/cycle. '
        'No BKV threshold is used; 96 is allowed only by a scheduled probe or observed yield.')
    cfg['terminal_cycle_definition'] = ('Terminal policy prevents deadline-sized reheating loops. '
        'At a boundary, if a requested cycle will not fit the estimated remaining time, or less '
        'than 5% of the budget remains, cool once across the remaining budget without restarting. '
        'This final pass uses wall time, or proposal index in proposal-limited tests. Other cycles '
        'remain proposal-count cycles; cycle caps are unchanged. Wall-time results are not bitwise replayable.')
    if len(cfg['seq']) < 4 or set(cfg['seq'])-{'H','P'}:
        ap.error('provide an HP sequence of at least four residues or a profile containing one')
    if (not isinstance(cfg['route_diversity'],bool) or not 0<cfg['archive_route_weight']<=1
            or not 1<=cfg['archive_routes_per_map']<=cfg['archive_capacity']):
        ap.error('Invalid route-diversity configuration')
    if cfg['route_diversity'] and not cfg['diversity_restart']:
        ap.error('Route diversity requires the restart archive')
    if (not isinstance(cfg['polar_refresh'],bool) or cfg['polar_refresh_attempts']<1
        or cfg['polar_refresh_expansion_cap']<1 or cfg['polar_refresh_patience_seconds']<=0
        or cfg['polar_refresh_cooldown_seconds']<=0):
        ap.error('Invalid neutral polar refresh configuration')
    if cfg['polar_refresh'] and (not cfg['diversity_restart'] or cfg['route_diversity']):
        ap.error('Polar refresh uses the original DR archive, not the passive route archive')
    if (not isinstance(cfg['diversity_restart'],bool) or cfg['archive_capacity']<2
        or cfg['archive_max_gap']<0 or cfg['archive_sample_steps']<1
        or cfg['archive_min_stage']<0 or not 0<=cfg['archive_min_distance']<=1):
        ap.error('Invalid diversity archive configuration')
    if 'target_HH' not in cfg or (cfg['target_HH'] is not None and cfg['target_HH'] <= 0):
        ap.error('target_HH must be positive, or explicitly disabled using --no-target-hh')
    if cfg['target_HH'] is None and cfg['target_stop_enabled']:
        ap.error('target stopping requires a reference target')
    for key in ('l_size','workers','control_chunk_steps','beam_width','adaptive_window_steps','cycles_without_improvement'):
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
    if not math.isfinite(cfg['joint_probability']) or not 0 <= cfg['joint_probability'] <= 1:
        ap.error('joint-probability must be between 0 and 1')
    if not math.isfinite(cfg['guided_coupled_probability']) or not 0 <= cfg['guided_coupled_probability'] <= 1:
        ap.error('guided-coupled-probability must be in [0,1]')
    if not 1 <= cfg['guided_anchor_attempts'] <= 16:
        ap.error('guided-anchor-attempts must be in [1,16]')
    if cfg['boundary_policy'] not in ('random','constrained_first') or cfg['effort_policy'] not in ('legacy','resource') or cfg['cycle_policy'] not in ('legacy','terminal'):
        ap.error('unknown BR controller policy')
    if not math.isfinite(cfg['unguided_probability']) or not 0 <= cfg['unguided_probability'] <= 1:
        ap.error('unguided-probability must be in [0,1]')
    if cfg['effort_policy']=='resource' and (cfg['beam_width']!=24 or cfg['max_adaptive_beam'] not in (24,48,96)):
        ap.error('resource effort uses base 24 and maximum 24, 48 or 96')
    if cfg['paired_beam_selection'] not in ('baseline','contact_stratified'):
        ap.error('unknown paired beam selection')
    if not math.isfinite(cfg['late_cycle_cap_fraction']) or not 0 < cfg['late_cycle_cap_fraction'] <= 1:
        ap.error('late-cycle-cap-fraction must be finite and in (0,1]')
    if not 1 <= cfg['joint_beam_width'] <= 24 or cfg['joint_parents'] != 1:
        ap.error('paired joint beam width must be 1-24 and joint-parents must be 1')
    if not 1 <= cfg['joint_expansion_budget'] <= 50000:
        ap.error('joint expansion budget must be 1-50000')
    for label in (cfg['benchmark'], args.run_name):
        if not label or label in ('.','..') or any(c in label for c in '/\\:'):
            ap.error('benchmark and run-name must be single safe directory names')
    if cfg.get('transition_rule') == 'uniform_metropolis':
        import re
        for key, value in list(cfg.items()):
            if key.endswith('_definition') and isinstance(value, str):
                cfg[key] = re.sub(r'(?:unchanged )?(?:direct )?(?:finite-pool )?heat[- ]bath(?: selection| transition| update)?',
                    'uniform candidate proposal with Metropolis acceptance', value, flags=re.IGNORECASE)
        cfg['transition_definition'] = (
            'Draw uniformly from finite-scored completed candidates, excluding current and '
            'excluded duplicates. Accept with min(1, exp(-(E_candidate-E_current)/T)); '
            'otherwise retain current. No heat-bath weighting or second acceptance test. '
            'The biased reconstruction proposal is not corrected for reverse proposal density; '
            'this is an optimization ablation, not exact equilibrium MH sampling. '
            'Construction, compact-core shaping, temperatures and archive behavior are retained.')
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
    cfg['block_telemetry_definition'] = ('Exact per-length counters; reconstruction timing sampled every 4096 '
        'cycle proposals, including Python timing-bridge overhead. Timing is diagnostic, not total construction time. '
        'Both arms have identical instrumentation; it consumes no random draws. Wall-capped trajectories can still differ.')
    cfg['coupled_telemetry_definition'] = ('Block length for joint proposals is the sum of both lengths; '
        'length 0 is a failed interval selection. Proposal statistics separate single and coupled updates. '
        'First-generated records describe scored complete candidates, not necessarily visited states; '
        'reported best/mean/SD continue to use visited worker-best folds only. '
        'Proposal windows separate single/joint counters and sampled construction time per budget third '
        '(hourly for production), observed at chunk boundaries; timings are samples, not total CPU time.')
    if cfg['triple_probability']:
        cfg['coupled_definition'] += (' At cap24, triple_probability is taken from this joint branch, '
            'not added: production 75% single / 15% paired / 10% triple.')
        cfg['coupled_telemetry_definition'] += (
            ' J3 records triple as a separate proposal/resource/window row; all_intervals_changed '
            'requires each selected interval to change. First-hit records include all three intervals '
            'and the blocker index. Legacy two-dimensional interval histograms exclude triples; '
            'the block-length histogram includes their total length. Failed triple selection is counted '
            'as triple with zero candidates, never converted to another proposal family.')
    if cfg['quadruple_probability']:
        cfg['coupled_definition'] += (
            f" J4 additionally takes {100*cfg['quadruple_probability']:g} percentage points from single proposals at cap24; "
            'production 70% single / 15% paired / 10% triple / 5% quadruple. '
            'Only quadruple proposals may move up to32 residues.')
        cfg['coupled_telemetry_definition'] += (
            ' J4 records quadruple proposals separately, including failed selection, '
            'four intervals, both blocker indices, exact union score and all-four-changed '
            'transitions. Legacy two-interval histograms exclude triples and quadruples; '
            'length histograms contain the total union size. Other proposal counts and '
            'selection weights are unchanged. Four-block construction is not exact Gibbs.')
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
    if cfg['p_run_probability']:
        h = np.asarray([c=='H' for c in cfg['seq']],np.int8)
        write_json(folder/'p_run_window_catalog.json',describe_catalog(h,cfg['stage_block_max']))
    warm_kernel()
    if cfg['p_run_probability']:
        xyz=np.asarray(coords,np.int32)
        warm_p_run_kernel(xyz,neighbors_array(fcc_neighbor_sets([tuple(p) for p in xyz])),
                         h,np.asarray(starts[0],np.int32),cfg)
    if cfg['polar_refresh']:
        xyz=np.asarray(coords,np.int32)
        warm_refresh(xyz,neighbors_array(fcc_neighbor_sets([tuple(p) for p in xyz])),
                     np.asarray([c=='H' for c in cfg['seq']],np.int8),np.asarray(starts[0],np.int32))
    context = mp.get_context('spawn')
    results=[]
    with context.Manager() as manager:
        event = manager.Event()
        abort_event = manager.Event()
        # In all-workers mode a target hit must not interrupt another worker.
        target_event = event if cfg['target_stop_scope'] == 'any_worker' else None
        tasks=[{'config':cfg,'worker_id':i,'folder':str(folder),'coords':coords,'initial':p,'stop_event':target_event,'abort_event':abort_event}
               for i,p in enumerate(starts)]
        write_json(folder/'batch_progress.json', {'completed_workers':0,
            'target_hit_workers':0 if cfg['target_HH'] is not None else None, 'workers':cfg['workers'], 'status':'running',
            'target_stop_enabled':cfg['target_stop_enabled'], 'target_stop_scope':cfg['target_stop_scope']})
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=context) as pool:
            jobs=[pool.submit(worker,t) for t in tasks]
            try:
                for future in concurrent.futures.as_completed(jobs):
                    result=future.result()
                    results.append(result)
                    print(f"worker {result['worker_id']:02d}: HH={result['best_HH']} wall={result['wall_seconds']:.3f}s stop={result['stop_reason']}",flush=True)
                    write_json(folder/'batch_progress.json', {'completed_workers':len(results),
                        'target_hit_workers':sum(bool(r['target_hit']) for r in results) if cfg['target_HH'] is not None else None,
                        'workers':cfg['workers'], 'status':'completed' if len(results)==cfg['workers'] else 'running',
                        'target_stop_enabled':cfg['target_stop_enabled'], 'target_stop_scope':cfg['target_stop_scope']})
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
        'workers_recorded':len(results),'worker_target_hit_rate_pct':100*sum(r['target_hit'] for r in results)/len(results) if cfg['target_HH'] is not None else None,
        'mean_worker_wall_seconds':statistics.mean(r['wall_seconds'] for r in results),
        'earliest_best_observed_wall_seconds':first_observation(max(contacts)),
        'earliest_target_observed_wall_seconds':first_observation(cfg['target_HH']) if cfg['target_HH'] is not None else None,
        'mean_valid_pools_per_second':statistics.mean(r['statistics']['valid_pools']/max(r['wall_seconds'],1.e-12) for r in results),
        'mean_candidates_per_second':statistics.mean(r['statistics']['candidates']/max(r['wall_seconds'],1.e-12) for r in results),
        'total_process_wall_seconds':time.perf_counter()-total_started,
        'statistical_unit':'independent worker-best fold; not repeated 16-worker batch-best',
        'full_annealing_budgets_completed':all(r['stop_reason']=='wall_budget' for r in results),
        'target_stop_enabled':cfg['target_stop_enabled'], 'target_stop_scope':cfg['target_stop_scope'],
        'all_workers_target_hit':all(r['target_hit'] for r in results) if cfg['target_HH'] is not None else None,
        'worker_stop_counts':{reason:sum(r['stop_reason']==reason for r in results)
                              for reason in sorted({r['stop_reason'] for r in results})},
        'run_dir':str(folder),'valid':all(r['valid'] for r in results)}
    summary['diversity_restarts'] = sum(r['diversity_restart_count'] for r in results)
    summary['polar_refresh_count']=sum(r['polar_refresh_count'] for r in results)
    summary['polar_refresh_proposals']=sum(r['polar_refresh_proposals'] for r in results)
    summary['polar_refresh_changed_routes']=sum(e['polar_pairs_changed']>0 for r in results for e in r['polar_refresh_trace'])
    summary['polar_refresh_wall_seconds_total']=sum(e['refresh_wall_seconds'] for r in results for e in r['polar_refresh_trace'])
    summary['vacancy_focus_steps']=sum(r['vacancy_focus_steps'] for r in results)
    summary['vacancy_focus_statistics']={key:sum(r['vacancy_focus_statistics'][key] for r in results)
                                         for key in results[0]['vacancy_focus_statistics']}
    if cfg['route_diversity']:
        summary['same_H_map_route_admissions'] = sum(r['route_archive_statistics']['same_H_map_admissions'] for r in results)
        summary['same_H_map_as_best_restarts'] = sum(r['route_archive_statistics']['same_H_map_as_best_restarts'] for r in results)
    summary['first_473_or_more_wall_seconds'] = first_observation(473) if cfg['benchmark']=='3no3' else None
    summary['first_474_or_more_wall_seconds'] = first_observation(474) if cfg['benchmark']=='3no3' else None
    summary['first_IWL_match_wall_seconds'] = first_observation(477) if cfg['benchmark']=='3no3' else None
    summary['first_IWL_exceeded_wall_seconds'] = first_observation(478) if cfg['benchmark']=='3no3' else None
    write_json(folder/'worker_results.json',results)
    write_json(folder/'best.json',dict(best,seq=cfg['seq'],L=cfg['l_size'],HH=best['best_HH']))
    write_json(folder/'validation_report.json',{'all_valid':True,'validated_workers':len(results),'method':'independent coordinate-distance contact count'})
    write_json(folder/'q_rbha_result.json',summary)
    for name,rows in [('restart_trace',[x for r in results for x in r['restart_trace']]),
                      ('cycle_trace',[x for r in results for x in r['cycle_trace']]),
                      ('interval_guidance_statistics',[x for r in results for x in r['interval_guidance_statistics']]),
                      ('near_best_contact_map_novelty',[dict(x,worker_id=r['worker_id']) for r in results for x in r['near_best_contact_map_novelty']]),
                      ('resource_statistics',[x for r in results for x in r['resource_statistics']]),
                      ('interval_statistics',[x for r in results for x in r['interval_statistics']]),
                      ('block_statistics',[x for r in results for x in r['block_statistics']]),
                      ('proposal_statistics',[x for r in results for x in r['proposal_statistics']]),
                      ('proposal_windows',[x for r in results for x in r['proposal_windows']]),
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
