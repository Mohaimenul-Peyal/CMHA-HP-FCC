"""Compiled feasible FCC block proposals and exact local contact deltas.

Routing scores affect construction only. Completed folds use the unchanged
pairwise-plus-decoded-shaping energy and a direct finite-pool heat-bath update.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def seed_kernel(seed):
    np.random.seed(seed)


@njit(cache=True)
def occupancy(pos, sites):
    out = np.full(sites, -1, np.int32)
    for i in range(len(pos)):
        out[pos[i]] = i
    return out


@njit(cache=True)
def full_metrics(pos, hmask, neighbors, threshold):
    occ = occupancy(pos, len(neighbors))
    degrees = np.zeros(len(pos), np.int32)
    hh = 0
    for i in range(len(pos)):
        if not hmask[i]:
            continue
        for v in neighbors[pos[i]]:
            if v >= 0:
                j = occ[v]
                if j >= 0 and hmask[j] and abs(i-j) > 1:
                    degrees[i] += 1
                    if j > i:
                        hh += 1
    excess = 0
    for i in range(len(pos)):
        if hmask[i]:
            excess += max(0, degrees[i] - threshold)
    return hh, excess, degrees


@njit(cache=True)
def local_delta(pos, block, a, b, hmask, neighbors, occ, hh, excess, degrees, threshold):
    """Inspect old/new block neighbors, count each changed contact exactly once.

    Restore occupancy before returning, including rejected candidates. Complexity
    is O(N + 12*m); the N term is the small degree-delta buffer, not an H-pair scan.
    """
    changes = np.zeros(len(pos), np.int32)
    count = 0
    for i in range(a, b):
        if not hmask[i]:
            continue
        for v in neighbors[pos[i]]:
            if v < 0:
                continue
            j = occ[v]
            if j >= 0 and hmask[j] and abs(i-j) > 1 and (j < a or j >= b or i < j):
                count -= 1
                changes[i] -= 1
                changes[j] -= 1
    for i in range(a, b):
        occ[pos[i]] = -1
    for i in range(a, b):
        occ[block[i-a]] = i
    for i in range(a, b):
        if not hmask[i]:
            continue
        for v in neighbors[block[i-a]]:
            if v < 0:
                continue
            j = occ[v]
            if j >= 0 and hmask[j] and abs(i-j) > 1 and (j < a or j >= b or i < j):
                count += 1
                changes[i] += 1
                changes[j] += 1
    for i in range(a, b):
        occ[block[i-a]] = -1
    for i in range(a, b):
        occ[pos[i]] = i
    for i in range(len(pos)):
        if hmask[i] and changes[i]:
            excess += max(0, degrees[i]+changes[i]-threshold) - max(0, degrees[i]-threshold)
    return hh+count, excess, changes


@njit(cache=True)
def graph_distance(coords, u, v):
    dx, dy, dz = abs(coords[u, 0]-coords[v, 0]), abs(coords[u, 1]-coords[v, 1]), abs(coords[u, 2]-coords[v, 2])
    return max(dx, dy, dz, (dx+dy+dz)//2)


@njit(cache=True)
def obstacle_distances(fixed, target, neighbors, depth_limit):
    """Optimistic free-graph distance: fixed obstacles removed, prefix not removed.

    Positive reachability does NOT certify a self-avoiding completion. FCC is not
    bipartite: do not apply the simple-cubic even/odd path-length pruning rule.
    """
    dist = np.full(len(neighbors), -1, np.int32)
    queue = np.empty(len(neighbors), np.int32)
    queue[0], dist[target] = target, 0
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        if dist[u] >= depth_limit:
            continue
        for v in neighbors[u]:
            if v >= 0 and fixed[v] < 0 and dist[v] < 0:
                dist[v] = dist[u]+1
                queue[tail] = v
                tail += 1
    return dist


@njit(cache=True)
def contains(path, depth, v):
    for i in range(depth):
        if path[i] == v:
            return True
    return False


@njit(cache=True)
def mix_id(value):
    x = np.uint64(value + 1)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def beam_candidates_scored(pos, a, b, hmask, coords, neighbors, occ, pair_reward, temperature,
                    width, routing, diversity, construction_floor, routing_weight, reverse=False):
    m, n = b-a, len(pos)
    fixed = occ.copy()
    for i in range(a, b):
        fixed[pos[i]] = -1
    forward = a > 0 and (b == n or not reverse)
    anchor = pos[a-1] if forward else pos[b]
    target = (pos[b] if b < n else -1) if forward else (pos[a-1] if a > 0 else -1)
    dist = np.full(len(neighbors), -1, np.int32)
    if routing and target >= 0:
        dist = obstacle_distances(fixed, target, neighbors, m+1)
    paths = np.full((width, m), -1, np.int32)
    scores = np.zeros(width)
    contact_sig = np.zeros(width, np.uint64)
    local_hh = np.zeros(width, np.int32)
    polar_sig = np.zeros(width, np.uint64)
    capacity = width * 12
    expansions = np.full((capacity, m), -1, np.int32)
    costs = np.empty(capacity)
    cs = np.empty(capacity, np.uint64)
    ps = np.empty(capacity, np.uint64)
    expanded_hh = np.empty(capacity, np.int32)
    fixed_counts = np.empty(len(neighbors), np.int32)
    fixed_signatures = np.empty(len(neighbors), np.uint64)
    beam_n, expanded_total, pruned_total = 1, 0, 0
    for depth in range(m):
        fixed_counts[:] = -1
        residue = a+depth if forward else b-1-depth
        remaining = m-depth-1
        count = 0
        for parent in range(beam_n):
            previous = anchor if depth == 0 else paths[parent, depth-1]
            for site in neighbors[previous]:
                if site < 0 or fixed[site] >= 0 or contains(paths[parent], depth, site):
                    continue
                expanded_total += 1
                if target >= 0:
                    d = graph_distance(coords, site, target)
                    if d > remaining+1 or (remaining == 0 and d != 1):
                        pruned_total += 1
                        continue
                    if routing and (dist[site] < 0 or dist[site] > remaining+1):
                        pruned_total += 1
                        continue
                continuations = 0
                if routing and remaining > 0:
                    for v in neighbors[site]:
                        if v < 0 or fixed[v] >= 0 or contains(paths[parent], depth, v):
                            continue
                        if target < 0 or (dist[v] >= 0 and dist[v] <= remaining):
                            continuations += 1
                    if continuations == 0:
                        pruned_total += 1
                        continue
                contacts = 0
                signature = contact_sig[parent]
                if hmask[residue]:
                    # Fixed contacts depend on residue and site, not on the beam prefix.
                    if fixed_counts[site] < 0:
                        field, sig = 0, np.uint64(0)
                        for v in neighbors[site]:
                            if v >= 0:
                                j = fixed[v]
                                if j >= 0 and hmask[j] and abs(j-residue) > 1:
                                    field += 1
                                    sig ^= mix_id(min(j, residue)*n+max(j, residue))
                        fixed_counts[site], fixed_signatures[site] = field, sig
                    contacts = fixed_counts[site]
                    signature ^= fixed_signatures[site]
                    for d in range(depth):
                        j = a+d if forward else b-1-d
                        if hmask[j] and abs(j-residue) > 1 and graph_distance(coords,site,paths[parent,d]) == 1:
                            contacts += 1
                            signature ^= mix_id(min(j, residue)*n+max(j, residue))
                psig = polar_sig[parent]
                if not hmask[residue]:
                    psig ^= mix_id(site)
                u = max(1.e-12, min(1.-1.e-12, np.random.random()))
                noise_scale = max(0.03, temperature)
                if diversity:
                    noise_scale = max(noise_scale, construction_floor)
                noise = -math.log(-math.log(u))*noise_scale*0.08
                distance_bias = 0.04*graph_distance(coords, site, target) if target >= 0 else 0.0
                score = scores[parent] + pair_reward*contacts + distance_bias - noise
                if routing:
                    score -= routing_weight*math.log1p(continuations)
                expansions[count, :] = paths[parent, :]
                expansions[count, depth] = site
                costs[count], cs[count], ps[count] = score, signature, psig
                expanded_hh[count] = local_hh[parent] + contacts
                count += 1
        if count == 0:
            return paths, 0, expanded_total, pruned_total, 0, local_hh
        order = np.argsort(costs[:count])
        picked = np.empty(width, np.int32)
        used = np.zeros(count, np.bool_)
        take = 0
        # Reserve half for score quality, half for contact/P-route/approach diversity.
        quality_quota = max(1, width//2) if diversity else width
        for k in range(min(quality_quota, count)):
            idx = order[k]
            picked[take], used[idx] = idx, True
            take += 1
        if diversity:
            for idx in order:
                if used[idx] or take >= width:
                    continue
                novel = True
                for p in range(take):
                    j = picked[p]
                    if cs[idx] == cs[j] and ps[idx] == ps[j] and expansions[idx, 0] == expansions[j, 0] and expansions[idx, depth] == expansions[j, depth]:
                        novel = False
                        break
                if novel:
                    picked[take], used[idx] = idx, True
                    take += 1
            for idx in order:
                if take >= width:
                    break
                if not used[idx]:
                    picked[take] = idx
                    take += 1
        beam_n = take
        for k in range(take):
            idx = picked[k]
            paths[k, :] = expansions[idx, :]
            scores[k], contact_sig[k], polar_sig[k] = costs[idx], cs[idx], ps[idx]
            local_hh[k] = expanded_hh[idx]
    unique = 0
    for k in range(beam_n):
        novel = True
        for j in range(k):
            if contact_sig[k] == contact_sig[j] and polar_sig[k] == polar_sig[j] and paths[k, 0] == paths[j, 0] and paths[k, m-1] == paths[j, m-1]:
                novel = False
                break
        unique += int(novel)
    if not forward:
        for k in range(beam_n):
            for d in range(m//2):
                v = paths[k, d]
                paths[k, d] = paths[k, m-1-d]
                paths[k, m-1-d] = v
    return paths, beam_n, expanded_total, pruned_total, unique, local_hh


@njit(cache=True)
def beam_candidates(pos, a, b, hmask, coords, neighbors, occ, pair_reward, temperature,
                    width, routing, diversity, construction_floor, routing_weight):
    paths,count,expanded,pruned,unique,_ = beam_candidates_scored(
        pos,a,b,hmask,coords,neighbors,occ,pair_reward,temperature,width,routing,diversity,
        construction_floor,routing_weight,False)
    return paths,count,expanded,pruned,unique


@njit(cache=True)
def block_contact_count(pos, a, b, hmask, neighbors, occ):
    result = 0
    for i in range(a,b):
        if hmask[i]:
            for v in neighbors[pos[i]]:
                if v >= 0:
                    j = occ[v]
                    if j >= 0 and hmask[j] and abs(i-j)>1 and (j<a or j>=b or i<j):
                        result += 1
    return result


@njit(cache=True)
def anneal_chunk(pos, best, hh, excess, degrees, best_hh, best_excess, hmask, coords, neighbors,
                 threshold, pair_reward, cluster_reward, min_len, max_len, beam_width,
                 t0, t1, cycle_length, offset, steps, stop_hh, routing, diversity,
                 construction_floor, routing_weight, bidirectional=False, short_beam_width=0):
    occ = occupancy(pos, len(neighbors))
    # steps, valid pools, candidates, unique signatures, transitions, expansions,
    # geometry prunes, duplicate complete assignments.
    stats = np.zeros(9, np.int64)
    milestones = np.full((steps, 2), -1, np.int64)
    events = 0
    for step in range(steps):
        temperature = t0*math.exp(math.log(t1/t0)*(offset+step)/max(1, cycle_length-1))
        cap = min(max_len, len(pos)-1)
        high = min(cap, max(min_len, 8)) if np.random.random() < 0.70 else cap
        length = np.random.randint(min_len, high+1)
        a = np.random.randint(0, len(pos)-length+1)
        b = a+length
        reverse = bidirectional and a>0 and b<len(pos) and np.random.random()<0.5
        width = min(beam_width,short_beam_width) if length<=8 and short_beam_width>0 else beam_width
        paths, count, expanded, pruned, unique, local_counts = beam_candidates_scored(
            pos, a, b, hmask, coords, neighbors, occ, pair_reward, temperature,
            width, routing, diversity, construction_floor, routing_weight, reverse)
        stats[0] += 1
        stats[5] += expanded
        stats[6] += pruned
        if not count:
            continue
        stats[1] += 1
        stats[8] += width
        stats[2] += count
        stats[3] += unique
        scores = np.full(count+1, np.inf)
        counts = np.zeros(count+1, np.int32)
        excesses = np.zeros(count+1, np.int32)
        fixed_hh = hh-block_contact_count(pos,a,b,hmask,neighbors,occ)
        scores[0], counts[0], excesses[0] = pair_reward*hh+cluster_reward*excess, hh, excess
        for k in range(count):
            duplicate = False
            if diversity:
                duplicate = True
                for d in range(length):
                    if paths[k, d] != pos[a+d]:
                        duplicate = False
                        break
                for j in range(k):
                    same = True
                    for d in range(length):
                        if paths[k, d] != paths[j, d]:
                            same = False
                            break
                    duplicate = duplicate or same
            if duplicate:
                stats[7] += 1
                continue
            ch, ce = fixed_hh+local_counts[k], 0
            if cluster_reward != 0.0:
                _,ce,_ = local_delta(pos,paths[k],a,b,hmask,neighbors,occ,hh,excess,degrees,threshold)
            counts[k+1], excesses[k+1] = ch, ce
            scores[k+1] = pair_reward*ch + cluster_reward*ce
        weights = np.exp(-(scores-np.min(scores))/max(temperature, 1.e-12))
        pick = np.random.random()*np.sum(weights)
        chosen, cumulative = 0, 0.0
        for j in range(count+1):
            cumulative += weights[j]
            if cumulative > pick:
                chosen = j
                break
        if chosen:
            for i in range(a, b):
                occ[pos[i]] = -1
            pos[a:b] = paths[chosen-1]
            for i in range(a, b):
                occ[pos[i]] = i
            hh, excess, degrees = full_metrics(pos,hmask,neighbors,threshold)
            stats[4] += 1
            if hh > best_hh or (hh == best_hh and excess > best_excess):
                if hh > best_hh:
                    milestones[events, 0], milestones[events, 1] = step+1, hh
                    events += 1
                best_hh, best_excess = hh, excess
                best[:] = pos
                if best_hh >= stop_hh:
                    break
    return hh, excess, degrees, best_hh, best_excess, stats, milestones[:events]
