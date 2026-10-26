"""Compiled feasible FCC block proposals and exact local contact deltas.

Routing scores affect construction only. Completed folds use the unchanged
pairwise-plus-decoded-shaping energy and a direct finite-pool heat-bath update.
"""
import math
import time

import numpy as np
from numba import njit, objmode


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
def boundary_choice_counts(pos, a, b, coords, neighbors, occ, c=0, d=0):
    """Cheap optimistic branching counts; a preference, NEVER a feasibility cut."""
    counts = np.zeros(2, np.int32)
    for side in range(2):
        anchor = pos[a-1] if side == 0 else pos[b]
        target = pos[b] if side == 0 else pos[a-1]
        for site in neighbors[anchor]:
            if site < 0:
                continue
            owner = occ[site]
            if owner >= 0 and not (a <= owner < b or c <= owner < d):
                continue
            if graph_distance(coords, site, target) <= b-a:
                counts[side] += 1
    return counts


@njit(cache=True)
def reconstruction_reverse(pos, a, b, coords, neighbors, occ, policy, c=0, d=0):
    if a == 0 or b == len(pos):
        return False
    if policy == 0:
        return np.random.random() < .5
    counts = boundary_choice_counts(pos,a,b,coords,neighbors,occ,c,d)
    if counts[0] == counts[1]:
        return np.random.random() < .5
    preferred = counts[1] < counts[0]
    # Explore the other orientation on 25% of proposals, including tight boxes.
    return preferred if np.random.random() < .75 else not preferred


@njit(cache=True)
def mix_id(value):
    x = np.uint64(value + 1)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def beam_candidates_scored(pos, a, b, hmask, coords, neighbors, occ, pair_reward, temperature,
                    width, routing, diversity, construction_floor, routing_weight, reverse=False,
                    released_a=0, released_b=0, expansion_budget=0, preference_scale=1.0):
    m, n = b-a, len(pos)
    fixed = occ.copy()
    for i in range(a, b):
        if fixed[pos[i]] == i:
            fixed[pos[i]] = -1
    for i in range(released_a, released_b):
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
                if expansion_budget > 0 and expanded_total >= expansion_budget:
                    return paths, 0, expanded_total, pruned_total, 0, local_hh
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
                score = scores[parent] + preference_scale*(pair_reward*contacts + distance_bias) - noise
                if routing:
                    score -= preference_scale*routing_weight*math.log1p(continuations)
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
def choose_coupled_intervals(pos, occ, neighbors, max_len):
    """Uniform first interval; uniform nearby disjoint second interval.

    Keep a fixed residue between intervals so that all four boundaries remain
    fixed. Total movable residues respect the current stage's block capacity.
    """
    cap = min(12, max_len // 2)
    if cap < 6 or len(pos) < 13:
        return -1, -1, -1, -1
    m1 = np.random.randint(6, cap+1)
    m2 = np.random.randint(6, cap+1)
    return choose_coupled_intervals_with_lengths(pos, occ, neighbors, m1, m2)


@njit(cache=True)
def choose_coupled_intervals_with_lengths(pos, occ, neighbors, m1, m2):
    a = np.random.randint(0, len(pos)-m1+1)
    b = a+m1
    nearby = np.zeros(len(pos), np.bool_)
    for i in range(a, b):
        for v in neighbors[pos[i]]:
            if v < 0:
                continue
            j = occ[v]
            if j >= 0:
                nearby[j] = True
            for u in neighbors[v]:
                if u >= 0 and occ[u] >= 0:
                    nearby[occ[u]] = True
    choices = np.empty(len(pos), np.int32)
    count = 0
    for c in range(len(pos)-m2+1):
        d = c+m2
        if not (b < c or d < a):
            continue
        if np.any(nearby[c:d]):
            choices[count] = c
            count += 1
    if not count:
        return -1, -1, -1, -1
    c = choices[np.random.randint(count)]
    return a, b, c, c+m2


@njit(cache=True)
def vacancy_exposure(site, occ, neighbors, vacancy_mask):
    count = 0
    if vacancy_mask is not None:
        for v in neighbors[site]:
            if v >= 0 and vacancy_mask[v] and occ[v] < 0:
                count += 1
    return count


@njit(cache=True)
def guided_weights(pos, occ, neighbors, hmask, degrees, vacancy_mask=None, vacancy_boost=0.0):
    weights = np.zeros(len(pos), np.float64)
    for i in range(len(pos)):
        if hmask[i]:
            weights[i] = 1.0/(1.0+degrees[i])**2
            if vacancy_boost > 0 and vacancy_mask is not None:
                exposure = vacancy_exposure(pos[i], occ, neighbors, vacancy_mask)
                weights[i] *= 1.0+vacancy_boost*min(exposure, 2)/2.0
    return weights


@njit(cache=True)
def choose_guided_intervals(pos, occ, neighbors, hmask, degrees, max_len, attempts=4,
                            vacancy_mask=None, vacancy_boost=0.0):
    """Bias interval selection only; selected missing contacts are not constraints."""
    cap = min(12, max_len//2)
    if cap < 6 or len(pos) < 13:
        return -1, -1, -1, -1, -1, -1, False
    m1, m2 = np.random.randint(6, cap+1), np.random.randint(6, cap+1)
    n = len(pos)
    weights = guided_weights(pos, occ, neighbors, hmask, degrees, vacancy_mask, vacancy_boost)
    total = np.sum(weights)
    seen = np.zeros(n, np.bool_)
    partners = np.empty(n, np.int32)
    for _ in range(attempts):
        if total <= 0:
            break
        pick, cumulative, i = np.random.random()*total, 0.0, n-1
        for index in range(n):
            cumulative += weights[index]
            if cumulative > pick:
                i = index
                break
        amin, amax = max(0, i-m1+1), min(i, n-m1)
        seen[:] = False
        seen[i] = True
        for site in neighbors[pos[i]]:
            if site >= 0 and occ[site] >= 0:
                seen[occ[site]] = True  # Exclude existing H-H neighbors.
        count, partner_weight = 0, 0.0
        for site in neighbors[pos[i]]:
            if site < 0:
                continue
            for other in neighbors[site]:
                j = occ[other] if other >= 0 else -1
                if j < 0 or seen[j]:
                    continue
                seen[j] = True
                if not hmask[j] or abs(i-j) <= 1:
                    continue
                cmin, cmax = max(0, j-m2+1), min(j, n-m2)
                if not (amin+m1 < cmax or cmin+m2 < amax):
                    continue
                partners[count] = j
                count += 1
                partner_weight += weights[j]
        if not count:
            continue
        pick, cumulative, j = np.random.random()*partner_weight, 0.0, partners[count-1]
        for k in range(count):
            cumulative += weights[partners[k]]
            if cumulative > pick:
                j = partners[k]
                break
        starts = np.empty((m1*m2, 2), np.int32)
        start_weights = np.empty(m1*m2, np.float64)
        count, total_start_weight = 0, 0.0
        for a in range(amin, amax+1):
            for c in range(max(0, j-m2+1), min(j, n-m2)+1):
                if not (a+m1 < c or c+m2 < a):
                    continue
                polar = 0
                for r in range(a, a+m1):
                    polar += int(not hmask[r])
                for r in range(c, c+m2):
                    polar += int(not hmask[r])
                weight = 1.0+polar/(m1+m2)
                starts[count, 0], starts[count, 1] = a, c
                start_weights[count] = weight
                total_start_weight += weight
                count += 1
        pick, cumulative, chosen = np.random.random()*total_start_weight, 0.0, count-1
        for k in range(count):
            cumulative += start_weights[k]
            if cumulative > pick:
                chosen = k
                break
        a, c = starts[chosen]
        return a, a+m1, c, c+m2, i, j, True
    # Preserve the initially drawn lengths even when no eligible H pair exists.
    a, b, c, d = choose_coupled_intervals_with_lengths(pos, occ, neighbors, m1, m2)
    return a, b, c, d, -1, -1, False


@njit(cache=True)
def p_run_windows_disjoint(windows, u, v):
    return (windows[u, 2] != windows[v, 2] and
            (windows[u, 1] < windows[v, 0] or windows[v, 1] < windows[u, 0]))


@njit(cache=True)
def p_run_pair_possible(windows, offsets, members, i, j):
    for u in range(offsets[i], offsets[i+1]):
        for v in range(offsets[j], offsets[j+1]):
            if p_run_windows_disjoint(windows, members[u], members[v]):
                return True
    return False


@njit(cache=True)
def choose_p_run_intervals(pos, occ, neighbors, hmask, degrees, windows, offsets, members,
                           attempts=4, vacancy_mask=None, vacancy_boost=0.0):
    """Two sequence-derived windows around nearby, deficient movable H flanks.

    Selection only: no forced contact, changed reconstruction or score bonus.
    Caller explicitly falls back to the inherited selector on failure.
    """
    if len(windows) < 2:
        return -1, -1, -1, -1, -1, -1, False
    weights = guided_weights(pos, occ, neighbors, hmask, degrees, vacancy_mask, vacancy_boost)
    for i in range(len(pos)):
        if offsets[i] == offsets[i+1]:
            weights[i] = 0.0
    total = np.sum(weights)
    seen = np.zeros(len(pos), np.bool_)
    partners = np.empty(len(pos), np.int32)
    for _ in range(attempts):
        if total <= 0:
            break
        pick, cumulative, i = np.random.random()*total, 0.0, len(pos)-1
        for index in range(len(pos)):
            cumulative += weights[index]
            if cumulative > pick:
                i = index
                break
        seen[:] = False
        seen[i] = True
        for site in neighbors[pos[i]]:
            if site >= 0 and occ[site] >= 0:
                seen[occ[site]] = True
        count, partner_weight = 0, 0.0
        for site in neighbors[pos[i]]:
            if site < 0:
                continue
            for other in neighbors[site]:
                j = occ[other] if other >= 0 else -1
                if j < 0 or seen[j]:
                    continue
                seen[j] = True
                if weights[j] <= 0 or abs(i-j) <= 1:
                    continue
                if not p_run_pair_possible(windows, offsets, members, i, j):
                    continue
                partners[count] = j
                count += 1
                partner_weight += weights[j]
        if not count:
            continue
        pick, cumulative, j = np.random.random()*partner_weight, 0.0, partners[count-1]
        for z in range(count):
            cumulative += weights[partners[z]]
            if cumulative > pick:
                j = partners[z]
                break
        pairs = np.empty(((offsets[i+1]-offsets[i])*(offsets[j+1]-offsets[j]), 2), np.int32)
        count = 0
        for ui in range(offsets[i], offsets[i+1]):
            for vi in range(offsets[j], offsets[j+1]):
                u, v = members[ui], members[vi]
                if p_run_windows_disjoint(windows, u, v):
                    pairs[count, 0], pairs[count, 1] = u, v
                    count += 1
        u, v = pairs[np.random.randint(count)]
        return windows[u, 0], windows[u, 1], windows[v, 0], windows[v, 1], i, j, True
    return -1, -1, -1, -1, -1, -1, False


@njit(cache=True)
def contact_changes(old, new, hmask, neighbors):
    created, broken = 0, 0
    for i in range(len(old)):
        if not hmask[i]:
            continue
        for j in range(i+2, len(old)):
            if hmask[j]:
                before = np.any(neighbors[old[i]] == old[j])
                after = np.any(neighbors[new[i]] == new[j])
                created += int(after and not before)
                broken += int(before and not after)
    return created, broken


@njit(cache=True)
def contact_map_fingerprint(pos, hmask, neighbors):
    occ = occupancy(pos, len(neighbors))
    signature = np.uint64(0)
    for i in range(len(pos)):
        if hmask[i]:
            for site in neighbors[pos[i]]:
                j = occ[site] if site >= 0 else -1
                if j > i+1 and hmask[j]:
                    signature ^= mix_id(np.uint64(i*len(pos)+j+1))
    return signature


@njit(cache=True)
def sequential_coupled_candidates(pos, a, b, c, d, hmask, coords, neighbors, occ,
                       pair_reward, temperature, width, parent_limit,
                       expansion_budget, routing, diversity, construction_floor,
                       routing_weight, bidirectional):
    """Bounded joint reconstruction, not independent accepted block updates.

    Both old intervals are released before first-block growth. Several first
    paths are tried without an intermediate energy test; only complete, mutually
    compatible two-block assignments are returned. This is heuristic, not exact.
    """
    size = b-a+d-c
    output = np.full((width*parent_limit, size), -1, np.int32)
    reverse = bidirectional and a > 0 and b < len(pos) and np.random.random() < .5
    first, count, expanded, pruned, _, _ = beam_candidates_scored(
        pos, a, b, hmask, coords, neighbors, occ, pair_reward, temperature,
        width, routing, diversity, construction_floor, routing_weight, reverse,
        c, d, expansion_budget)
    if not count:
        return output, 0, expanded, pruned, 0
    # Random parent order avoids always spending the budget on the first prefix.
    order = np.arange(count)
    for k in range(count-1, 0, -1):
        j = np.random.randint(k+1)
        order[k], order[j] = order[j], order[k]
    base_occ = occ.copy()
    for i in range(a, b):
        base_occ[pos[i]] = -1
    for i in range(c, d):
        base_occ[pos[i]] = -1
    completed = 0
    for p in range(min(count, parent_limit)):
        if expanded >= expansion_budget:
            break
        path = first[order[p]]
        temporary = pos.copy()
        temporary[a:b] = path
        partial_occ = base_occ.copy()
        for i in range(a, b):
            partial_occ[path[i-a]] = i
        # Old second-block sites may now belong to the first reconstructed block.
        # The constructor must not clear them a second time.
        reverse = bidirectional and c > 0 and d < len(pos) and np.random.random() < .5
        second, nsecond, e, q, _, _ = beam_candidates_scored(
            temporary, c, d, hmask, coords, neighbors, partial_occ,
            pair_reward, temperature, width, routing, diversity,
            construction_floor, routing_weight, reverse, 0, 0, expansion_budget-expanded)
        expanded += e
        pruned += q
        for k in range(nsecond):
            candidate = np.empty(size, np.int32)
            candidate[:b-a] = path
            candidate[b-a:] = second[k]
            duplicate = False
            for j in range(completed):
                if np.all(candidate == output[j]):
                    duplicate = True
                    break
            if not duplicate:
                output[completed] = candidate
                completed += 1
    return output, completed, expanded, pruned, completed


@njit(cache=True)
def paired_signature_novel(idx, picked, take, cs, ps, paths, tips):
    for p in range(take):
        j = picked[p]
        same_tips = True
        for t in range(len(tips)):
            if tips[t] >= 0 and paths[idx,tips[t]] != paths[j,tips[t]]:
                same_tips = False
        if cs[idx] == cs[j] and ps[idx] == ps[j] and same_tips:
            return False
    return True


@njit(cache=True)
def stratified_paired_selection(costs, contacts, cs, ps, paths, tips, count,
                               width, diversity):
    """Reserve 1/2, 1/3, 1/6 of the beam for contact deficits 0, 1, 2.

    Counts refer to the same placed-residue set at a common growth depth,
    excluding fixed-fixed contacts. No extra RNG draws or target information.
    Within each quota prefer novel signatures, then fill by construction score.
    Unused quota is redistributed across all remaining candidates by that score,
    with novel signatures first. This may retain deficits larger than two.
    """
    order = np.argsort(costs[:count])
    best = np.max(contacts[:count])
    quotas = np.array([max(1,width//2), width//3, 0], np.int32)
    quotas[2] = width-quotas[0]-quotas[1]
    picked = np.empty(width, np.int32)
    used = np.zeros(count, np.bool_)
    take = 0
    for gap in range(3):
        allocated = 0
        for diversity_pass in range(2):
            if diversity_pass == 0 and not diversity:
                continue
            for idx in order:
                if allocated >= quotas[gap] or take >= width:
                    break
                if used[idx] or best-contacts[idx] != gap:
                    continue
                if diversity_pass == 0 and not paired_signature_novel(idx,picked,take,cs,ps,paths,tips):
                    continue
                picked[take], used[idx] = idx, True
                take += 1
                allocated += 1
    for diversity_pass in range(2):
        if diversity_pass == 0 and not diversity:
            continue
        for idx in order:
            if take >= width:
                break
            if used[idx]:
                continue
            if diversity_pass == 0 and not paired_signature_novel(idx,picked,take,cs,ps,paths,tips):
                continue
            picked[take], used[idx] = idx, True
            take += 1
    return picked, take


@njit(cache=True)
def coupled_candidates(pos, a, b, c, d, hmask, coords, neighbors, occ,
                       pair_reward, temperature, width, parent_limit,
                       expansion_budget, routing, diversity, construction_floor,
                       routing_weight, bidirectional, stratified=True,
                       selection_stats=None, partial_contacts_out=None,
                       boundary_policy=0, preference_scale=1.0,
                       forced_i=-1, forced_j=-1, forced_prunes=None):
    assert parent_limit == 1
    return joint_candidates(pos, np.array([a,c],np.int32), np.array([b,d],np.int32),
        hmask,coords,neighbors,occ,pair_reward,temperature,width,expansion_budget,
        routing,diversity,construction_floor,routing_weight,bidirectional,
        stratified,selection_stats,partial_contacts_out,boundary_policy,preference_scale,
        forced_i,forced_j,forced_prunes)


@njit(cache=True)
def contact_completion_possible(path, slot, site, pair_slots, pair_blocks, pair_ranks,
                                placed, tips, anchors, targets, lengths, distances,
                                fixed, coords, neighbors, growing_block):
    locations = np.empty(2, np.int32)
    for z in range(2):
        locations[z] = site if pair_slots[z] == slot else path[pair_slots[z]]
    if locations[0] >= 0 and locations[1] >= 0:
        return graph_distance(coords, locations[0], locations[1]) == 1
    if locations[0] < 0 and locations[1] < 0:
        return True
    assigned = 0 if locations[0] >= 0 else 1
    pending = 1-assigned
    block = pair_blocks[pending]
    completed = placed[block] + int(block == growing_block)
    tip = (site if block == growing_block else
           (path[tips[block]] if completed else anchors[block]))
    to_h = pair_ranks[pending]+1-completed
    to_boundary = lengths[block]-pair_ranks[pending]
    # Optimistic necessary bounds, not a proof of a valid completion.
    for v in neighbors[locations[assigned]]:
        if v < 0 or fixed[v] >= 0 or v == site or contains(path, len(path), v):
            continue
        if tip >= 0 and graph_distance(coords, tip, v) > to_h:
            continue
        target = targets[block]
        if target >= 0:
            if graph_distance(coords, v, target) > to_boundary:
                continue
            if distances[block, v] < 0 or distances[block, v] > to_boundary:
                continue
        return True
    return False


@njit(cache=True)
def joint_candidates(pos, starts, ends, hmask, coords, neighbors, occ,
                     pair_reward, temperature, width, expansion_budget,
                     routing, diversity, construction_floor, routing_weight,
                     bidirectional=False, stratified=True, selection_stats=None,
                     partial_contacts_out=None, boundary_policy=0, preference_scale=1.0,
                     forced_i=-1, forced_j=-1, forced_prunes=None):
    """One joint beam, alternating intervals; no first-block parent bottleneck.

    Paths are stored in sequence order, including unfilled (-1) slots. Each
    extension sees fixed residues and both partial intervals. Every physical
    contact involving movable residues is scored when its later endpoint is
    placed. The shared expansion bound counts feasible-site examinations.
    parent_limit is retained in the call signature, but must be one here.
    The stratified branch changes retention only, not expansion or acceptance.
    """
    nblocks = len(starts)
    assert nblocks in (2,3,4) and len(ends)==nblocks
    lengths = ends-starts
    bases = np.zeros(nblocks,np.int32)
    for q in range(nblocks):
        assert 0 <= starts[q] < ends[q] <= len(pos)
        if q:
            bases[q] = bases[q-1]+lengths[q-1]
        for r in range(q):
            assert ends[q] < starts[r] or ends[r] < starts[q]
    size = np.sum(lengths)
    fixed = occ.copy()
    for q in range(nblocks):
        for i in range(starts[q],ends[q]):
            fixed[pos[i]] = -1
    forward = np.empty(nblocks, np.bool_)
    anchors = np.empty(nblocks, np.int32)
    targets = np.empty(nblocks, np.int32)
    distances = np.full((nblocks, len(neighbors)), -1, np.int32)
    for q in range(nblocks):
        reverse = (bidirectional and reconstruction_reverse(pos,starts[q],ends[q],
                   coords,neighbors,fixed,boundary_policy))
        forward[q] = starts[q] > 0 and (ends[q] == len(pos) or not reverse)
        anchors[q] = pos[starts[q]-1] if forward[q] else pos[ends[q]]
        targets[q] = ((pos[ends[q]] if ends[q] < len(pos) else -1) if forward[q]
                      else (pos[starts[q]-1] if starts[q] > 0 else -1))
        if (routing or forced_i >= 0) and targets[q] >= 0:
            distances[q] = obstacle_distances(fixed, targets[q], neighbors, lengths[q]+1)
    pair_slots = np.full(2, -1, np.int32)
    pair_blocks = np.full(2, -1, np.int32)
    pair_ranks = np.full(2, -1, np.int32)
    if forced_i >= 0:
        assert 0 <= forced_j < len(pos) and forced_i < len(pos)
        assert hmask[forced_i] and hmask[forced_j] and abs(forced_i-forced_j) > 1
        assert graph_distance(coords, pos[forced_i], pos[forced_j]) != 1
        for z, residue in enumerate((forced_i, forced_j)):
            for q in range(nblocks):
                if starts[q] <= residue < ends[q]:
                    relative = residue-starts[q]
                    pair_slots[z], pair_blocks[z] = bases[q]+relative, q
                    pair_ranks[z] = relative if forward[q] else lengths[q]-1-relative
        assert pair_blocks[0] >= 0 and pair_blocks[1] >= 0
        assert pair_blocks[0] != pair_blocks[1]
    residues = np.empty(size, np.int32)
    for q in range(nblocks):
        residues[bases[q]:bases[q]+lengths[q]] = np.arange(starts[q],ends[q])
    paths = np.full((width, size), -1, np.int32)
    scores = np.zeros(width)
    partial_hh = np.zeros(width, np.int32)
    contact_sig = np.zeros(width, np.uint64)
    polar_sig = np.zeros(width, np.uint64)
    capacity = width*12
    expanded_paths = np.full((capacity, size), -1, np.int32)
    costs = np.empty(capacity)
    expanded_hh = np.empty(capacity, np.int32)
    cs = np.empty(capacity, np.uint64)
    ps = np.empty(capacity, np.uint64)
    fixed_counts = np.empty(len(neighbors), np.int32)
    fixed_signatures = np.empty(len(neighbors), np.uint64)
    placed = np.zeros(nblocks, np.int32)
    tips = np.full(nblocks, -1, np.int32)
    beam_n, expansions, pruned = 1, 0, 0
    for depth in range(size):
        q = depth % nblocks
        while placed[q] == lengths[q]:
            q = (q+1) % nblocks
        relative = placed[q] if forward[q] else lengths[q]-1-placed[q]
        slot = bases[q]+relative
        residue = residues[slot]
        remaining = lengths[q]-placed[q]-1
        target = targets[q]
        fixed_counts[:] = -1
        count = 0
        for parent in range(beam_n):
            previous = anchors[q] if placed[q] == 0 else paths[parent, tips[q]]
            for site in neighbors[previous]:
                if site < 0 or fixed[site] >= 0 or contains(paths[parent], size, site):
                    continue
                if expansions >= expansion_budget:
                    # Never expose incomplete paths to the heat-bath pool.
                    return paths, 0, expansions, pruned, 0
                expansions += 1
                if target >= 0:
                    distance = graph_distance(coords, site, target)
                    if distance > remaining+1 or (remaining == 0 and distance != 1):
                        pruned += 1
                        continue
                    if routing and (distances[q, site] < 0 or distances[q, site] > remaining+1):
                        pruned += 1
                        continue
                continuations = 0
                if routing and remaining > 0:
                    for v in neighbors[site]:
                        if v < 0 or fixed[v] >= 0 or contains(paths[parent], size, v):
                            continue
                        if target < 0 or (0 <= distances[q, v] <= remaining):
                            continuations += 1
                    if continuations == 0:
                        pruned += 1
                        continue
                if forced_i >= 0 and not contact_completion_possible(
                        paths[parent],slot,site,pair_slots,pair_blocks,pair_ranks,
                        placed,tips,anchors,targets,lengths,distances,fixed,
                        coords,neighbors,q):
                    pruned += 1
                    if forced_prunes is not None:
                        forced_prunes[0] += 1
                    continue
                contacts = 0
                signature = contact_sig[parent]
                if hmask[residue]:
                    if fixed_counts[site] < 0:
                        field, sig = 0, np.uint64(0)
                        for v in neighbors[site]:
                            if v >= 0:
                                j = fixed[v]
                                if j >= 0 and hmask[j] and abs(j-residue) > 1:
                                    field += 1
                                    sig ^= mix_id(min(j,residue)*len(pos)+max(j,residue))
                        fixed_counts[site], fixed_signatures[site] = field, sig
                    contacts = fixed_counts[site]
                    signature ^= fixed_signatures[site]
                    for z in range(size):
                        j = residues[z]
                        if (paths[parent,z] >= 0 and hmask[j] and abs(j-residue)>1
                                and graph_distance(coords,site,paths[parent,z]) == 1):
                            contacts += 1
                            signature ^= mix_id(min(j,residue)*len(pos)+max(j,residue))
                psig = polar_sig[parent]
                if not hmask[residue]:
                    psig ^= mix_id(site)
                u = max(1.e-12, min(1.-1.e-12, np.random.random()))
                noise_scale = max(.03, temperature)
                if diversity:
                    noise_scale = max(noise_scale, construction_floor)
                noise = -math.log(-math.log(u))*noise_scale*.08
                bias = .04*graph_distance(coords,site,target) if target >= 0 else 0.
                cost = scores[parent]+preference_scale*(pair_reward*contacts+bias)-noise
                if routing:
                    cost -= preference_scale*routing_weight*math.log1p(continuations)
                expanded_paths[count] = paths[parent]
                expanded_paths[count,slot] = site
                costs[count], cs[count], ps[count] = cost, signature, psig
                expanded_hh[count] = partial_hh[parent]+contacts
                count += 1
        if count == 0:
            return paths, 0, expansions, pruned, 0
        tips[q] = slot
        placed[q] += 1
        if stratified:
            picked, take = stratified_paired_selection(costs,expanded_hh,cs,ps,
                expanded_paths,tips,count,width,diversity)
        else:
            # Retained baseline branch permits fixed-proposal regression tests.
            order = np.argsort(costs[:count])
            picked = np.empty(width, np.int32)
            used = np.zeros(count, np.bool_)
            take = min(max(1,width//2) if diversity else width, count)
            for k in range(take):
                picked[k], used[order[k]] = order[k], True
            if diversity:
                for idx in order:
                    if used[idx] or take >= width:
                        continue
                    if paired_signature_novel(idx,picked,take,cs,ps,expanded_paths,tips):
                        picked[take], used[idx] = idx, True
                        take += 1
                for idx in order:
                    if take >= width:
                        break
                    if not used[idx]:
                        picked[take] = idx
                        take += 1
        if selection_stats is not None:
            top_partial = np.max(expanded_hh[:count])
            selection_stats[0] += 1
            selection_stats[5] += np.sum(expanded_hh[:count] == top_partial-1)
            selection_stats[6] += np.sum(expanded_hh[:count] == top_partial-2)
            for k in range(take):
                gap = min(3, top_partial-expanded_hh[picked[k]])
                selection_stats[1+gap] += 1
        beam_n = take
        for k in range(take):
            idx = picked[k]
            paths[k] = expanded_paths[idx]
            scores[k], contact_sig[k], polar_sig[k] = costs[idx], cs[idx], ps[idx]
            partial_hh[k] = expanded_hh[idx]
    if partial_contacts_out is not None:
        partial_contacts_out[:beam_n] = partial_hh[:beam_n]
    return paths, beam_n, expansions, pruned, beam_n


@njit(cache=True)
def coupled_delta(pos, block, a, b, c, d, hmask, neighbors, occ,
                  hh, excess, degrees, threshold):
    """Exact contact/shaping delta over the union; cross-block contacts count once."""
    indices = np.empty(b-a+d-c, np.int32)
    indices[:b-a] = np.arange(a, b)
    indices[b-a:] = np.arange(c, d)
    changes = np.zeros(len(pos), np.int32)
    delta = 0
    for i in indices:
        if not hmask[i]:
            continue
        for site in neighbors[pos[i]]:
            if site < 0:
                continue
            j = occ[site]
            if j >= 0 and hmask[j] and abs(i-j) > 1:
                if not (a <= j < b or c <= j < d) or i < j:
                    delta -= 1
                    changes[i] -= 1
                    changes[j] -= 1
    for i in indices:
        occ[pos[i]] = -1
    for k in range(len(indices)):
        occ[block[k]] = indices[k]
    for k in range(len(indices)):
        i = indices[k]
        if not hmask[i]:
            continue
        for site in neighbors[block[k]]:
            if site < 0:
                continue
            j = occ[site]
            if j >= 0 and hmask[j] and abs(i-j) > 1:
                if not (a <= j < b or c <= j < d) or i < j:
                    delta += 1
                    changes[i] += 1
                    changes[j] += 1
    for site in block:
        occ[site] = -1
    for i in indices:
        occ[pos[i]] = i
    for i in range(len(pos)):
        if hmask[i] and changes[i]:
            excess += max(0, degrees[i]+changes[i]-threshold)-max(0, degrees[i]-threshold)
    return hh+delta, excess, changes


@njit(cache=True)
def choose_three_intervals(pos, occ, neighbors, hmask, degrees, attempts=4,
                           vacancy_mask=None, vacancy_boost=0.0):
    """Two guided 6-8 intervals plus a 6-8 interval around a blocking H or P.

    A blocker occupies a common neighbor of the selected noncontacting H pair.
    Intervals are disjoint and every boundary remains fixed. No adjacency
    constraint is imposed on completed candidates. Failed selection is a null
    proposal, not a silently relabeled two-block proposal.
    """
    options = np.empty((96,3),np.int32)
    for _ in range(attempts):
        a,b,c,d,i,j,guided = choose_guided_intervals(
            pos,occ,neighbors,hmask,degrees,16,1,vacancy_mask,vacancy_boost)
        if not guided:
            continue
        length = np.random.randint(6,9)
        count = 0
        for site in neighbors[pos[i]]:
            if site < 0 or occ[site] < 0:
                continue
            common = False
            for v in neighbors[pos[j]]:
                if v == site:
                    common = True
                    break
            if not common:
                continue
            blocker = occ[site]
            for e in range(max(0,blocker-length+1),min(blocker,len(pos)-length)+1):
                f = e+length
                if (b < e or f < a) and (d < e or f < c):
                    options[count] = np.array([e,f,blocker],np.int32)
                    count += 1
        if count:
            e,f,blocker = options[np.random.randint(count)]
            return a,b,c,d,e,f,i,j,blocker
    return -1,-1,-1,-1,-1,-1,-1,-1,-1


@njit(cache=True)
def choose_four_intervals(pos, occ, neighbors, hmask, degrees, attempts=4,
                          vacancy_mask=None, vacancy_boost=0.0):
    """Extend J3 with a disjoint short interval around a second routing blocker.

    Eligible sites are other common neighbors of the selected H pair, or
    occupied neighbors of the first blocker. Fixed boundaries of all four
    intervals must remain outside the released union. No contact is forced.
    """
    options = np.empty((192,3),np.int32)
    for _ in range(attempts):
        a,b,c,d,e,f,i,j,blocker = choose_three_intervals(
            pos,occ,neighbors,hmask,degrees,1,vacancy_mask,vacancy_boost)
        if a < 0:
            continue
        sites = np.empty(24,np.int32)
        site_count = 0
        for site in neighbors[pos[i]]:
            if site < 0 or occ[site] < 0:
                continue
            for v in neighbors[pos[j]]:
                if v == site:
                    sites[site_count] = site
                    site_count += 1
                    break
        for site in neighbors[pos[blocker]]:
            if site < 0 or occ[site] < 0:
                continue
            duplicate = False
            for z in range(site_count):
                duplicate = duplicate or sites[z] == site
            if not duplicate:
                sites[site_count] = site
                site_count += 1
        length = np.random.randint(6,9)
        count = 0
        for z in range(site_count):
            second_blocker = occ[sites[z]]
            for g in range(max(0,second_blocker-length+1),min(second_blocker,len(pos)-length)+1):
                h = g+length
                if ((b < g or h < a) and (d < g or h < c) and (f < g or h < e)):
                    options[count] = np.array([g,h,second_blocker],np.int32)
                    count += 1
        if count:
            g,h,second_blocker = options[np.random.randint(count)]
            return a,b,c,d,e,f,g,h,i,j,blocker,second_blocker
    return -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1


@njit(cache=True)
def joint_delta(pos, block, indices, hmask, neighbors, occ, hh, excess, degrees, threshold):
    """Exact union delta, including contacts between any two moving intervals."""
    moving = np.zeros(len(pos),np.bool_)
    changes = np.zeros(len(pos),np.int32)
    for i in indices:
        moving[i] = True
    delta = 0
    for phase in range(2):
        sign = -1 if phase == 0 else 1
        for z in range(len(indices)):
            i = indices[z]
            if not hmask[i]:
                continue
            site = pos[i] if phase == 0 else block[z]
            for v in neighbors[site]:
                j = occ[v] if v >= 0 else -1
                if j >= 0 and hmask[j] and abs(i-j)>1 and (not moving[j] or i<j):
                    delta += sign
                    changes[i] += sign
                    changes[j] += sign
        if phase == 0:
            for i in indices:
                occ[pos[i]] = -1
            for z in range(len(indices)):
                occ[block[z]] = indices[z]
    for site in block:
        occ[site] = -1
    for i in indices:
        occ[pos[i]] = i
    for i in range(len(pos)):
        if hmask[i] and changes[i]:
            excess += max(0,degrees[i]+changes[i]-threshold)-max(0,degrees[i]-threshold)
    return hh+delta,excess,changes


@njit(cache=True)
def metropolis_accept(delta_energy, temperature, uniform_draw):
    return delta_energy <= 0.0 or uniform_draw < math.exp(-delta_energy/max(temperature, 1.e-12))


@njit(cache=True)
def uniform_metropolis_selection(scores, temperature, diagnostics=None):
    """Uniform completed-candidate proposal, then one Metropolis energy test.

    Index zero is the current fold, retained on rejection. Infinite scores mark
    excluded duplicates. This is an optimization rule, not proposal-corrected MH.
    """
    valid = 0
    for j in range(1, len(scores)):
        if math.isfinite(scores[j]):
            valid += 1
    if diagnostics is not None:
        diagnostics[0] += 1
    if not valid:
        if diagnostics is not None:
            diagnostics[6] += 1
        return 0
    rank = np.random.randint(valid)
    chosen = 0
    for j in range(1, len(scores)):
        if math.isfinite(scores[j]):
            if rank == 0:
                chosen = j
                break
            rank -= 1
    if diagnostics is not None:
        diagnostics[1] += 1
    delta = scores[chosen]-scores[0]
    if delta <= 0.0:
        if diagnostics is not None:
            diagnostics[2] += 1
        return chosen
    if diagnostics is not None:
        diagnostics[3] += 1
    if metropolis_accept(delta, temperature, np.random.random()):
        if diagnostics is not None:
            diagnostics[4] += 1
        return chosen
    if diagnostics is not None:
        diagnostics[5] += 1
    return 0


@njit(cache=True)
def anneal_chunk(pos, best, hh, excess, degrees, best_hh, best_excess, hmask, coords, neighbors,
                 threshold, pair_reward, cluster_reward, min_len, max_len, beam_width,
                 t0, t1, cycle_length, offset, steps, stop_hh, routing, diversity,
                 construction_floor, routing_weight, bidirectional=False, short_beam_width=0,
                 block_stats=None, joint_probability=0.0, joint_width=24,
                 joint_parents=1, joint_expansion_budget=12000,
                 proposal_stats=None, candidate_records=None, construction_timings=None,
                 stratified=True, selection_stats=None, boundary_policy=0,
                 unguided_probability=0.0, effort_stats=None, decision_stats=None,
                 interval_stats=None, temperature_override=-1.0,
                 guided_coupled_probability=0.0, guided_anchor_attempts=4,
                 guidance_stats=None, proposal_details=None,
                 vacancy_mask=None, vacancy_boost=0.0, vacancy_stats=None,
                 triple_probability=0.0, quadruple_probability=0.0,
                 p_run_probability=0.0, p_run_windows=None, p_run_offsets=None,
                 p_run_members=None, contact_condition_probability=0.0,
                 contact_condition_gate=False, contact_condition_stats=None,
                 transition_rule=0, transition_stats=None):
    occ = occupancy(pos, len(neighbors))
    # steps, valid pools, candidates, unique signatures, transitions, expansions,
    # geometry prunes, duplicate complete assignments.
    stats = np.zeros(9, np.int64)
    milestones = np.full((steps, 2), -1, np.int64)
    events = 0
    condition_cycle_best = best_hh
    for step in range(steps):
        temperature = (temperature_override if temperature_override > 0 else
            t0*math.exp(math.log(t1/t0)*min(1., (offset+step)/max(1, cycle_length-1))))
        draw = np.random.random() if joint_probability > 0.0 else 1.0
        quadruple = (max_len >= 24 and quadruple_probability > 0.0 and
                     joint_probability <= draw < joint_probability+quadruple_probability)
        joint = draw < joint_probability or quadruple
        triple = joint and not quadruple and max_len >= 24 and draw < triple_probability
        kind = 3 if quadruple else (2 if triple else int(joint))
        unguided = unguided_probability > 0 and np.random.random() < unguided_probability
        preference = 0.0 if unguided else 1.0
        c, d, e, f, g, h, blocker, second_blocker = 0, 0, 0, 0, 0, 0, -1, -1
        mode, target_i, target_j = 0, -1, -1
        if quadruple:
            a,b,c,d,e,f,g,h,target_i,target_j,blocker,second_blocker = choose_four_intervals(
                pos,occ,neighbors,hmask,degrees,guided_anchor_attempts,vacancy_mask,vacancy_boost)
            length = b-a+d-c+f-e+h-g if a >= 0 else 0
            width = joint_width
            mode = 6 if a >= 0 else 7
        elif triple:
            a,b,c,d,e,f,target_i,target_j,blocker = choose_three_intervals(
                pos,occ,neighbors,hmask,degrees,guided_anchor_attempts,vacancy_mask,vacancy_boost)
            length = b-a+d-c+f-e if a >= 0 else 0
            width = joint_width
            mode = 4 if a >= 0 else 5
        elif joint:
            mode = 1
            p_run_attempt = (p_run_probability > 0 and
                             np.random.random() < p_run_probability)
            p_run_selected = False
            if (p_run_attempt and p_run_windows is not None and
                    p_run_offsets is not None and p_run_members is not None):
                a,b,c,d,target_i,target_j,p_run_selected = choose_p_run_intervals(
                    pos,occ,neighbors,hmask,degrees,p_run_windows,p_run_offsets,p_run_members,
                    guided_anchor_attempts,vacancy_mask,vacancy_boost)
            if p_run_selected:
                mode = 8
            elif guided_coupled_probability > 0 and np.random.random() < guided_coupled_probability:
                a, b, c, d, target_i, target_j, guided = choose_guided_intervals(
                    pos, occ, neighbors, hmask, degrees, max_len, guided_anchor_attempts,
                    vacancy_mask, vacancy_boost)
                mode = 2 if guided else 3
            else:
                a, b, c, d = choose_coupled_intervals(pos, occ, neighbors, max_len)
            if p_run_attempt and not p_run_selected:
                mode = 9
            length = b-a+d-c if a >= 0 else 0
            width = joint_width
        else:
            cap = min(max_len, len(pos)-1)
            high = min(cap, max(min_len, 8)) if np.random.random() < 0.70 else cap
            length = np.random.randint(min_len, high+1)
            a = np.random.randint(0, len(pos)-length+1)
            b = a+length
            reverse = bidirectional and reconstruction_reverse(pos,a,b,coords,neighbors,occ,boundary_policy)
            width = min(beam_width,short_beam_width) if length<=8 and short_beam_width>0 else beam_width
        condition_eligible = (contact_condition_probability > 0 and contact_condition_gate
                              and best_hh == condition_cycle_best
                              and mode in (2,8) and a >= 0 and hh >= best_hh-1)
        conditioned = condition_eligible and np.random.random() < contact_condition_probability
        condition_arm = int(conditioned)
        condition_prunes = np.zeros(1,np.int64)
        if condition_eligible and contact_condition_stats is not None:
            contact_condition_stats[condition_arm,0] += 1
        focused = (mode in (2,8) and vacancy_boost > 0 and vacancy_mask is not None)
        if focused and vacancy_stats is not None:
            vacancy_stats[0] += 1
            vacancy_stats[1] += int(vacancy_exposure(pos[target_i],occ,neighbors,vacancy_mask)>0)
            vacancy_stats[2] += int(vacancy_exposure(pos[target_j],occ,neighbors,vacancy_mask)>0)
        arm = 0 if width <= 24 else (1 if width <= 48 else 2)
        if proposal_details is not None:
            proposal_details[step, :9] = np.array(
                [a, b, c, d, width, temperature, mode, target_i, target_j], np.float64)
            if (triple or quadruple) and proposal_details.shape[1] >= 14:
                proposal_details[step,11:14] = np.array([e,f,blocker],np.float64)
            if quadruple and proposal_details.shape[1] >= 17:
                proposal_details[step,14:17] = np.array([g,h,second_blocker],np.float64)
            if proposal_details.shape[1] >= 19:
                proposal_details[step,17] = int(conditioned)
                proposal_details[step,18] = int(condition_eligible)
        indices = np.empty(length,np.int32)
        if length:
            indices[:b-a] = np.arange(a,b)
            if joint:
                indices[b-a:b-a+d-c] = np.arange(c,d)
            if triple or quadruple:
                indices[b-a+d-c:b-a+d-c+f-e] = np.arange(e,f)
            if quadruple:
                indices[b-a+d-c+f-e:] = np.arange(g,h)
        length1, length2 = (b-a, d-c) if joint and a >= 0 else (length, 0)
        sampled = block_stats is not None and (offset+step) % 4096 == 0
        construction_started = 0.0
        if (sampled and block_stats is not None) or condition_eligible:
            with objmode(construction_started='float64'):
                construction_started = time.perf_counter()
        if quadruple:
            if a >= 0:
                paths,count,expanded,pruned,unique = joint_candidates(
                    pos,np.array([a,c,e,g],np.int32),np.array([b,d,f,h],np.int32),
                    hmask,coords,neighbors,occ,pair_reward,temperature,joint_width,
                    joint_expansion_budget,routing,diversity,construction_floor,routing_weight,
                    bidirectional,stratified and not unguided,selection_stats,None,
                    boundary_policy,preference)
            else:
                paths = np.empty((0,0),np.int32)
                count,expanded,pruned,unique = 0,0,0,0
            local_counts = np.empty(0,np.int32)
        elif triple:
            if a >= 0:
                paths,count,expanded,pruned,unique = joint_candidates(
                    pos,np.array([a,c,e],np.int32),np.array([b,d,f],np.int32),
                    hmask,coords,neighbors,occ,pair_reward,temperature,joint_width,
                    joint_expansion_budget,routing,diversity,construction_floor,routing_weight,
                    bidirectional,stratified and not unguided,selection_stats,None,
                    boundary_policy,preference)
            else:
                paths = np.empty((0,0),np.int32)
                count,expanded,pruned,unique = 0,0,0,0
            local_counts = np.empty(0,np.int32)
        elif joint:
            if a >= 0:
                paths, count, expanded, pruned, unique = coupled_candidates(
                    pos, a, b, c, d, hmask, coords, neighbors, occ, pair_reward,
                    temperature, joint_width, joint_parents, joint_expansion_budget,
                    routing, diversity, construction_floor, routing_weight, bidirectional,
                    stratified and not unguided, selection_stats, None,
                    boundary_policy, preference,
                    target_i if conditioned else -1, target_j if conditioned else -1,
                    condition_prunes)
            else:
                paths = np.empty((0, 0), np.int32)
                count, expanded, pruned, unique = 0, 0, 0, 0
            local_counts = np.empty(0, np.int32)
        else:
            paths, count, expanded, pruned, unique, local_counts = beam_candidates_scored(
                pos, a, b, hmask, coords, neighbors, occ, pair_reward, temperature,
                width, routing, diversity, construction_floor, routing_weight, reverse,
                0,0,0,preference)
        if (sampled and block_stats is not None) or condition_eligible:
            with objmode(construction_seconds='float64'):
                construction_seconds = time.perf_counter()-construction_started
            if condition_eligible and contact_condition_stats is not None:
                contact_condition_stats[condition_arm,14] += construction_seconds
        if sampled and block_stats is not None:
            block_stats[length, 11] += 1
            block_stats[length, 12] += construction_seconds
            if construction_timings is not None:
                construction_timings[kind, 0] += 1
                construction_timings[kind, 1] += construction_seconds
        if block_stats is not None:
            block_stats[length, 0] += 1
            block_stats[length, 5] += expanded
            block_stats[length, 6] += pruned
        stats[0] += 1
        stats[5] += expanded
        stats[6] += pruned
        if condition_eligible and contact_condition_stats is not None:
            contact_condition_stats[condition_arm,3] += expanded
            contact_condition_stats[condition_arm,4] += condition_prunes[0]
        if focused and vacancy_stats is not None:
            vacancy_stats[7] += expanded
        if guidance_stats is not None:
            guidance_stats[mode, 0] += 1
            guidance_stats[mode, 2] += expanded
        if effort_stats is not None and not joint and length > 8:
            effort_stats[arm,0] += 1
            effort_stats[arm,1] += expanded
        if decision_stats is not None:
            decision_stats[kind,arm,int(unguided),0] += 1
            decision_stats[kind,arm,int(unguided),2] += expanded
        if interval_stats is not None and not (triple or quadruple):
            interval_stats[length1,length2,0] += 1
            interval_stats[length1,length2,3] += expanded
        if proposal_stats is not None:
            proposal_stats[kind, 0] += 1
            proposal_stats[kind, 8] += expanded
            proposal_stats[kind, 9] += pruned
            if joint and a < 0:
                proposal_stats[kind, 11] += 1
        if not count:
            if transition_rule == 1 and transition_stats is not None:
                transition_stats[7] += 1
            continue
        if block_stats is not None:
            block_stats[length, 1] += 1
            block_stats[length, 2] += count
            block_stats[length, 3] += unique
            block_stats[length, 8] += width
        stats[1] += 1
        stats[8] += width
        stats[2] += count
        stats[3] += unique
        scores = np.full(count+1, np.inf)
        counts = np.zeros(count+1, np.int32)
        excesses = np.zeros(count+1, np.int32)
        fixed_hh = hh-block_contact_count(pos,a,b,hmask,neighbors,occ) if not joint else 0
        scores[0], counts[0], excesses[0] = pair_reward*hh+cluster_reward*excess, hh, excess
        distinct_count, pool_best, completed_best = 0, hh, -1
        for k in range(count):
            duplicate = False
            if diversity:
                duplicate = True
                for site_offset in range(length):
                    residue = indices[site_offset]
                    if paths[k, site_offset] != pos[residue]:
                        duplicate = False
                        break
                for j in range(k):
                    same = True
                    for site_offset in range(length):
                        if paths[k, site_offset] != paths[j, site_offset]:
                            same = False
                            break
                    duplicate = duplicate or same
            if duplicate:
                stats[7] += 1
                if block_stats is not None:
                    block_stats[length, 7] += 1
                continue
            if triple or quadruple:
                ch,ce,_ = joint_delta(pos,paths[k],indices,hmask,neighbors,occ,hh,excess,degrees,threshold)
            elif joint:
                ch, ce, _ = coupled_delta(pos,paths[k],a,b,c,d,hmask,
                                          neighbors,occ,hh,excess,degrees,threshold)
            else:
                ch, ce = fixed_hh+local_counts[k], 0
            if not joint and cluster_reward != 0.0:
                _,ce,_ = local_delta(pos,paths[k],a,b,hmask,neighbors,occ,hh,excess,degrees,threshold)
            counts[k+1], excesses[k+1] = ch, ce
            scores[k+1] = pair_reward*ch + cluster_reward*ce
            distinct_count += 1
            pool_best = max(pool_best, ch)
            completed_best = max(completed_best,ch)
        old_best, old_hh = best_hh, hh
        if condition_eligible and contact_condition_stats is not None:
            closed = 0
            for candidate in range(count):
                first_site = paths[candidate,target_i-a]
                second_site = paths[candidate,b-a+target_j-c]
                is_closed = graph_distance(coords,first_site,second_site) == 1
                assert not conditioned or is_closed
                closed += int(is_closed)
            contact_condition_stats[condition_arm,1] += 1
            contact_condition_stats[condition_arm,2] += count
            contact_condition_stats[condition_arm,5] += closed
            contact_condition_stats[condition_arm,6] += int(closed > 0)
            contact_condition_stats[condition_arm,11] += max(0,pool_best-old_best)
            contact_condition_stats[condition_arm,12] += max(0,pool_best-hh)
            contact_condition_stats[condition_arm,15] = max(
                contact_condition_stats[condition_arm,15],pool_best)
        if focused and vacancy_stats is not None:
            vacancy_stats[3] += 1
            vacancy_stats[4] += int(pool_best > old_best)
        if guidance_stats is not None:
            guidance_stats[mode, 1] += 1
            guidance_stats[mode, 3] += int(pool_best > old_best)
        if effort_stats is not None and not joint and length > 8:
            effort_stats[arm,2] += max(0,pool_best-hh)
            effort_stats[arm,3] += max(0,pool_best-old_best)
            effort_stats[arm,4] += int(completed_best >= old_best)
        if decision_stats is not None:
            decision_stats[kind,arm,int(unguided),1] += 1
            decision_stats[kind,arm,int(unguided),3] += max(0,pool_best-hh)
            decision_stats[kind,arm,int(unguided),4] += max(0,pool_best-old_best)
        if interval_stats is not None and not (triple or quadruple):
            interval_stats[length1,length2,1] += 1
        if proposal_stats is not None:
            proposal_stats[kind, 1] += 1
            proposal_stats[kind, 2] += distinct_count
            proposal_stats[kind, 5] += int(pool_best > old_best)
            proposal_stats[kind, 10] = max(proposal_stats[kind, 10], pool_best)
        if transition_rule == 1:
            chosen = uniform_metropolis_selection(scores, temperature, transition_stats)
        else:
            weights = np.exp(-(scores-np.min(scores))/max(temperature, 1.e-12))
            pick = np.random.random()*np.sum(weights)
            chosen, cumulative = 0, 0.0
            for j in range(count+1):
                cumulative += weights[j]
                if cumulative > pick:
                    chosen = j
                    break
        if chosen:
            first_changed = np.any(pos[a:b] != paths[chosen-1, :b-a])
            second_changed = joint and np.any(pos[c:d] != paths[chosen-1,b-a:b-a+d-c])
            all_changed = first_changed and second_changed
            if triple or quadruple:
                all_changed = all_changed and np.any(pos[e:f] != paths[chosen-1,b-a+d-c:b-a+d-c+f-e])
            if quadruple:
                all_changed = all_changed and np.any(pos[g:h] != paths[chosen-1,b-a+d-c+f-e:])
            for i in indices:
                occ[pos[i]] = -1
            for z in range(length):
                pos[indices[z]] = paths[chosen-1,z]
            for i in indices:
                occ[pos[i]] = i
            hh, excess, degrees = full_metrics(pos,hmask,neighbors,threshold)
            if condition_eligible and contact_condition_stats is not None:
                closed = graph_distance(coords,pos[target_i],pos[target_j]) == 1
                assert not conditioned or closed
                contact_condition_stats[condition_arm,7] += 1
                contact_condition_stats[condition_arm,8] += int(closed)
                contact_condition_stats[condition_arm,9] += int(hh > best_hh)
                contact_condition_stats[condition_arm,10] += max(0,hh-best_hh)
                contact_condition_stats[condition_arm,13] += hh-old_hh
            if proposal_stats is not None:
                proposal_stats[kind, 3] += 1
                proposal_stats[kind, 4] += int(all_changed)
            stats[4] += 1
            if block_stats is not None:
                block_stats[length, 4] += 1
            if hh > best_hh or (hh == best_hh and excess > best_excess):
                if hh > best_hh:
                    if focused and vacancy_stats is not None:
                        vacancy_stats[5] += 1
                        vacancy_stats[6] += hh-best_hh
                    if proposal_details is not None:
                        created, broken = contact_changes(best, pos, hmask, neighbors)
                        proposal_details[step, 9] = created
                        proposal_details[step, 10] = broken
                    if guidance_stats is not None:
                        guidance_stats[mode, 4] += 1
                        guidance_stats[mode, 5] += hh-best_hh
                    if decision_stats is not None:
                        decision_stats[kind,arm,int(unguided),5] += hh-best_hh
                    if interval_stats is not None and not (triple or quadruple):
                        interval_stats[length1,length2,2] += hh-best_hh
                    if proposal_stats is not None:
                        proposal_stats[kind, 6] += 1
                        proposal_stats[kind, 7] += hh-best_hh
                    if block_stats is not None:
                        block_stats[length, 9] += 1
                        block_stats[length, 10] += hh-best_hh
                    milestones[events, 0], milestones[events, 1] = step+1, hh
                    events += 1
                best_hh, best_excess = hh, excess
                best[:] = pos
                if best_hh >= stop_hh:
                    if candidate_records is not None:
                        candidate_records[step] = np.array([kind, pool_best, hh, old_best], np.int64)
                    break
        if candidate_records is not None:
            candidate_records[step] = np.array([kind, pool_best, hh, old_best], np.int64)
    return hh, excess, degrees, best_hh, best_excess, stats, milestones[:events]
