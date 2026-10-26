"""Route-aware restart memory; physical scoring and candidate kernel stay unchanged."""
from dataclasses import dataclass
import math

from .diversity_restart import DiversityArchive, Entry, map_id


def polar_environment(positions, hmask, neighbors):
    """Noncovalent HP/PP adjacency pairs, a rotation-invariant route proxy.

    These are NOT energetic contacts in the HP model. They characterize which
    residues obstruct each other's immediate surroundings and affect memory only.
    """
    occupied = {int(site): i for i, site in enumerate(positions)}
    pairs = set()
    for i, site in enumerate(positions):
        for neighbor in neighbors[int(site)]:
            j = occupied.get(int(neighbor), -1)
            if j > i+1 and not (hmask[i] and hmask[j]):
                pairs.add((i, j))
    return frozenset(pairs)


def pair_bits(pairs, n):
    value = 0
    for i, j in pairs:
        value |= 1 << (i*n+j)
    return value


def bit_distance(first, second):
    union = (first | second).bit_count()
    return (first ^ second).bit_count()/union if union else 0.0


@dataclass
class RouteEntry(Entry):
    routing: frozenset = frozenset()
    hbits: int = 0
    rbits: int = 0

    def record(self):
        data = super().record()
        data.update(polar_environment_pairs=[list(p) for p in sorted(self.routing)],
                    polar_environment_sha256=map_id(self.routing))
        return data


class RouteDiversityArchive(DiversityArchive):
    """Same memory capacity/gap/restart clock, finer-grained spatial diversity."""

    def __init__(self, seed, hmask, neighbors, capacity=16, max_gap=2,
                 min_distance=.02, route_weight=.25, routes_per_map=2):
        super().__init__(seed,capacity,max_gap,min_distance)
        if not 0 < route_weight <= 1 or routes_per_map < 1:
            raise ValueError('Invalid route-diversity configuration')
        self.hmask = tuple(map(bool,hmask))
        self.neighbors = neighbors
        self.route_weight = float(route_weight)
        self.routes_per_map = int(routes_per_map)
        self.same_H_map_observations = 0
        self.same_H_map_admissions = 0
        self.same_H_map_as_best_restarts = 0

    def separation(self, first, second):
        return max(bit_distance(first.hbits,second.hbits),
                   self.route_weight*bit_distance(first.rbits,second.rbits))

    def entry(self, positions, contacts, proposal=0, elapsed=0.):
        routing=polar_environment(positions,self.hmask,self.neighbors)
        contacts=frozenset(contacts)
        return RouteEntry(tuple(map(int,positions)),contacts,int(proposal),float(elapsed),
                          0,routing,pair_bits(contacts,len(positions)),
                          pair_bits(routing,len(positions)))

    def observe(self, positions, contacts, best_hh, proposal, elapsed):
        self.observations += 1
        contacts=frozenset(contacts)
        if len(contacts)>best_hh:
            raise ValueError('Archive candidate cannot exceed recorded worker best')
        self.prune(best_hh)
        if len(contacts)<self.best_hh-self.max_gap:
            return False
        candidate=self.entry(positions,contacts,proposal,elapsed)
        same_map=[e for e in self.entries if e.contacts==contacts]
        self.same_H_map_observations += int(bool(same_map))
        if any(e.routing==candidate.routing for e in same_map):
            return False
        nearest=min(self.entries,key=lambda e:self.separation(e,candidate),default=None)
        if nearest is not None and self.separation(nearest,candidate)<self.min_distance:
            if candidate.hh<=nearest.hh:
                return False
            self.entries.remove(nearest)
        pool=self.entries+[candidate]
        # Apply one bounded farthest-point selection, preserving the physical
        # best while preventing a single H map from consuming every slot.
        anchor=max(pool,key=lambda e:(e.hh,-e.visits,-e.proposal))
        retained=[anchor]
        remaining=[e for e in pool if e is not anchor]
        map_counts={anchor.hbits:1}
        closest={id(e):self.separation(e,anchor) for e in remaining}
        while len(retained)<self.capacity and remaining:
            eligible=[e for e in remaining if map_counts.get(e.hbits,0)<self.routes_per_map]
            if not eligible:
                break
            chosen=max(eligible,key=lambda e:(
                math.exp(-.5*(self.best_hh-e.hh))*closest[id(e)],
                e.hh,-e.visits,-e.proposal))
            retained.append(chosen)
            remaining.remove(chosen)
            map_counts[chosen.hbits]=map_counts.get(chosen.hbits,0)+1
            for e in remaining:
                closest[id(e)]=min(closest[id(e)],self.separation(e,chosen))
        self.entries=retained
        accepted=any(e is candidate for e in retained)
        self.admissions += int(accepted)
        self.same_H_map_admissions += int(accepted and bool(same_map))
        return accepted

    def choose(self, best_positions, best_contacts, best_hh, current_positions):
        self.prune(best_hh)
        best=self.entry(best_positions,best_contacts)
        current=tuple(map(int,current_positions))
        options=[e for e in self.entries if e.positions!=current and
                 self.separation(e,best)>0 and self.separation(e,best)>=self.min_distance]
        if not options:
            return tuple(map(int,best_positions)),None
        weights=[math.exp(-(best_hh-e.hh))*self.separation(e,best)/(1+e.visits)
                 for e in options]
        choice=self.rng.choices(options,weights=weights,k=1)[0]
        choice.visits+=1
        self.restarts+=1
        same=choice.contacts==best_contacts
        self.same_H_map_as_best_restarts+=int(same)
        return choice.positions,dict(source_contact_map_sha256=map_id(choice.contacts),
            source_polar_environment_sha256=map_id(choice.routing),
            source_HH=choice.hh,gap_to_worker_best=best_hh-choice.hh,
            jaccard_distance_to_worker_best=bit_distance(choice.hbits,best.hbits),
            polar_environment_distance_to_worker_best=bit_distance(choice.rbits,best.rbits),
            combined_archive_distance=self.separation(choice,best),
            same_H_map_as_best=same,
            source_first_observed_proposal=choice.proposal,
            source_first_observed_wall_seconds=choice.elapsed,
            source_restart_selections=choice.visits)

    def snapshot(self):
        data=super().snapshot()
        data.update(route_weight=self.route_weight,max_routes_per_H_map=self.routes_per_map,
            same_H_map_observations=self.same_H_map_observations,
            same_H_map_admissions=self.same_H_map_admissions,
            same_H_map_as_best_restarts=self.same_H_map_as_best_restarts,
            distinct_H_maps=len({e.hbits for e in self.entries}),
            route_proxy='Noncovalent HP/PP adjacency, not added HP energy and not a complete topology invariant')
        return data
