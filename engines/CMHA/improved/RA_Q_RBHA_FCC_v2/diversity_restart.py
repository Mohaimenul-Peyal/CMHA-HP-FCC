"""Worker-local, quality-diverse restart archive; no changes to proposal energy."""
from dataclasses import dataclass
import hashlib
import math
import random


def contact_map(positions, hmask, neighbors):
    occupied = {int(site): i for i, site in enumerate(positions)}
    pairs = set()
    for i, site in enumerate(positions):
        if not hmask[i]:
            continue
        for neighbor in neighbors[int(site)]:
            j = occupied.get(int(neighbor), -1)
            if j > i + 1 and hmask[j]:
                pairs.add((i, j))
    return frozenset(pairs)


def distance(a, b):
    union = len(a | b)
    return len(a ^ b) / union if union else 0.0


def map_id(contacts):
    value = ';'.join(f'{a},{b}' for a, b in sorted(contacts))
    return hashlib.sha256(value.encode('ascii')).hexdigest()


@dataclass
class Entry:
    positions: tuple
    contacts: frozenset
    proposal: int
    elapsed: float
    visits: int = 0

    @property
    def hh(self):
        return len(self.contacts)

    def record(self):
        return dict(positions=list(self.positions), HH=self.hh,
                    contact_pairs=[list(p) for p in sorted(self.contacts)],
                    contact_map_sha256=map_id(self.contacts),
                    first_observed_proposal=self.proposal,
                    first_observed_wall_seconds=self.elapsed,
                    restart_selections=self.visits)


class DiversityArchive:
    """Bounded farthest-point archive, scored relative to the worker's own best."""

    def __init__(self, seed, capacity=16, max_gap=2, min_distance=0.02):
        if capacity < 2 or max_gap < 0 or not 0 <= min_distance <= 1:
            raise ValueError('Invalid diversity archive configuration')
        self.capacity = capacity
        self.max_gap = max_gap
        self.min_distance = min_distance
        # Do not consume random draws from the annealing kernel or initializer.
        self.rng = random.Random(int(seed) ^ 0x44525141)
        self.entries = []
        self.best_hh = -1
        self.observations = 0
        self.admissions = 0
        self.restarts = 0

    def prune(self, best_hh):
        self.best_hh = max(self.best_hh, int(best_hh))
        self.entries = [e for e in self.entries
                        if e.hh >= self.best_hh - self.max_gap]

    def observe(self, positions, contacts, best_hh, proposal, elapsed):
        self.observations += 1
        contacts = frozenset(contacts)
        if len(contacts) > best_hh:
            raise ValueError('Archive candidate cannot exceed recorded worker best')
        self.prune(best_hh)
        if len(contacts) < self.best_hh - self.max_gap:
            return False
        if any(e.contacts == contacts for e in self.entries):
            return False
        candidate = Entry(tuple(map(int, positions)), contacts, int(proposal), float(elapsed))
        nearest = min(self.entries, key=lambda e: distance(e.contacts, contacts), default=None)
        if nearest is not None and distance(nearest.contacts, contacts) < self.min_distance:
            if candidate.hh <= nearest.hh:
                return False
            self.entries.remove(nearest)
        pool = self.entries + [candidate]
        if len(pool) > self.capacity:
            # Keep quality and distinct contact maps, not just the newest states.
            anchor = max(pool, key=lambda e: (e.hh, -e.visits, -e.proposal))
            retained = [anchor]
            remaining = [e for e in pool if e is not anchor]
            while len(retained) < self.capacity:
                chosen = max(remaining, key=lambda e: (
                    math.exp(-0.5 * (self.best_hh - e.hh)) *
                    min(distance(e.contacts, r.contacts) for r in retained),
                    e.hh, -e.visits, -e.proposal))
                retained.append(chosen)
                remaining.remove(chosen)
            self.entries = retained
        else:
            self.entries = pool
        accepted = any(e is candidate for e in self.entries)
        self.admissions += int(accepted)
        return accepted

    def choose(self, best_positions, best_contacts, best_hh, current_positions):
        self.prune(best_hh)
        current = tuple(map(int, current_positions))
        options = [e for e in self.entries
                   if e.positions != current and e.contacts != best_contacts
                   and distance(e.contacts, best_contacts) >= self.min_distance]
        if not options:
            return tuple(map(int, best_positions)), None
        weights = [math.exp(-(best_hh - e.hh)) *
                   distance(e.contacts, best_contacts) / (1 + e.visits) for e in options]
        choice = self.rng.choices(options, weights=weights, k=1)[0]
        choice.visits += 1
        self.restarts += 1
        return choice.positions, dict(source_contact_map_sha256=map_id(choice.contacts),
            source_HH=choice.hh, gap_to_worker_best=best_hh-choice.hh,
            jaccard_distance_to_worker_best=distance(choice.contacts, best_contacts),
            source_first_observed_proposal=choice.proposal,
            source_first_observed_wall_seconds=choice.elapsed,
            source_restart_selections=choice.visits)

    def snapshot(self):
        return dict(capacity=self.capacity, max_contact_gap=self.max_gap,
                    min_jaccard_distance=self.min_distance, observations=self.observations,
                    admissions=self.admissions, restart_count=self.restarts,
                    best_HH=self.best_hh, entries=[e.record() for e in self.entries])
