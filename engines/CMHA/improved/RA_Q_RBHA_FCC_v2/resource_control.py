"""Budget-aware policies; no benchmark target enters controller decisions."""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class ResourceEffortController:
    """Probe each width, then prefer measured candidate yield per expansion.

    The statistics contain ONLY single-block proposals longer than eight.
    Expansion count is a deterministic cost proxy, not measured CPU time.
    Every eight windows contain three probe windows and five exploit windows.
    """
    base: int = 24
    maximum: int = 96
    window_steps: int = 8000
    width: int = 24
    next_check: int = 8000
    window_index: int = 0
    previous: np.ndarray = field(default_factory=lambda: np.zeros((3, 5)))
    recent: np.ndarray = field(default_factory=lambda: np.zeros((3, 5)))

    def update(self, steps, values):
        if steps < self.next_check:
            return 'continue'
        delta = values - self.previous
        self.previous[:] = values
        self.recent = .8*self.recent + delta
        self.next_check = steps + self.window_steps
        self.window_index += 1
        widths = [w for w in (24, 48, 96) if w <= self.maximum]
        slot = self.window_index % 8
        if slot < len(widths):
            self.width = widths[slot]
            return 'scheduled_width_probe'
        # Prefer the cheapest arm on ties, including an unproductive plateau.
        quality = self.recent[:, 2] + 100*self.recent[:, 3] + .1*self.recent[:, 4]
        rates = quality / np.maximum(1., self.recent[:, 1])
        rates[self.recent[:, 0] < 64] = -1.
        self.width = widths[int(np.argmax(rates[:len(widths)]))]
        return 'candidate_yield_per_expansion'

    def new_cycle(self):
        # Retain evidence but discount the preceding temperature trajectory.
        self.recent *= .5


def terminal_cycle_plan(requested, stage, elapsed, wall, steps, limit, rate,
                        cap_fraction, legacy_length):
    """Keep ordinary proposal cycles; use one final cooling pass, then stop.

    A full requested cycle that no longer fits the remaining budget becomes a
    terminal pass. The last 5% also cannot start another reheating cycle.
    The terminal temperature uses remaining wall time (or proposals in tests).
    """
    if limit is not None:
        remaining = max(0, limit-steps)
        terminal = stage >= 3 and (remaining <= limit*.05 or requested >= remaining)
        return (max(1, remaining) if terminal else legacy_length), terminal
    remaining = max(0., wall-elapsed)
    measured = rate is not None and rate > 0 and math.isfinite(rate)
    terminal = stage >= 3 and (remaining <= wall*.05 or (measured and requested/rate >= remaining))
    if terminal:
        return max(1, int(remaining*rate)) if measured else requested, True
    return legacy_length, False


def terminal_temperature(t0, t1, elapsed, start, wall, steps, start_steps, limit):
    if limit is not None:
        fraction = (steps-start_steps)/max(1, limit-start_steps-1)
    else:
        fraction = (elapsed-start)/max(1.e-12, wall-start)
    return t0*math.exp(math.log(t1/t0)*min(1., max(0., fraction)))
