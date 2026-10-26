"""Target-independent beam allocation and configurable late-stage cycle cap."""
from dataclasses import dataclass
import math


def budget_stage(elapsed, wall_seconds, steps, max_proposals=None):
    """Minimum objective stage; proposal caps take precedence for repeatable tests."""
    fraction = steps/max_proposals if max_proposals is not None else elapsed/wall_seconds
    return min(3,max(0,int(4*fraction)))


def capped_cycle(requested, stage, elapsed, wall_seconds, steps, max_proposals, rate,
                 late_cycle_cap_fraction=0.125):
    """Keep the original cap before stage 3; optionally lengthen later cycles.

    Proposal-limited tests never use measured rates or clocks to select a cycle.
    With wall budgets, rate is frozen at the cycle boundary and logged by the caller.
    Remaining stage/budget time is still a hard input to the estimated step cap.
    """
    fraction = late_cycle_cap_fraction if stage >= 3 else 0.125
    if max_proposals is not None:
        phase_end = (min(stage,3)+1)*max_proposals/4
        remaining = min(max_proposals-steps,phase_end-steps,max_proposals*fraction)
    elif rate is not None and rate>0 and math.isfinite(rate):
        phase_end = (min(stage,3)+1)*wall_seconds/4
        remaining = rate*min(wall_seconds-elapsed,phase_end-elapsed,wall_seconds*fraction)
    else:
        return requested
    return max(1,min(requested,int(max(1,remaining))))


@dataclass
class EffortController:
    width: int = 24
    base: int = 24
    maximum: int = 96
    window_steps: int = 8000
    next_check: int = 8000
    last_improvement: int = 0

    def update(self, steps, improved, valid_fraction, diversity_fraction, fill_fraction):
        if improved:
            self.last_improvement = steps
            self.next_check = steps + self.window_steps
            self.width = self.base
            return 'physical_improvement_reset'
        if steps < self.next_check:
            return 'continue'
        self.next_check = steps + self.window_steps
        if steps - self.last_improvement < self.window_steps:
            return 'recent_improvement'
        if valid_fraction < 0.1:
            return 'geometry_limited_keep_width'
        if diversity_fraction < 0.2:
            return 'diversity_limited_keep_width'
        if fill_fraction < 0.75:
            return 'pool_not_saturated_keep_width'
        self.width = min(self.maximum, self.width*2)
        return 'stagnant_saturated_pool_widen'
