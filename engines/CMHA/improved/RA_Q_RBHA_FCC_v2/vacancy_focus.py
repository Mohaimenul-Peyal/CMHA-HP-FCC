"""Bounded proposal-selection pulse after a neutral polar-route refresh."""
import numpy as np

FIELDS = ('guided_pairs', 'exposed_anchors', 'exposed_partners', 'valid_pools',
          'pools_beating_best', 'best_improvements', 'contacts_gained', 'expansions')


def vacated_mask(before, after, site_count):
    mask = np.zeros(site_count, np.bool_)
    mask[np.asarray(before, np.int32)] = True
    mask[np.asarray(after, np.int32)] = False
    return mask


def pulse_remaining(proposal, end_proposal):
    return max(0, int(end_proposal)-int(proposal))


def counts(values):
    return dict(zip(FIELDS, map(int, values)))
