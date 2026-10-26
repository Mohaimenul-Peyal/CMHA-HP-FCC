import itertools
import random
import unittest

import numpy as np

from cmha_runner import ROOT, load_engine
from cmha_results import validate_fold

engine = load_engine(ROOT / "engines/CMHA")
from improved.RA_Q_RBHA_FCC_v2 import kernel as kernel
from improved.RA_Q_RBHA_FCC_v2.controller import budget_stage
from improved.RA_Q_RBHA_FCC_v2.resource_control import terminal_cycle_plan, terminal_temperature


def fixture(side=4, n=13, seed=7):
    xyz = np.asarray(engine.fcc_lattice_coords(side), np.int32)
    neighbors = engine.neighbors_array(engine.fcc_neighbor_sets([tuple(p) for p in xyz]))
    rng = random.Random(seed)
    for _ in range(100):
        pos = [rng.randrange(len(xyz))]
        while len(pos) < n:
            choices = [int(v) for v in neighbors[pos[-1]] if v >= 0 and v not in pos]
            if not choices:
                break
            pos.append(rng.choice(choices))
        if len(pos) == n:
            return xyz, neighbors, np.asarray(pos, np.int32), np.array([rng.randrange(3) != 0 for _ in pos], np.int8)
    raise AssertionError("Tiny SAW fixture failed")


def candidates(xyz, neighbors, pos, h, starts, ends, width=24, budget=12000):
    kernel.seed_kernel(19)
    return kernel.joint_candidates(pos, np.array(starts, np.int32), np.array(ends, np.int32),
        h, xyz, neighbors, kernel.occupancy(pos, len(neighbors)), -2., 1.2, width,
        budget, True, True, .4, .15, True, False, None, None, 1, 1.)


class FrozenKernelTests(unittest.TestCase):
    def test_metrics_agree_with_independent_geometry(self):
        xyz, neighbors, pos, h = fixture()
        hh, excess, degrees = kernel.full_metrics(pos, h, neighbors, 2)
        check = validate_fold("".join("H" if x else "P" for x in h), xyz[pos].tolist(), 4)
        self.assertEqual(hh, check["HH"])
        self.assertEqual(excess, sum(max(0, int(degrees[i])-2) for i in range(len(h)) if h[i]))

    def test_joint_delta_keeps_occupancy_and_fixed_residues(self):
        xyz, neighbors, pos, h = fixture()
        starts, ends = [1, 7], [4, 10]
        indices = np.array([1, 2, 3, 7, 8, 9], np.int32)
        hh, excess, degrees = kernel.full_metrics(pos, h, neighbors, 2)
        occ = kernel.occupancy(pos, len(neighbors))
        original = occ.copy()
        paths, count, expansions, *_ = candidates(xyz, neighbors, pos, h, starts, ends)
        self.assertGreater(count, 0)
        self.assertLessEqual(expansions, 12000)
        for path in paths[:count]:
            trial = pos.copy()
            trial[indices] = path
            ch, ce, dd = kernel.joint_delta(pos, path, indices, h, neighbors, occ, hh, excess, degrees, 2)
            full_hh, full_ex, full_degrees = kernel.full_metrics(trial, h, neighbors, 2)
            self.assertEqual((ch, ce), (full_hh, full_ex))
            np.testing.assert_array_equal(degrees + dd, full_degrees)
            np.testing.assert_array_equal(occ, original)
            sequence = "".join("H" if x else "P" for x in h)
            validate_fold(sequence, xyz[trial].tolist(), 4, ch)

    def test_four_singletons_match_tiny_exhaustive_enumeration(self):
        xyz, neighbors, pos, h = fixture(seed=4)
        starts = [1, 4, 7, 10]
        ends = [2, 5, 8, 11]
        fixed = set(pos.tolist()) - {int(pos[i]) for i in starts}
        domains = [(set(neighbors[pos[i-1]]) & set(neighbors[pos[i+1]])) - fixed - {-1} for i in starts]
        expected = {tuple(map(int, row)) for row in itertools.product(*domains) if len(set(row)) == 4}
        paths, count, expansions, *_ = candidates(xyz, neighbors, pos, h, starts, ends, 20736, 50000)
        self.assertEqual(set(map(tuple, paths[:count])), expected)
        self.assertLessEqual(expansions, 50000)

    def test_cycle_control_and_terminal_cooling(self):
        self.assertEqual(budget_stage(0, 10800, 0), 0)
        self.assertEqual(budget_stage(8100, 10800, 0), 3)
        length, terminal = terminal_cycle_plan(10240000, 7, 10000, 10800, 100, None, 2000, .125, 1600000)
        self.assertTrue(terminal)
        self.assertEqual(length, 1600000)
        temperatures = [terminal_temperature(8., .02, t, 10000, 10800, 0, 0, None) for t in (10000, 10400, 10800)]
        self.assertEqual(temperatures[0], 8.)
        self.assertAlmostEqual(temperatures[-1], .02)
        self.assertTrue(all(a >= b for a, b in zip(temperatures, temperatures[1:])))


if __name__ == "__main__":
    unittest.main()
