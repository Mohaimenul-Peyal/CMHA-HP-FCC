"""Independent checks of the archived single-block baseline evidence."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NEIGHBORS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
             if sorted(map(abs, (a, b, c))) == [0, 1, 1]]


def contacts(seq, coords, side):
    points = [tuple(p) for p in coords]
    assert len(points) == len(seq) and len(set(points)) == len(points)
    assert all(sum(p) % 2 == 0 and all(0 <= v < side for v in p) for p in points)
    assert all(sum((a - b) ** 2 for a, b in zip(p, q)) == 2 for p, q in zip(points, points[1:]))
    index = {p: i for i, p in enumerate(points)}
    count = 0
    for i, p in enumerate(points):
        if seq[i] == 'H':
            for d in NEIGHBORS:
                j = index.get(tuple(a + b for a, b in zip(p, d)))
                if j is not None and j > i + 1 and seq[j] == 'H':
                    count += 1
    return count


class SingleBlockBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads((ROOT / 'data/single_block_baseline.json').read_text())['instances']
        cls.primary = {row['benchmark']: row for row in json.loads((ROOT / 'data/paper_data.json').read_text())['primary']}

    def test_all_192_folds_are_valid(self):
        self.assertEqual(len(self.baseline), 12)
        for name, item in self.baseline.items():
            self.assertEqual(item['sequence'], self.primary[name]['seq'])
            self.assertEqual(item['L'], self.primary[name]['L'])
            self.assertEqual(item['wall_s_per_worker'], self.primary[name]['wall_s'])
            self.assertTrue(item['initial_walks_identical_to_primary_CMHA'])
            self.assertEqual(len(item['workers']), 16)
            for worker in item['workers']:
                self.assertEqual(contacts(item['sequence'], worker['coords'], item['L']), worker['best_HH'])

    def test_archived_sources_match_recorded_hashes(self):
        for snapshot in ('snapshot_20260916', 'snapshot_20260921'):
            folder = ROOT / 'baselines/single_block' / snapshot
            recorded = json.loads((folder / 'source_sha256.json').read_text())
            flat = {}

            def walk(node, prefix=''):
                for key, value in node.items():
                    if isinstance(value, dict):
                        walk(value, prefix + key + '/')
                    else:
                        flat[prefix + key] = value
            walk(recorded)
            self.assertGreater(len(flat), 0)
            for relative, digest in flat.items():
                path = folder / relative.split('source_snapshot/', 1)[-1]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, relative)


if __name__ == '__main__':
    unittest.main()
