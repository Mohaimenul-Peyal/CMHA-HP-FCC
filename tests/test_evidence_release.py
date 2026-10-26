import csv
import json
from pathlib import Path
import statistics
import unittest

from cmha_results import validate_fold
from tools.analyze_proposals import analyze
from tools.draw_cmha_schematics import validate_example

ROOT = Path(__file__).resolve().parents[1]


class EvidenceReleaseTests(unittest.TestCase):
    def test_synthetic_example_geometry_scores_and_draw(self):
        example = validate_example(json.loads((ROOT/'data/schematic_multiblock_example.json').read_text()))
        self.assertEqual(example['N'], 16)
        self.assertEqual(len(example['pool']), 25)
        self.assertEqual(example['fixed_boundary_indices_one_based'], [1,8,9,16])
        self.assertEqual(example['pool'][example['selected_index']]['HH'], 6)

    def test_additional_records_and_distinct_execution_keys(self):
        data = json.loads((ROOT/'data/confirmation_data.json').read_text())
        extra = json.loads((ROOT/'data/confirmation_worker_folds.json').read_text())
        self.assertEqual(len(extra), 64)
        groups = {}
        for row in extra:
            validate_fold(row['seq'], row['coords'], row['L'], row['HH'])
            validate_fold(row['seq'], row['initial_coords'], row['L'])
            self.assertGreaterEqual(row['wall_seconds'], 21600)
            self.assertEqual(row['seed'], row['base_seed']+1000003*row['worker_id'])
            groups.setdefault((row['benchmark'],row['base_seed']),[]).append(row['HH'])
        self.assertEqual(len(groups), 4)
        self.assertEqual(len({r['whole_run_key'] for r in data['runs']}), 6)
        for row in data['runs']:
            if row['role']=='original_seed_control':
                continue
            values = groups[(row['benchmark'],row['base_seed'])]
            self.assertEqual(len(values), 16)
            self.assertEqual(sorted(values), sorted(row['worker_HH']))
            self.assertEqual(max(values), row['best_HH'])
            self.assertAlmostEqual(statistics.mean(values), row['mean_HH'])
            self.assertAlmostEqual(statistics.stdev(values), row['SD_HH'])

    def test_complete_run_summary_retains_weaker_outcomes(self):
        data = json.loads((ROOT/'data/confirmation_data.json').read_text())
        summaries = {r['benchmark']:r for r in data['summaries']}
        self.assertEqual(summaries['3mr7']['complete_run_bests'], [350,350,350])
        self.assertEqual(summaries['3no3']['complete_run_bests'], [474,473,473])
        self.assertAlmostEqual(summaries['3no3']['mean_complete_run_best_HH'], 473+1/3)

    def test_sampled_productivity_regenerates_from_bundled_windows(self):
        actual = analyze()
        with (ROOT/'data/proposal_productivity.csv').open(newline='') as stream:
            expected = list(csv.DictReader(stream))
        self.assertEqual(len(actual), 32)
        self.assertEqual(len(actual), len(expected))
        for row, saved in zip(actual, expected):
            self.assertEqual((row['benchmark'],row['period'],row['family']),
                             (saved['benchmark'],saved['period'],saved['family']))
            for field in ('proposals','valid_pools','physical_best_improvements','timing_samples'):
                self.assertEqual(row[field], int(saved[field]))
            self.assertAlmostEqual(row['estimated_construction_seconds'],
                                   float(saved['estimated_construction_seconds']),places=6)


if __name__=='__main__':
    unittest.main()
