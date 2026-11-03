"""Regression checks for corruption and misleading aggregation of published evidence."""
import copy
import json
import unittest

import numpy as np
import pyarrow.parquet as pq

from analyze_checkpoint_statistics import holm
from verify_results import RESULTS, SOURCE, validate_rows, verify, verify_manifest


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pq.read_table(RESULTS / 'rollouts.parquet').to_pylist()
        cls.summaries = pq.read_table(RESULTS / 'question_means.parquet').to_pylist()
        cls.config = json.loads((RESULTS / 'config.json').read_text())
        cls.source = pq.read_table(SOURCE).to_pylist()

    def test_complete_saved_evidence(self):
        verify_manifest()
        counts = verify()
        self.assertEqual([counts[s]['complete'] for s in (0, 50, 100)], [980, 947, 970])

    def test_missing_or_duplicated_rollouts_rejected(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaisesRegex(ValueError, 'rollouts'):
                validate_rows(rows, self.summaries, self.config, self.source)

    def test_corrupt_entropy_scores_rejected(self):
        rows = list(self.rows)
        rows[0] = copy.deepcopy(rows[0])
        rows[0]['input_terms_bits'][0] += 1
        with self.assertRaisesRegex(ValueError, 'Entropy differences'):
            validate_rows(rows, self.summaries, self.config, self.source)

    def test_early_stop_cannot_be_imputed_as_zero(self):
        rows = list(self.rows)
        index = next(i for i, r in enumerate(rows) if not r['b_complete'])
        rows[index] = dict(rows[index], injection_bits_per_token=0.0)
        with self.assertRaisesRegex(ValueError, 'Incorrect fixed mean'):
            validate_rows(rows, self.summaries, self.config, self.source)

    def test_incorrect_question_mean_rejected(self):
        summaries = list(self.summaries)
        summaries[0] = dict(summaries[0], injection_mean=summaries[0]['injection_mean'] + 0.1)
        with self.assertRaisesRegex(ValueError, 'Incorrect injection_mean'):
            validate_rows(self.rows, summaries, self.config, self.source)

    def test_holm_family_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])


if __name__ == '__main__':
    unittest.main()
