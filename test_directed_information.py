import math
import unittest

import torch

from directed_information import conditional_information, estimate


class Channel:
    eos = {2}

    def __init__(self):
        self.calls = []

    def predict(self, ids, cache=None):
        history = (cache or []) + ids
        self.calls.append((ids[:], history[:]))
        # The answer repeats X XOR R twice, then emits EOS.
        token = (history[0] - 10) ^ (history[1] - 20) if len(history) < 5 else 2
        return torch.eye(3)[token], history


class DirectedInformationTests(unittest.TestCase):
    def test_known_binary_channel(self):
        probabilities = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        actual = conditional_information(probabilities, torch.zeros(2))
        expected = 1 + 0.9 * math.log2(0.9) + 0.1 * math.log2(0.1)
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(conditional_information(probabilities[:1], torch.zeros(1)), 0)

    def test_chain_rule_and_posterior_update(self):
        channel = Channel()
        result = estimate(channel, [[10], [11]], [[20], [21]], [30], trials=8, limit=6)
        self.assertEqual(result['mean_di_bits'], [1, 1])
        for row in result['trials']:
            self.assertEqual(row['per_token_bits'], [[1, 1], [0, 0], [0, 0]])
            self.assertTrue(row['ended_with_eos'])
        # Both blocks have one bit, but repeated answer tokens add no new information.
        self.assertTrue(all(len(ids) == 1 for ids, history in channel.calls if len(history) > 3))

    def test_joint_prior_changes_conditional_information(self):
        result = estimate(Channel(), [[10], [11]], [[20], [21]], [30],
                          trials=4, prior=[[0.5, 0], [0, 0.5]])
        # Each block determines the other under this prior; conditional DI is zero.
        self.assertEqual(result['mean_di_bits'], [0, 0])

    def test_noisy_two_token_channel_matches_exact_information(self):
        class Noisy(Channel):
            def predict(self, ids, cache=None):
                history = (cache or []) + ids
                p = [0.9, 0.1, 0] if history[0] == 10 else [0.1, 0.9, 0]
                return torch.tensor(p), history
        result = estimate(Noisy(), [[10], [11]], [[20]], [30], trials=8, limit=2)
        h = lambda p: -p * math.log2(p) - (1-p) * math.log2(1-p)
        self.assertAlmostEqual(result['mean_di_bits'][0], 1 + h(0.18) - 2*h(0.1), places=6)
        self.assertEqual(result['mean_di_bits'][1], 0)

    def test_cap_and_validation(self):
        result = estimate(Channel(), [[10], [11]], [[20], [21]], [30], trials=1, limit=1)
        self.assertEqual(result['capped_trials'], 1)
        self.assertIsNone(result['monte_carlo_se_bits'])
        with self.assertRaises(ValueError):
            estimate(Channel(), [[10], [10]], [[20]], [30])
        with self.assertRaises(ValueError):
            estimate(Channel(), [[10]], [[20]], [30], prior=[[-1]])


if __name__ == '__main__':
    unittest.main()
