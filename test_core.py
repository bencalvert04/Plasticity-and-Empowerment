import unittest
import torch
from core import Experiment, answer_indices, compare, entropy


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return ''.join({0: 'Final answer: ', 1: '7', 2: ''}[t] for t in tokens)


class Tests(unittest.TestCase):
    def experiment(self, distribution):
        exp = Experiment.__new__(Experiment)
        exp.tokenizer, exp.eos = Tokenizer(), {2}
        self.seen = []
        def predict(ids):
            self.seen.append(ids.copy())
            return torch.tensor(distribution)
        exp.predict = predict
        return exp

    def test_metrics(self):
        p, q = torch.tensor([1., 0.]), torch.tensor([0., 1.])
        self.assertEqual(float(entropy(p)), 0)
        self.assertEqual(compare(p, p), [0, 0])
        self.assertEqual(compare(p, q), [0, 1])
        self.assertEqual(compare(p, q), compare(q, p))
        self.assertAlmostEqual(compare(p, (p + q) / 2)[0], 1)
        self.assertAlmostEqual(compare((p + q) / 2, p)[0], -1)

    def test_shared_history_and_truncation(self):
        exp = self.experiment([0., 1., 0.])
        result = exp.run({'full': [10], 'input': [20]}, 2, history=True)
        self.assertEqual(self.seen, [[10], [10], [20], [10, 1], [20, 1]])
        self.assertEqual(result['tokens'], [1, 1])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['answer_means']['input'])
        self.assertEqual(result['scores']['history'][0], [0, 0])

    def test_eos_and_invalid_limit(self):
        exp = self.experiment([0., 0., 1.])
        result = exp.run({'full': [10]}, 4)
        self.assertEqual(result['tokens'], [2])
        self.assertTrue(result['complete'])
        with self.assertRaises(ValueError):
            exp.run({'full': [10]}, 0)
        exp.chat = lambda *args: [10]
        self.assertIn('skipped', exp.inject('Question', 'Correction'))

    def test_answer_alignment_and_ambiguity(self):
        tok = Tokenizer()
        self.assertEqual(answer_indices(tok, [0, 1, 2], 'Final answer: 7'), [1])
        for text in ['The intermediate value is 7', 'Final answer: 7 or 8', 'Final answer: 7/8']:
            self.assertEqual(answer_indices(tok, [0, 1, 2], text), [])

    def test_completed_answer_mean_uses_only_numeric_token(self):
        exp = self.experiment([0., 1., 0.])
        def predict(ids):
            if ids[0] == 20:
                return torch.tensor([1/3, 1/3, 1/3])
            return torch.eye(3)[len(ids) - 1]
        exp.predict = predict
        result = exp.run({'full': [10], 'input': [20]}, 5)
        self.assertEqual(result['answer_indices'], [1])
        self.assertEqual(result['answer_means']['input'], result['scores']['input'][1])
        self.assertTrue(result['complete'])

    def test_injection_preserves_exact_tokens_and_turns(self):
        exp = self.experiment([0., 1., 0.])
        exp.chat = lambda *texts: [10] if len(texts) == 1 else [10, 20, len(texts[-1]), 30]
        exp.inject('Question', 'Correction', pause=1, limit=2)
        full, no_injection, no_span = self.seen[1:4]
        self.assertEqual(full[:2], [10, 1])
        self.assertEqual(no_injection[:2], [10, 1])
        self.assertEqual(no_span, full[:1] + full[2:])
        self.assertNotEqual(full, no_injection)
        self.assertEqual(self.seen[4:], [ids + [1] for ids in [full, no_injection, no_span]])


if __name__ == '__main__':
    unittest.main()
