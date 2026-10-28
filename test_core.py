import unittest
from unittest.mock import patch
import torch
from core import ANSWER, Experiment, answer_indices, answer_match, compare, entropy


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return ''.join({0: 'Final answer: ', 1: '7', 2: ''}[t] for t in tokens)


class Tests(unittest.TestCase):
    def test_real_model_caches_match_full_context(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        exp = Experiment.__new__(Experiment)
        exp.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=16)).eval()
        contexts = [[3, 4], [5]]
        with torch.inference_mode():
            states = [exp.predict(ids) for ids in contexts]
            self.assertIsNot(states[0][1], states[1][1])
            for token in [6, 7, 8]:
                for ids, (p, cache) in zip(contexts, states):
                    expected = exp.model(torch.tensor([ids]), use_cache=False).logits[0, -1].softmax(-1)
                    torch.testing.assert_close(p, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(cache.get_seq_length(), len(ids))
                    ids.append(token)
                states = [exp.predict([token], cache) for _, cache in states]
            with self.assertRaises(ValueError):
                exp.predict([1] * 16, states[0][1])

    def experiment(self, distribution):
        exp = Experiment.__new__(Experiment)
        exp.tokenizer, exp.eos = Tokenizer(), {2}
        self.seen = []
        def predict(ids, cache=None):
            ids = (cache or []) + ids
            self.seen.append(ids.copy())
            return torch.tensor(distribution), ids
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
        self.assertEqual(self.seen, [[10], [20], [10, 1], [20, 1]])
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

    def test_units_and_ambiguous_answers(self):
        for text, number in [('Final answer: 395 food items.', '395'),
                             ('Final answer: $1,200.50', '1,200.50'),
                             ('Thus:\n\\[\n\\boxed{395}\n\\]', '395'),
                             ('Final answer: -2.5 meters', '-2.5')]:
            self.assertEqual(answer_match(text)['number'], number)
        for text in ['Final answer: 3 or four', 'Final answer: 7/8',
                     'Final answer: 3 and 4', 'Final answer: 1e3', 'Intermediate: 3']:
            self.assertIsNone(answer_match(text))

    def test_reasoning_removed_but_answer_prefix_retained(self):
        exp = self.experiment([0.1, 0.8, 0.1])
        exp.tokenizer.decode = lambda ids, **kw: ''.join(
            {0: 'Reasoning.\n', 1: 'Final answer: ', 2: '3', 3: '9', 4: ''}[i] for i in ids)
        exp.chat = lambda *texts: [10]
        result = dict(tokens=[0, 1, 2, 3, 4], text='Reasoning.\nFinal answer: 39',
                      answer_indices=[2, 3], answer_means={})
        extra = exp.reasoning_answer('Q', result)
        self.assertEqual(self.seen, [[10, 0, 1], [10, 1], [10, 0, 1, 2], [10, 1, 2]])
        self.assertEqual(extra['reasoning_end_token'], 1)
        self.assertEqual(result['answer_means']['reasoning'], [0, 0])

    def test_input_ablation_keeps_answer_instruction(self):
        exp = self.experiment([0.1, 0.8, 0.1])
        exp.chat = lambda text: text
        result = dict(answer_indices=[], answer_means={'input': None, 'history': None})
        with patch.object(exp, 'run', return_value=result) as run:
            exp.prompt('Q' + ANSWER)
        self.assertEqual(run.call_args.args[0], {'full': 'Q' + ANSWER, 'input': ANSWER})
        self.assertNotIn('history', result['answer_means'])

    def test_completed_answer_mean_uses_only_numeric_token(self):
        exp = self.experiment([0., 1., 0.])
        def predict(ids, cache=None):
            ids = (cache or []) + ids
            if ids[0] == 20:
                return torch.tensor([1/3, 1/3, 1/3]), ids
            return torch.eye(3)[len(ids) - 1], ids
        exp.predict = predict
        result = exp.run({'full': [10], 'input': [20]}, 5)
        self.assertEqual(result['answer_indices'], [1])
        self.assertEqual(result['answer_means']['input'], result['scores']['input'][1])
        self.assertTrue(result['complete'])

    def test_injection_preserves_exact_tokens_and_turns(self):
        exp = self.experiment([0., 1., 0.])
        exp.chat = lambda *texts: [10] if len(texts) == 1 else [10, 20, len(texts[-1]), 30]
        result = exp.inject('Question', 'Correction', pause_max=1)
        full, no_injection, no_span = self.seen[1:4]
        self.assertEqual(full[:2], [10, 1])
        self.assertEqual(no_injection[:2], [10, 1])
        self.assertEqual(no_span, full[:1] + full[2:])
        self.assertNotEqual(full, no_injection)
        self.assertEqual(self.seen[4:7], [ids + [1] for ids in [full, no_injection, no_span]])
        self.assertEqual(result['scored_tokens'], 10)
        self.assertEqual(len(result['continuation']['tokens']), 10)

    def test_random_pause_and_ten_token_mean(self):
        exp = self.experiment([0., 1., 0.])
        exp.chat = lambda *texts: [10]
        values = [[float(i), i / 10] for i in range(10)]
        first = dict(complete=False, tokens=[1] * 7)
        continuation = dict(tokens=[1] * 10, scores={'injection': values})
        with patch('core.random.randint', return_value=7) as draw, patch.object(
                exp, 'run', side_effect=[first, continuation]) as run:
            result = exp.inject('Q', 'Correction', pause_max=32)
        draw.assert_called_once_with(1, 32)
        self.assertEqual(run.call_args_list[0].args[1], 7)
        self.assertEqual(run.call_args_list[1].args[1], 10)
        self.assertEqual(result['pause'], 7)
        self.assertAlmostEqual(result['window_means']['injection'][0], 4.5)
        self.assertAlmostEqual(result['window_means']['injection'][1], 0.45)
        continuation = dict(tokens=[1, 2], scores={'injection': values[:2]})
        with patch.object(exp, 'run', side_effect=[first, continuation]):
            result = exp.inject('Q', 'Correction')
        self.assertEqual(result['scored_tokens'], 1)
        self.assertIsNone(result['window_means']['injection'])


if __name__ == '__main__':
    unittest.main()
