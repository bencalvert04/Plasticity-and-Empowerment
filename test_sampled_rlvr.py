import unittest
from types import SimpleNamespace
import torch
from sampled_rlvr_experiment import rollout, flatten, summarize, sample_span


class Tokenizer:
    eos_token_id = 0
    def apply_chat_template(self, messages, **kwargs):
        ids = []
        for message in messages:
            ids += [1 if message['role'] == 'user' else 2]
            ids += [3] if message['content'] else []
            ids += [4]
        return ids + [2]
    def decode(self, tokens, **kwargs):
        return str(tokens)


class Model:
    device = 'cpu'
    config = SimpleNamespace(max_position_embeddings=1000)
    generation_config = SimpleNamespace(eos_token_id=[0])
    def __init__(self, stop=None):
        self.calls = []
        self.stop = stop
    def __call__(self, ids, past_key_values=None, **kwargs):
        context = (past_key_values or []) + ids[0].tolist()
        self.calls.append(context)
        # A has one user turn; B has two. Content tokens generated are always 5.
        is_b = context.count(1) == 2
        stop = self.stop == 'a' or (self.stop == 'b' and is_b)
        logits = torch.full((1, 1, 6), -100.0)
        logits[0, 0, 0 if stop else 5] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=context)


class Tests(unittest.TestCase):
    def test_context_sensitive_sampling_and_cache_equivalence(self):
        class Sensitive(Model):
            def __call__(self, ids, past_key_values=None, **kwargs):
                context = (past_key_values or []) + ids[0].tolist()
                state = sum((i + 1) * t for i, t in enumerate(context))
                logits = torch.tensor([-100.0, -100.0, -100.0, -100.0,
                                       (state % 7) / 3, (state % 11) / 4]).reshape(1, 1, 6)
                return SimpleNamespace(logits=logits, past_key_values=context)
        cached = rollout(Sensitive(), Tokenizer(), 'question', 'correction', 12)
        uncached = rollout(Sensitive(), Tokenizer(), 'question', 'correction', 12, cached=False)
        self.assertEqual(cached, uncached)
        self.assertTrue(any(abs(t) > 0.001 for t in cached['input']['terms_bits']))
        control = rollout(Sensitive(), Tokenizer(), 'question', '', 12)
        self.assertEqual(control['injection']['terms_bits'], [0.0] * 20)

    def test_contexts_and_cache(self):
        tokenizer = Tokenizer()
        model = Model()
        result = rollout(model, tokenizer, 'question', 'correction', 1)
        self.assertEqual(result['status'], 'complete')
        prefix = [1, 3, 4, 2]
        a = [5] * 20
        for t in range(20):
            self.assertEqual(model.calls[2*t], prefix + [5]*t)
            self.assertEqual(model.calls[2*t+1], [1,4,2] + [5]*t)
            for side, (span, message) in enumerate(((a,[3]),(a,[]),([],[3]))):
                self.assertEqual(model.calls[40+3*t+side], prefix+span+[4,1]+message+[4,2]+[5]*t)
        uncached = rollout(Model(), tokenizer, 'question', 'correction', 1, cached=False)
        self.assertEqual(result, uncached)

    def test_early_a_no_injection_or_zero_imputation(self):
        result = rollout(Model('a'), Tokenizer(), 'question', 'correction', 2)
        self.assertEqual(result['status'], 'a_early_eos')
        self.assertIsNone(result['b'])
        self.assertEqual(result['a']['content_token_count'], 0)
        self.assertIsNone(result['input']['bits_per_token'])
        self.assertEqual(result['input']['scored_positions'], 1)
        self.assertIsNone(result['injection']['observed_total_bits'])

    def test_early_b_and_summary_counts(self):
        tok = Tokenizer()
        records = []
        for stop in (None, 'a', 'b'):
            r = flatten(rollout(Model(stop), tok, 'question', 'correction', 3),tok)
            records.append(dict(r, id='x', question='q', injection='j', source_row=0,
                                rlvr_step=0, model_repo='fake', model_revision='fake'))
        summary = summarize(records,3)[0]
        self.assertEqual(summary['input_n'],2)
        self.assertEqual(summary['injection_n'],1)
        self.assertEqual(summary['empowerment_n'],1)
        self.assertEqual(summary['input_joint_complete_n'],1)
        self.assertEqual(summary['injection_observed_n'],2)
        self.assertEqual(summary['a_early_stop_rate'],1/3)
        self.assertEqual(summary['b_early_stop_rate_given_reached'],0.5)
        self.assertTrue(summary['aggregation_complete'])

    def test_unmasked_distribution(self):
        class Uniform(Model):
            def __call__(self, ids, **kwargs):
                return SimpleNamespace(logits=torch.zeros((1,1,6)),past_key_values=None)
        span = sample_span(Uniform(), [[1]], {0}, torch.Generator().manual_seed(0))
        self.assertAlmostEqual(span['entropies'][0][0], 2.5849625, places=5)
        self.assertAlmostEqual(span['eos_probabilities'][0][0], 1/6, places=6)


if __name__ == '__main__':
    unittest.main()
