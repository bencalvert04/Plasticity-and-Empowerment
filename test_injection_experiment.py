import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from injection_experiment import continuation_contexts, first_span, scored_continuation, score_summary
from input_and_output_directed_information import jensen_shannon, entropy


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize, **kwargs):
        text = ''.join(f"<{m['role']}>" + m['content'] + '</turn>' for m in messages) + '<assistant>'
        return self.encode(text) if tokenize else text

    def decode(self, ids, **kwargs):
        return ''.join({1:'Final answer: ', 2:'7', 9:''}.get(i, '?') for i in ids)


class Cache:
    def __init__(self, ids):
        self.ids = ids
    def get_seq_length(self):
        return len(self.ids)


class Model:
    device = 'cpu'
    config = SimpleNamespace(max_position_embeddings=4096)
    generation_config = SimpleNamespace(eos_token_id=[9])

    def __init__(self):
        self.seen = []

    def __call__(self, x, past_key_values=None, **kwargs):
        ids = ([] if past_key_values is None else past_key_values.ids) + x[0].tolist()
        self.seen.append(ids[:])
        logits = torch.arange(10, dtype=torch.float32) * ((sum(ids) % 3) - 1)
        return SimpleNamespace(logits=logits[None,None,:], past_key_values=Cache(ids))


class InjectionTests(unittest.TestCase):
    def test_contexts_preserve_span_ids_and_empty_turns(self):
        tok = Tokenizer()
        # IDs need not decode to well-formed text: never decode/re-encode the span.
        c = continuation_contexts(tok, 'Question', [250, 251], 'Change 60 to 80')
        self.assertIn(250, c['full'])
        self.assertIn(250, c['without_injection'])
        self.assertNotIn(250, c['without_first_span'])
        self.assertEqual(c['full'], c['without_first_span'][:len(tok.apply_chat_template([{'role':'user','content':'Question'}],True))] + [250,251] + c['without_first_span'][len(tok.apply_chat_template([{'role':'user','content':'Question'}],True)):])
        self.assertNotEqual(c['full'], c['without_injection'])

    def test_shared_continuation_history_and_pre_sample_scores(self):
        contexts = {'full':[1], 'without_injection':[2], 'without_first_span':[3]}
        model = Model()
        with patch('injection_experiment.sample_token', side_effect=[2,9]):
            tokens, done, scores = scored_continuation(model, Tokenizer(), contexts, 5)
        self.assertEqual(tokens, [2,9])
        self.assertTrue(done)
        self.assertEqual(model.seen, [[1],[2],[3],[1,2],[2,2],[3,2]])
        for values in scores.values():
            self.assertEqual(len(values), 2)
        full = torch.arange(10,dtype=torch.float32)*0
        ablated = torch.arange(10,dtype=torch.float32)
        self.assertAlmostEqual(scores['injection_entropy_bits'][0], entropy(ablated)-entropy(full))
        self.assertAlmostEqual(scores['injection_js_bits'][0], jensen_shannon(ablated,full))

    def test_pause_and_early_eos(self):
        with patch('injection_experiment.sample_token', side_effect=[2,9]):
            tokens, done = first_span(Model(), Tokenizer(), 'Q', 4)
        self.assertEqual(tokens,[2,9]); self.assertTrue(done)
        with patch('injection_experiment.sample_token', return_value=2):
            tokens, done = first_span(Model(), Tokenizer(), 'Q', 3)
        self.assertEqual(tokens,[2,2,2]); self.assertFalse(done)

    def test_answer_averages_exclude_explanation_and_eos(self):
        scores={k:[10,2,100] for k in ('injection_entropy_bits','first_span_entropy_bits','injection_js_bits','first_span_js_bits')}
        summary=score_summary(Tokenizer(),[1,2,9],True,scores,{9})
        self.assertEqual(summary['answer'], '7')
        self.assertEqual(summary['answer_token_count'],1)
        self.assertEqual(summary['answer_means']['injection_entropy_bits_per_token'],2)
        self.assertEqual(summary['continuation_means']['injection_entropy_bits_per_token'],6)
        truncated=score_summary(Tokenizer(),[1,2,9],False,scores,{9})
        self.assertIsNone(truncated['answer_means']['injection_js_bits_per_token'])


if __name__ == '__main__':
    unittest.main()
