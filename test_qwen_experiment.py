"""Check the new dataset and exact-token Qwen3 framing without model weights."""
import json
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from sampled_rlvr_experiment import chat, continuation_prefix
from build_gsm8k200 import arithmetic
from analyze_checkpoint_statistics import analyze

ROOT = Path(__file__).resolve().parent


class QwenTests(unittest.TestCase):
    def test_training_steps_and_explicit_missing_pair_exclusions(self):
        rows = []
        for j, step in enumerate((0, 77, 251)):
            for i in range(4):
                rows.append(dict(id=f'q{i}', rlvr_step=step,
                                 **{m + '_mean': i + j*(i+1)*0.1 for m in ('input', 'injection', 'empowerment')}))
        rows[-1]['injection_mean'] = None
        with self.assertRaisesRegex(ValueError, 'missing groups'):
            analyze(rows)
        result = analyze(rows, allow_missing=True)
        self.assertEqual(result['omitted_question_ids_by_metric']['injection'], ['q3'])
        self.assertEqual(result['omitted_question_ids_by_metric']['input'], [])
        self.assertEqual(len(result['comparisons']), 9)
        self.assertEqual({r['rlvr_step'] for r in result['means']}, {0, 77, 251})
        for row in result['means']:
            self.assertEqual(row['n_questions'], 3 if row['metric'] == 'injection' else 4)

    def test_frozen_dataset_and_replacements(self):
        original = pq.read_table(ROOT / 'data/gsm8k50/gsm8k50_injections.parquet').to_pylist()
        rows = pq.read_table(ROOT / 'data/gsm8k200/gsm8k200_injections.parquet').to_pylist()
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({r['id'] for r in rows}), 200)
        for old, new in zip(original, rows):
            self.assertEqual(old, {k: new[k] for k in old})
        for row in rows[50:]:
            self.assertEqual(row['question'].count(row['replacement_old']), 1)
            self.assertEqual(row['revised_question'], row['question'].replace(row['replacement_old'], row['replacement_new'], 1))
            self.assertAlmostEqual(float(arithmetic(row['revised_answer_calculation'])), float(row['revised_answer']))
            self.assertIsNone(row['screening_seed'])

    def test_actual_qwen_templates_and_generated_token_splice(self):
        config = json.loads((ROOT / 'configs/qwen06b_gsm8k200.json').read_text())
        try:
            tokenizers = [AutoTokenizer.from_pretrained(repo, revision=rev, local_files_only=True)
                          for _, repo, rev in config['checkpoints']]
        except OSError:
            self.skipTest('Pinned tokenizer files not cached; run the model smoke test before production')
        reference = tokenizers[0]
        for tokenizer in tokenizers:
            self.assertEqual(tokenizer.get_vocab(), reference.get_vocab())
            for message in ('I meant 12 apples, not 8.', ''):
                mode = 'qwen3_nonthinking'
                full = chat(tokenizer, 'There are 8 apples.', chat_mode=mode)
                # Deliberately include token IDs that must never be decoded/re-encoded.
                generated = [123, 456, 789]
                prefix = continuation_prefix(tokenizer, 'There are 8 apples.', message, generated, mode)
                self.assertEqual(prefix[:len(full)], full)
                self.assertEqual(prefix[len(full):len(full)+3], generated)
                suffix = tokenizer.encode('<|im_end|>\n<|im_start|>user\n' + message +
                                          '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
                                          add_special_tokens=False)
                self.assertEqual(prefix[len(full)+3:], suffix)
                self.assertEqual(prefix, continuation_prefix(reference, 'There are 8 apples.', message, generated, mode))
                empty = continuation_prefix(tokenizer, 'There are 8 apples.', message, [], mode)
                self.assertEqual(empty, full + suffix)


if __name__ == '__main__':
    unittest.main()
