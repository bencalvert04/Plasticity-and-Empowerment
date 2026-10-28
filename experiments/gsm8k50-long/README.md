# Screened GSM8K injection dataset

50 test questions screened using Qwen2.5-0.5B-Instruct. All 50 sampled responses exceeded 50 non-EOS tokens. Most screening stopped at 51 tokens, establishing a lower bound rather than a complete response length. A fresh generation may be shorter. This is a length-conditioned sample, not a guarantee of accuracy or representative coverage.

## Parquet columns

- `question`: unchanged GSM8K question.
- `answer`: original numeric GSM8K reference, preserved verbatim.
- `injection`: a short correction changing a specific quantity.
- `injection_after_tokens`: 20 generated assistant tokens.
- `revised_answer`: manually derived answer after applying the correction.
- Additional columns retain IDs, derivations, review notes, seeds, screening lengths/completion status, and exact first-20-token prefixes.

Read `review.md` before grading: test:1309 has an incorrect source reference (2280 instead of 2180); test:255 has conflicting stall counts. These are flagged, not silently replaced. Other interpretation notes are included.

## Load and export

Install `pyarrow` in your Python environment, then:

```python
import pyarrow.parquet as pq
cases = pq.read_table('experiments/gsm8k50-long/gsm8k50_injections.parquet').to_pylist()
```

`export_parquet.py` rebuilds the Parquet from `cases.json` and verifies a round-trip including metadata. Run from the repository root:

```sh
python experiments/gsm8k50-long/export_parquet.py
```

`screening.jsonl` stores screening responses and tokens; `selected.json` stores selection provenance, including model revision and source hash. `screen.py` reproduces or resumes screening and additionally requires `certifi`. It writes screening files in this directory, so use a separate copy to preserve the original records when reproducing.

## Integration

This PR supplies data only. The existing runner does not read Parquet or implement a fixed 20-token pause: its `--pause-max` is random, and it scores only 10 continuation tokens. Your consumer must use `injection_after_tokens` to interrupt at exactly token 20. The column alone does not change runner behavior. Original and revised answers are evaluation metadata; do not include them in model prompts. Saved prefixes enable a future replay experiment; the existing runner does not replay them automatically.
