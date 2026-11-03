# Screened GSM8K correction dataset

This directory contains the exact 50-row Parquet input used for the sampled
checkpoint run, plus the selection and review records from
[contribution #7](https://github.com/bencalvert04/Plasticity-and-Empowerment/pull/7)
at commit `0d9c86fe1fc855412979287e8db3140c4ca58124`.

Questions come from the GSM8K test split in seeded random order. Selection
required a sampled Qwen2.5-0.5B-Instruct response longer than 50 non-EOS tokens.
Most screening stopped at 51 tokens, so these lengths are lower bounds. This is
a length-conditioned sample, not an unbiased benchmark or a guarantee that new
models or new samples will reach 20 tokens. `selected.json`, `screening.jsonl`,
and Parquet metadata preserve screening seeds, model revision, and source hash.

`cases.json` contains human-readable questions, corrections, original/revised
references, calculations, and review notes. [review.md](review.md) explains each
correction and source issue. In particular, `test:1309` has an incorrect original
numeric reference and `test:255` has inconsistent original stall counts. Original
reference values are preserved, not silently fixed.

The current runner reads `id`, `question`, `injection`, and the fixed timing
`injection_after_tokens=20`. Other fields are retained as provenance. It generates
fresh A tokens and never reuses the saved screening prefix. Neither reference
answer nor the revised calculation enters the model context. This run does not
grade answers.

```python
import pyarrow.parquet as pq
rows = pq.read_table('data/gsm8k50/gsm8k50_injections.parquet').to_pylist()
```

## Parquet columns

- `question`: unchanged GSM8K question.
- `answer`: original numeric GSM8K reference, preserved verbatim.
- `injection`: a short correction changing a specific quantity.
- `injection_after_tokens`: 20 generated assistant tokens.
- `revised_answer`: manually derived answer after applying the correction.
- Additional columns retain IDs, derivations, review notes, seeds, screening
  lengths/completion status, and exact first-20-token prefixes.

## Rebuild scripts

The screening and export scripts from contribution #7 live in [scripts/](scripts).
They are not needed to consume or verify the frozen dataset.

- `python3 data/gsm8k50/scripts/export_parquet.py` rebuilds
  `gsm8k50_injections.parquet` from `cases.json` and checks a round-trip,
  including metadata. It reproduces the committed file byte-for-byte.
- `scripts/screen.py` reproduces or resumes screening. It imports `core`, the
  retired answer-level runner (see [cleanup notes](../../docs/CLEANUP.md)), which
  is no longer in this branch; check it out from Git history to run screening.
  It appends to `screening.jsonl` and overwrites `selected.json` in this
  directory, so run it on a copy to preserve the original records.
