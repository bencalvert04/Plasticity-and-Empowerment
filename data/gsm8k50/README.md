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

The original screening/export scripts remain in the linked contribution's Git
history; they are not needed to consume or verify the frozen dataset here.
