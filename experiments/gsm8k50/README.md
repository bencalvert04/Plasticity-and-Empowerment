# First injection experiment

`cases.json` contains 50 distinct GSM8K test questions and one short correction per question. Selection uses `random.Random(42).sample(range(1319), 50)` in that order. `source_records.json` preserves the original questions and solutions; `review.md` pairs each question with its correction and revised numerical reference.

The test split is kept separate from training data. Once used for experiment development, these 50 should be considered a development subset rather than an untouched final evaluation set.

From the repository root, run all 50 cases:

```sh
.venv/bin/python experiments.py injection \
  --cases experiments/gsm8k50/cases.json \
  --pause-max 20 \
  --trials 1 \
  --seed 42 > gsm8k50-results.json
```

Use a new result filename for each run to avoid overwriting earlier results. No `--sample` is necessary: all supplied injection cases run. `--trials` repeats responses with the same correction.

The current main-branch runner chooses a random interruption between 1 and `--pause-max` generated tokens (not words). Thus `--pause-max 20` does NOT mean a fixed 20-token interruption. It scores a 10-token continuation window, rather than generating a complete revised solution. These existing runner settings are unchanged by this dataset addition. A fixed pause and full-answer completion would require a separate runner change.

If Qwen finishes before the pause, the runner skips the injection. The correction is sent as a new user turn after the partial assistant turn; earlier tokens are retained. Only question and injection text are sent from each case. Reference answers and review notes are metadata, not model inputs. References are manually derived review aids; automatic correctness grading is not implemented.

## Source issues

- `test:1309`: supplied answer is 2280, but the original question totals 2180. Revised answer is 2380.
- `test:255`: original wording conflicts between ten and twenty stalls. The correction resolves this to twenty. This case differs from an ordinary number switch.
- `test:696`: follows the source's interpretation of “3 times more” as three times as many and Marta/Martha as one person.
- `test:161`: follows the source in counting all listed creatures, including starfish.

The random sample was not silently changed to remove problematic cases. Review the flagged cases and report how they are handled before interpreting aggregate results. Entropy differences and Jensen–Shannon scores remain context-sensitivity proxies, not formal directed information, plasticity capacity, or empowerment.
