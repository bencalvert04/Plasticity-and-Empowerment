# Plasticity and Empowerment

Experiments on how an initial prompt changes a language model's next-token entropy.

- `directed_information_convergence_to_zero.py`: runs Pythia-70m on 20 fixed prompts and plots the mean per-token entropy change. The current output limit is 125 tokens, with early stopping at EOS.
- `effect_of_prompt_length_on_directed_information.py`: uses five fixed long prompts and prints and plots averages for input lengths from 5 through 49 tokens in steps of 2. Each average is the mean summed entropy change across the five prompt prefixes of that length. Every run uses the same 125-token output limit, with early stopping at EOS. Input lengths exclude BOS.

```sh
python3 -m pip install -r requirements.txt
python3 directed_information_convergence_to_zero.py
python3 effect_of_prompt_length_on_directed_information.py
```

The first run downloads the model. Generation currently uses greedy decoding on CPU with float64 parameters. A commented sampling alternative is included. The graph averages only positions reached by every response.

At each output position, the score is:

```text
H(next token | BOS, previous output)
  - H(next token | BOS, prompt, previous output)
```

Entropies are in bits. Despite the experiment names, this is prompt-induced entropy reduction, not a formal directed-information estimator or a proof of convergence to zero.

### Embedding-based question selection

`cluster_gsm8k.py` embeds all 8,792 training and test questions using
`sentence-transformers/all-MiniLM-L6-v2`, then applies KMeans to normalized
embeddings with seed 0 and 10 initializations. It selects the nearest member of
each cluster to its center and saves the questions, reference answers, IDs,
cluster sizes and weights to `gsm8k_clustered_50.json` by default.
Only question text is embedded, using the embedding model’s default token limit.

The GSM8K experiment now uses random sampling only. The clustering script
and map remain available as standalone exploratory tools:

```sh
python3 -m pip install -r requirements-clustering.txt
python3 cluster_gsm8k.py --clusters 50
```

Cluster representatives are not an unbiased random sample, and semantic
coverage does not guarantee coverage of mathematical difficulty.

### Interactive UMAP map

Open `gsm8k_umap.html` in a browser to explore all 8,792 questions. The 50
saved representatives are gold diamonds. Hover to read questions, scroll to
zoom, drag to pan, and use the legend or Selected only button to filter.
The HTML includes Plotly and works offline after downloading.

To rebuild the map from the saved selection:

```sh
python3 -m pip install -r requirements-visualization.txt
python3 visualize_gsm8k.py --selection gsm8k_clustered_50.json
```

The map uses question embeddings and original saved cluster memberships,
with UMAP seed 0, cosine distance, 15 neighbors, and minimum distance 0.1.
For compact selection files without memberships, KMeans is rerun with the
saved seed. The selected IDs remain those in the selection file.
Projection coordinates and selected flags are also saved in `gsm8k_umap.json`.
UMAP is a distorted 2D view, not an exact topology or a measure of difficulty.

### Optional final numerical-answer scores

```sh
python3 gsm8k_directed_information.py --sample 10 --trials 5 --answer-only-scores
```

This adds two per-response means while retaining all original full-response
outputs and leaving the prompt and sampling unchanged. It selects the original
generated tokens overlapping the final numerical answer and averages each of
the existing score arrays over those positions. For three answer tokens the
denominator is three, not the length of the whole response. EOS and explanation
tokens are excluded. A token containing both numeric text and adjacent characters
is indivisible and is counted once. Signs, decimals, and grouping punctuation
within the number are included.

Extraction conservatively checks the last nonempty line for a boxed number,
an explicit `####` or `answer is/:/=` marker, a standalone numeric answer, or a concluding Thus/Therefore/Hence/So sentence
with an "is" clause and exactly one numeric value. Single-number conclusions
using result verbs such as "runs", "equals", or "totals" are also supported.
This is heuristic extraction, not semantic answer verification.
Missing/ambiguous answers, unstable token alignment, and responses without EOS
are reported as unavailable, never zero. Some valid prose answer formats will
therefore be missed. Extraction does not check mathematical correctness.

The input score removes the question while retaining all preceding output.
The history score retains the question and compares the original question-only
entropy to the entropy with the entire preceding output (including any earlier
digits of the answer). It does not isolate only the explanation before the
answer. These remain next-token vocabulary entropy differences evaluated at
answer positions, not information carried by the realized numeral, formal DI,
or established plasticity/empowerment estimates. No new cross-trial maximum or
dataset aggregate is introduced.

Dependency-free extraction/alignment tests: `python3 -m unittest test_answer_token_scores.py`.

### Optional Jensen–Shannon comparisons and saved records

```sh
.venv/bin/python gsm8k_directed_information.py --sample 10 --trials 3 \
  --answer-only-scores --js-scores --max-new-tokens 256 --seed 42 \
  --output ../output/answer-js-run-01
```

No follow-ups, hints, corrections, or correctness grading are added. The original
question and generation procedure are retained. JS compares exactly the same
context pairs as the original entropy proxies: question+history versus empty
question+history, and question+history versus question alone. It uses the full
vocabulary distributions, in float32 and base-2 units, at each generated token.
JS is symmetric and bounded by 1 bit per position; it is not directed information
or an established empowerment measure. The history comparison retains the
original first-token-distribution baseline, including its position mismatch.

With both scoring flags, each JS score is averaged over the same extracted
numerical-answer token positions as the answer entropy score. Whole-response
means include EOS like the original code; answer means exclude EOS. Missing or
truncated answers are unavailable rather than zero. No independent generations
are aligned or compared.

`--output` writes metadata.json (settings, sampled question IDs/text, model
revision) and incrementally appends trials.jsonl (response, token IDs, EOS status,
per-token entropy/JS values, and answer averages). Existing output directories
are refused. Saved records have no correctness labels. `--seed` controls random
sampling but does not guarantee bitwise GPU reproducibility. The token cap is
optional and does not change the original uncapped default. Earlier validation
smoke-result files outside this repository remain archived as historical runs;
they are not outputs of this simplified experiment.

### Mid-response injection experiment

```sh
.venv/bin/python injection_experiment.py --pause-tokens 32 \
  --max-new-tokens 256 --trials 1 --seed 42 --output ../output/injection-run-01
```

This is a separate runner. It samples a first assistant span, pauses after exactly
`--pause-tokens` generated tokens, closes that assistant turn, and supplies the
injection in a new user turn. Qwen then generates a second assistant span.
If EOS occurs before or at the pause boundary, the trial is recorded as skipped:
it does not inject after an already completed response. The continuation cap is
independent of the first-span budget. No answer-format instruction is added to
the original question; a common continuation/answer-format instruction appears
in every follow-up context.

At each continuation position, three independent KV caches evaluate:

1. Full: question + first span + injection and continuation instruction.
2. Without injection: same conversation with only the injected content removed;
   the user turn and continuation instruction remain.
3. Without first span: same conversation with the first assistant content empty;
   the assistant turn and injected user content remain.

Only the full context generates tokens. All three caches receive exactly those
same continuation tokens, at their own context positions. First-span token IDs
are preserved directly, not decoded and re-tokenized. Empty-turn ablations keep
the chat roles/boundaries, but context lengths/positions still differ. Metrics
therefore include these positional/context effects; they are not isolated causal
estimates. The injection experiment is a turn interruption, not insertion of
user text into an assistant message.

Two comparisons are reported: **injection sensitivity** and **first-span
sensitivity**. Each has signed entropy reduction (ablated minus full) and
symmetric Jensen–Shannon divergence over next-token distributions. Full-span
means exclude EOS. Final-numerical-answer means select only the numerical tokens
in the continuation, using the existing conservative extractor; truncated or
unparsed answers have unavailable/null answer scores. Per-token arrays retain
EOS scores when EOS is sampled, so recompute full-span means excluding EOS.
These are context-sensitivity proxies, not formal DI or established plasticity
or empowerment. Earlier output affecting later output is not by itself agent
control over environmental observations.

Supply reviewed cases with `--cases my_cases.json`:

```json
{"cases": [{"id": "case-1", "question": "...", "injection": "Correction: ..."}]}
```

Every case is used unless `--sample N` selects a seeded random subset. The bundled
case is a hand-authored number-switch demonstration adapted from the GSM8K sprint
example (60 becomes 80; expected answer 540 becomes 720). Optional reference
fields are recorded as metadata only: no correctness grading is performed.
Numerical switches must be written and checked by a researcher, not guessed by
automatically replacing every matching digit. Different conditions, neutral
injection controls, and matched injection lengths may be needed for a research
study; this runner does not add those experimental branches.

`metadata.json` saves cases, versions, model revision, and settings.
`trials.jsonl` saves both generated spans, exact token IDs for all comparison
contexts, injection text, EOS/truncation/skip status, per-token scores, and span
and answer averages. Existing output directories are never overwritten. Seeds
are assigned per trial; reproducibility across GPU/software versions is not
guaranteed. Treat output as a pilot, not evidence of scientific validity.

Tests: `.venv/bin/python -m unittest test_injection_experiment.py test_js_scores.py test_answer_token_scores.py`.
