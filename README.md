# Plasticity and Empowerment: small experiments

We ask how a prompt, earlier output, or a correction changes a language model's
next-token distribution. These are context-sensitivity experiments, not formal
estimates of directed information, plasticity, or empowerment.

## Read the code in this order

1. `core.py`: load Qwen, predict the next token, compare contexts, repeat.
2. `experiments.py`: choose prompts and run the five experiments below.
3. `explore.py`: optional embedding clusters and a simple UMAP scatter plot.

There is one generation loop. It samples a token from the full context, then
appends that same token to every ablated context. Each context has an independent
KV cache, so its prefix is processed once and later steps process only the newest
token. Probabilities stay on the model device, and only the last position's logits
are computed. Model loading happens only when an experiment starts.

## What we measure

Every score is `[entropy reduction, Jensen–Shannon divergence]`, in bits:

```
entropy reduction = H(baseline) - H(full)
JS = H((baseline + full)/2) - (H(baseline) + H(full))/2
```

Entropy reduction can be negative. JS is symmetric and lies between 0 and 1.
Two distributions can have equal entropy yet predict entirely different tokens.

| Score | Baseline compared with the full context |
|---|---|
| input | Empty user message, followed by the same output prefix |
| history | Original question only, fixed at its first-token distribution |
| injection | Same interrupted conversation, but empty correction content |
| first_span | Same corrected conversation, but empty first assistant span |

The injection experiment pauses after a uniformly random 1–32 tokens and inserts a
new user turn. All three conditions retain the same message roles and final-answer
instruction. The original first-span token IDs are preserved exactly.
If the first response ends before the pause, the injection is skipped.
`--pause-max` changes the upper bound; `--seed` seeds the random selection.
Only the first 10 continuation tokens are scored. `window_means` contains the
arithmetic mean of each metric across those 10 tokens, and `pause` records the
chosen interruption point. If EOS occurs within the window, means are null;
`scored_tokens` records the available non-EOS count. Skipped/short runs are not
zero scores and should be counted separately when aggregating trials.

## Run

```sh
python3 -m pip install -r requirements.txt
python3 experiments.py prompt --prompt 'What is 12 times 15?' --plot
python3 experiments.py position --plot
python3 experiments.py length --plot
python3 experiments.py gsm8k --sample 10 --trials 3 --tokens 256 > results.json
python3 experiments.py injection --pause-max 32 --trials 5 > injection.json
python3 -m unittest -v
```

`position` averages prompt-removal scores by output position over 20 fixed prompts.
Only positions reached by every response are plotted. `length` truncates each of
five fixed prompts to 5, 7, ..., 49 tokens and plots mean summed prompt-removal
scores. Both use greedy decoding. Edit the original prompt sets in `prompts.json`.

`gsm8k` randomly samples distinct questions from the combined official train and
test splits. Only questions go to the model. `prompt`, `gsm8k`, and `injection`
sample the full softmax at temperature 1. Change correction examples in
`injection_cases.example.json`. Reference answers are not used for grading.

All runs use Qwen2.5-0.5B-Instruct, Apple MPS when available, otherwise CPU.
The first run downloads the model. `--seed` defaults to 42; this does not promise
bit-for-bit reproducibility across devices. `--tokens` defaults to 125 and caps
non-injection generation. Injection always uses a 10-token continuation window.
Start with `--tokens 8` for a quick non-injection execution check.

JSON output contains settings, text, exact generated token IDs, completion status,
per-token scores (including EOS), and numerical-answer means. A capped response
has `complete: false`; its answer means are null. Only a completed response ending
with `Final answer: 720` (signed integers or decimals) is recognized. Unexpected
formats and unstable decoded token boundaries produce no answer score. Answer
means include tokens overlapping the number, excluding explanation and EOS.
Scores are computed before sampling the corresponding token.

## Optional question coverage

```sh
python3 -m pip install -r requirements-visualization.txt
python3 explore.py
```

This embeds GSM8K questions with MiniLM, runs 50-cluster KMeans, selects the nearest
member to each center, and saves `representatives.json` and `clusters.png`.
UMAP is only a distorted 2D view. Cluster representatives are not an unbiased
sample or a measure of mathematical difficulty.

## Deliberate simplifications

The old script names and APIs are replaced by the commands above. Optional scoring
flags, unlimited generation, flexible answer extraction, HTML
hover controls, and multi-file result directories are removed. Generated maps
and dataset copies are no longer tracked. Both metrics are always calculated;
use JSON arrays for totals or other summaries. No correctness grader is included.

Context ablations have different lengths and are not marginalization over possible
histories. Prompt length can also change output length, so summed scores mix both
effects. A downward curve does not prove convergence to zero.
