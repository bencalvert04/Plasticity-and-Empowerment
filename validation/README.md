# Method audit

Verdict: numerically working sensitivity measurements, not three validated measures
of plasticity or empowerment. Do not treat this pilot as a research result.

## Definitions and fixes

At answer position t, let X be the question plus the answer-format instruction,
R the generated reasoning, M the answer marker, and A_<t the already-generated
answer tokens. The corrected comparisons use:

- Full: p(next | X, R, M, A_<t).
- Input removed: p(next | format instruction, R, M, A_<t).
- Reasoning removed: p(next | X, M, A_<t).

Compute H(baseline) - H(full) and JS(full, baseline), then average separately over
numerical-answer tokens. These are full-vocabulary next-token statistics along
the observed response, not entropy of the final numerical value as a random
variable, conditional mutual information, or correlation coefficients.

The old fixed first-token baseline did not isolate reasoning-to-answer. It is now
excluded from answer means. Answer markers, numerical prefixes and formatting
instructions are retained in the appropriate comparisons. Cached decoding remains.
The parser accepts explicit final answers with units and boxed numbers, while
rejecting unsupported or ambiguous output. Missing scores remain null.

For injection, choose a random pause uniformly from 1 to 32 and average both
metrics across exactly ten continuation text tokens. Short windows remain null.
The two ablations remove correction content or the initial assistant span.

## Pilot evidence

Model: Qwen2.5-0.5B-Instruct on Apple MPS, float16 weights, float32 metrics.
Six GSM8K questions sampled with Python seed 42 from combined train/test data;
PyTorch generation seeds 42 through 47; output cap 384. Reference answers are
used only for this audit, never supplied to the model.

| Question ID | Generated tokens | Completed | Usable answer metrics |
|---|---:|---|---|
| train:1824 | 295 | yes | yes |
| train:409 | 384 | no | no |
| train:4506 | 101 | yes | no |
| train:4012 | 11 | yes | no |
| train:3657 | 90 | yes | no |
| train:2286 | 252 | yes | no |

Coverage was 1/6. The one parsed answer was 395, while the reference was 375.
Its input JS was 0.000136 bits and reasoning JS was 0.739788 bits. This illustrates
that high reasoning dependence does not imply correctness. Other responses included
unmarked prose, an equation, multiple quantities, and truncation. We did not guess
an answer or substitute zero for these cases.

Injection used the existing sprint example, three seeds (42, 43, 44), and matched
random pauses (8, 3, 27). Each seed tested an actual 60-to-80 correction, a reminder
that the value remains 60, and empty correction content.

| Condition | Full windows | Mean entropy difference | Mean JS |
|---|---:|---:|---:|
| Changed value | 3/3 | 0.640684 | 0.074705 |
| Unchanged-value reminder | 3/3 | 0.325736 | 0.075585 |
| Empty correction | 2/3 | 0 | 0 |

One empty-correction trial ended after eight text tokens and was excluded from
window averages. Its observed injection scores were also zero. Empty controls
and an independent identical-context real-model test returned zero differences.
These controls verify numerical behavior. They do not establish semantic validity:
the unchanged reminder had higher JS than the actual correction in two seeds.
The different conditions can generate different continuations, even with the same
seed. This is an exploratory comparison, not a controlled effect-size estimate.

Eleven unit tests pass, covering known metric values, cache equivalence, exact
context removals, answer prefixes, answer alignment, ambiguous formats, token
averaging, random pause bounds, early EOS and context limits.

## What remains before a research study

1. Establish a reliable answer stage with an explicit reasoning/answer boundary,
   or choose a model that follows the format consistently. Report coverage and
   correctness alongside all scores. Do not silently cherry-pick parsed answers.
2. Test multiple GSM8K correction cases with unchanged-fact and irrelevant-text
   controls. The current injection runner contains only one hand-authored case.
3. Validate whether scores predict adoption of the updated correct answer in a
   separate behavioral run. Keep the requested ten-token metric window fixed.
4. Define the claimed construct with the professor before naming these scores
   plasticity or empowerment. There is no universal threshold for either metric.

Intervening on a reasoning chain is an established way to investigate dependence,
but dependence varies by task and model; it is not automatically faithful reasoning
([Lanham et al., 2023](https://arxiv.org/abs/2307.13702)). Formal empowerment concerns
action-to-observation channel capacity, which these context-ablation scores do not
estimate ([Salge and Polani, 2017](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00025/full)).

## Reproduce

From the repository root:

```sh
python3 -m unittest -v
python3 -m validation.pilot
```

The pilot saves full local records to `validation/pilot_results.json` (gitignored).
It takes about a minute after loading on this machine. Device/kernel differences
can change sampled responses, so the table is a recorded pilot, not a guaranteed
future output. The script does not establish statistical significance.
