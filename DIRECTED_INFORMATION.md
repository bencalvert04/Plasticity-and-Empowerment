# Directed information between token blocks

Let X be the input block, R the reasoning block, and A the answer-stage tokens.
All X and R tokens precede A. With formatting held fixed, the two targets are:

```
input:     sum_t I(X; A_t | R, A_<t) = I(X; A | R)
reasoning: sum_t I(R; A_t | X, A_<t) = I(R; A | X)
```

This follows the directed-information chain construction when source blocks are
fully available before the target begins
([definition](https://web.stanford.edu/~tsachy/pdf_files/Directed%20Information%20and%20Causal%20Estimation%20in%20Continuous%20Time.pdf)).
These are observational information quantities for the specified joint law, not
automatically causal effects, plasticity, or empowerment.

## The source distribution is part of the measurement

One observed question, reasoning chain, and answer do not define a nontrivial
mutual-information estimate. Supply finite candidate question and reasoning blocks
and a joint prior over their pairings. The joint distribution used here is:

```
p(X, R, A) = supplied_prior(X, R) * model(A | X, R, fixed_answer_marker)
```

The example JSON contains two actual GSM8K questions and two exact recorded Qwen
reasoning-token blocks, cut before their final-answer markers. Both original
answers were wrong; this is a runnable demonstration, not a validation dataset.
The example prior is uniform over their Cartesian product. Some pairings are
deliberately inconsistent. This experimental distribution is not the natural
GSM8K joint distribution and should not be reported as such.

Change `joint_prior` to specify nonnegative weights with rows indexing questions
and columns indexing reasoning blocks. Omission means uniform Cartesian weights.
Zeros are supported. Duplicate token blocks are rejected rather than counted as
different source values. If each reasoning block uniquely identifies its question
under your prior, conditioning on R can leave no uncertainty about X; input DI
is then zero. The result depends materially on this scientific design choice.

## Calculation

For each Monte Carlo rollout, sample one (X, R) pair from the joint prior and keep
it fixed. Sample the answer from the model conditioned on that pair. At each step:

1. Evaluate next-token distributions for all candidate questions with R fixed,
   and for all candidate reasoning blocks with X fixed. Each has its own KV cache.
2. Marginalize with posterior weights conditional on the observed answer prefix.
3. Compute the weighted mean KL divergence to that mixture, in bits. This equals
   the local conditional mutual information, unlike a fixed full/deleted JS pair.
4. Feed the same sampled answer token to every cache, then update the source
   weights using Bayes' rule before evaluating the next position.

For the input comparison, for example:

```
w_i(t) = p(X=x_i | R=r, A_<t)
m_t(v) = sum_i w_i(t) * p(v | x_i, r, A_<t)
term_t = sum_i w_i(t) * KL(p(. | x_i, r, A_<t) || m_t)
w_i(t+1) proportional to w_i(t) * p(A_t | x_i, r, A_<t)
```

Source weights must update after each observed token. Resetting to uniform weights
would repeatedly count information already supplied by earlier answer tokens.
The finite-source marginalization is exact up to numerical precision; the average
over generated answer histories is a Monte Carlo estimate.

## Answer boundaries and output

The fixed marker `Final answer:` ends the reasoning stage. Everything generated
after that marker is the answer stage, including EOS. No successful-response or
numeric-parser filtering is performed: dropping samples based on their generated
answers would change the joint law. A model can still emit prose after the marker.
This estimates answer-stage DI, not DI restricted post hoc to extracted digits.

Generation is capped by `--answer-tokens`. EOS is included because stopping can
carry information; subsequent positions are implicitly absorbing and add zero.
If a run reaches the cap, the target is the capped answer sequence, not its unknown
full continuation. `capped_trials` makes this visible. Increase the cap if needed.

- `mean_di_bits`: Monte Carlo estimate of each conditional sequence DI.
- `monte_carlo_se_bits`: standard error across rollouts (null for one rollout).
- `source_entropy_bounds_bits` and `exceeds_source_bound`: population DI cannot
  exceed the source's conditional entropy. An estimate above that bound needs
  more sampling or investigation; it is not silently clipped. The reported sample
  standard error can miss rare trajectories in a small run and is not a guaranteed
  confidence interval.
- `mean_bits_per_token`: average of each rollout's total divided by its observed
  answer-stage length, including EOS. This length-normalized summary is not an
  additional conditional-MI identity.
- `metric_order`: input-to-answer given reasoning, then reasoning-to-answer given input.

Use `--details` to retain source blocks, exact target IDs, generated answer text,
per-token terms, and zero-based inclusive token ranges for input, reasoning,
fixed marker, and answer. Both metrics always use the same generated answer path.

```sh
python3 directed_information.py --sources di_sources.example.json --trials 32 --answer-tokens 32
python3 -m unittest -v
```

Tests check zero information for known/identical sources, exact one-bit transfer,
no double counting of repeated answer tokens, a two-token noisy channel against
its analytic MI, joint-prior conditioning, incremental caches, and stopping/caps.
Small Monte Carlo estimates are noisy and can exceed population bounds by sampling
error. This runner does not claim a population estimate from an arbitrary pool.

Recorded smoke check: 32 real Qwen/MPS rollouts with the included pool and a
32-token cap returned 0.38666 input bits and 1.03873 reasoning bits, with sample
standard errors 0.03166 and 0.03205. Each source-entropy bound is one bit, so the
reasoning estimate is explicitly flagged as above its population bound. One rollout
hit the cap. These results test execution and uncertainty reporting, not scientific
validity or convergence. Sixteen unit tests pass; analytic toy channels are used
to test estimator correctness independently of this noisy model run.
