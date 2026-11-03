# Whole-A contribution experiment — handoff for Ben

This supersedes the previous late-A-only Daytona protocol. No corrected run has launched. Coordinate from Ben's machine; inference runs on Daytona.

## Main change

Remove ALL sampled A1–125 tokens in the empowerment baseline, not just A75–125. Keep the scoring window at B75–125 inclusive (51 tokens). Plasticity remains unchanged. There is NO added system prompt, arithmetic-only instruction, or supplied B opening.

## Checkpoints

| Additional GRPO step | Repository | Exact revision |
|---|---|---|
| 0 | Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| 77 | cmpatino/qwen-grpo-r2 | 628c96470d7289309a6fe9b1292b8039e7b66794 |
| 251 | cmpatino/qwen-grpo-r5 | 06c0dac913ec3ca17e3a6418c136be7bf8fcedb8 |

Weights stay frozen during evaluation. Step 0 already has prior training. These are sequential public GRPO stages, with settings varying across stages, rather than independent controlled training seeds.

## Dataset and generation

- Take exactly the FIRST 50 rows of `data/gsm8k200/gsm8k200_injections.parquet`, preserving IDs, question text, original correction text, row order, revised answers and source-issue flags. Do not randomly sample another 50 from the 200.
- Ten trials per question per checkpoint: 1,500 attempted evaluations. The original 50 were historically length-screened, not an unbiased GSM8K sample.
- Present Q and generate fresh A separately at each checkpoint/trial: stop at EOS or 125 generated tokens. Do not fix A across checkpoints.
- Require 125 non-EOS A content tokens for eligibility. If shorter, save attempt, exclude from primary scores and skip B. No replacements.
- Insert the original correction J as a new USER message. Generate B as a new assistant message, EOS or 125 generated tokens.
- Require 125 non-EOS B content tokens for primary scores. Shorter B is saved but excluded. A/B reaching the cap without EOS remain eligible. 124 content tokens followed by EOS is ineligible.
- Use `qwen3_nonthinking`, `enable_thinking=False`, and existing exact chat-template splicing. Keep forced empty-thinking markers and original A token IDs; do not decode and re-encode A.

## Three B contexts

| Context | Question | Generated A | Correction | B history |
|---|---|---|---|---|
| Full | Q | All A1–125 | J | Same actual B_<t |
| Plasticity baseline | Q | All A1–125 | Empty correction turn | Same actual B_<t |
| Empowerment baseline | Q | Empty generated-A span | J | Same actual B_<t |

Preserve role boundaries and empty assistant framing when removing A. None of the 125 sampled A tokens may remain in the empowerment baseline. Only full context samples B. Baselines independently cache their own contexts and evaluate the SAME exact generated B history, including B1–74. Never generate separate baseline continuations.

At each target position t:

    P_t = H(next token | Q,A,empty J,B_<t) - H(next token | Q,A,J,B_<t)
    E_t = H(next token | Q,empty A,J,B_<t) - H(next token | Q,A,J,B_<t)

Entropy uses the full vocabulary, log base 2 and float32 arithmetic. Score positions 75–125 inclusive: Python `range(74,125)` or slice `[74:125]`, exactly 51 positions. Average the 51 differences to report bits/token. Do not substitute realized-token surprisal for vocabulary entropy.

Positive means the included source lowers entropy; negative means it raises entropy. Near zero does not establish no distributional/behavioral influence. These are working entropy-ablation proxies, not formal DI/MI or established environment-control empowerment.

## Sampling and seeds

Match the previous CPU protocol: float32 weights/entropy, eval/inference mode, temperature 1.0, full-softmax sampling, no top-k/top-p truncation, four CPU threads. Preserve existing generator/RNG implementation.

    base_seed = 20261030
    B_seed = base_seed + step*10000000 + source_row*10000 + trial
    A_seed = B_seed + 1000000000000

Use source_row 0–49 and trial 0–9. Save actual seeds, pinned revisions, dependency versions, code/data hashes and full config.

## Daytona execution

Nine sandboxes: three question shards per checkpoint, `source_row % 3`, giving 170/170/160 attempts per checkpoint. Previous resources: 4 vCPU, 8 GiB RAM, 10 GiB disk per worker. Previous environment: Python 3.11, CPU torch 2.11.0, transformers 5.5.0, pyarrow 26.0.0, Daytona SDK 0.227.3.

Use a NEW descriptive protocol/run directory, such as `qwen06b_original50_free_a125_whole_a_b75_125`. Never overwrite the completed late-A registry or source snapshots. Use Ben's Daytona credentials stored outside Git; do not upload credentials to workers or send keys in messages.

For SDK 0.227.3, `auto_delete_interval=-1` disables automatic deletion. Zero means immediate deletion on stop. Retry transient command-proxy timeouts; download and validate before deletion. Resume only missing (step,row,trial) records with identical config and seeds. Preserve all already saved records. Keep the coordinating machine awake/online unless coordination is explicitly moved to persistent infrastructure; workers and local monitoring/download/cleanup are different processes.

## Saved data and analysis

Save EVERY attempt: source ID/row, question, correction, revised answer/source flags; A/B original token IDs and text; EOS/cap status and content lengths; all three per-position B entropies; per-token differences and trial means; eligibility reason; seeds, revisions, timing and hashes. Flush JSONL incrementally; provide final parquet, status and completion markers.

Average eligible trials within question, then equally weight represented questions at each checkpoint. Plot both proxy means against steps 0/77/251 with question-bootstrap intervals. Report attempts, eligible trials and represented questions. Include shared-question sensitivity analysis; matching questions does not remove trial-selection bias. Retain early-EOS data for later analysis. Distinguish entropy changes from actual correction-following accuracy.

## Code location and implementation status

Runnable code is now included under `daytona_for_ben/`: `run_shard.py` and `controller.py`. See `README.md` in that folder for installation, credential setup, no-compute plan and launch commands. Shared helpers remain at repository root. No experiment was launched while preparing this handoff.

The third context uses `([], q['injection'])`, retaining empty assistant framing while removing all sampled A tokens. Metadata identifies source `[1,125]`, target `[75,125]`, selected A count 125 and B score count 51. Unit tests check this distinction. The new coordinator uses a separate output registry; historical snapshots are immutable.

The previous run measured late A's additional contribution given early A. This run measures the whole A contribution given Q, J and the same B history.
