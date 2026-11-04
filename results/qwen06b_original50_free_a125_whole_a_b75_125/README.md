# Qwen3 0.6B — original 50, fresh A125, whole-A removal, B75–125

Completed November 3, 2026 on nine Daytona CPU sandboxes with `daytona_for_ben/` (protocol `whole_a1_125_b75_125_v1`). All 1,500 attempts were downloaded and passed the controller's `verify()`. Every sandbox was deleted afterwards. This is the corrected run: the empowerment baseline removes **all** sampled A tokens 1–125, not only A75–125 as the earlier Daytona run did.

Protocol, in brief (full details in `daytona_for_ben/README.md` and `EXPERIMENT_OVERVIEW.md`):
- First 50 original rows of the GSM8K200 dataset, 10 trials each, at frozen checkpoints 0, 77 and 251.
- Fresh A up to 125 tokens, then the original correction as a new user message, then B up to 125 tokens.
- No system prompt, no arithmetic-only instruction, no supplied opening.
- An A or B that reaches EOS before 125 content tokens is saved but excluded, with no replacement. If A is short, B is skipped.
- **Plasticity proxy:** baseline entropy with A kept and the correction emptied, minus full-context entropy.
- **Empowerment proxy:** baseline entropy with all of A removed (question and correction kept), minus full-context entropy.
- Both are averaged over B positions 75–125 inclusive (51 tokens), in bits/token. Trials are averaged within each question, then represented questions are weighted equally.

## Results

| Step | Eligible / attempted | A short | B short | Questions represented | Plasticity proxy | Empowerment proxy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 343 / 500 | 65 | 92 | 46 / 50 | 0.0186 | 0.0407 |
| 77 | 485 / 500 | 0 | 15 | 50 / 50 | 0.0106 | 0.0316 |
| 251 | 495 / 500 | 0 | 5 | 50 / 50 | −0.0060 | 0.0303 |

Coverage differs across checkpoints: step 0 loses 31% of attempts to early EOS, and four questions have no eligible trials. Restricting every checkpoint to the 46 questions represented at step 0 barely changes the means: 0.0186 / 0.0407, 0.0094 / 0.0320 and −0.0065 / 0.0314 (plasticity / empowerment, steps 0 / 77 / 251).

These are point estimates. Confidence intervals and plots are follow-up analysis. These are entropy-ablation proxies, not formal directed information or validated environment-control empowerment.

## Files

- `rollouts.parquet` — all 1,500 records: A and B token IDs and text, per-position entropies (full / plasticity baseline / empowerment baseline), seeds, eligibility flags, and per-record scores.
- `question_means.parquet` — per-checkpoint, per-question means and coverage counts.
- `summary.json` — checkpoint-level means and coverage. `COMPLETE` is the controller's completion marker.
- `source_sha256.json` — hashes of the code and data uploaded to the workers.
- `shards/<step>_<shard>/` — each worker's `config.json`, `DONE.json` and the controller's `validated.json`, which includes the SHA-256 of the raw JSONL.
- `status.json` — final controller status. `status_before_resume.json` is the status before recovery (see below).
- `provenance/` — coordinator logs and the one-off resume script.

## Interruption and recovery

At about 17:05 MST the coordinator lost contact with the sandboxes through the Daytona proxy. The workers kept generating. With no activity reaching them, Daytona's `auto_stop_interval=30` stopped the seven unfinished sandboxes at around 17:40–17:50. Their disks were retained (`auto_delete_interval=-1`).

Recovery steps:
1. The sandboxes were restarted with auto-stop disabled, and their remote outputs were downloaded.
2. In every shard, the coordinator's local JSONL was an exact prefix of the remote JSONL. The configs were byte-identical, and no lines were corrupt.
3. The remote records were installed locally. All seven shards were then resumed with the controller's own `worker(..., resume=True)` path (`provenance/dt_resume.py`), which runs `run_shard.py --resume` with the identical config.
4. Three shards were already complete. The other four needed 19 attempts in total, generated with their original seeds; no saved attempt was resampled.
5. All shards then passed `verify()` and were combined by `controller.combine()`.

Suggestion for the controller: use `auto_stop_interval=0`, plus explicit cleanup, or keep the sandboxes active independently of status checks, so a coordinator connection drop cannot kill the workers.

No API keys or account credentials are included.
