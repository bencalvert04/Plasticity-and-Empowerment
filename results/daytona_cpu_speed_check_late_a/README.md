# Daytona CPU speed check — one question, three checkpoints

Completed November 2, 2026. The temporary sandbox was deleted after outputs were downloaded. This is a functionality/timing check, not a research comparison.

Resources: 4 vCPU, 8 GiB RAM, 10 GiB disk; Python 3.11, torch 2.11.0+cpu, transformers 5.5.0, pyarrow 26.0.0. Float32 weights and entropy calculations. One question (`test:51`), one trial per checkpoint. Full-distribution sampling at temperature 1.0, no top-k/top-p. Model revisions are in each result record.

The original question asks about a return journey at 6 mph. After A completes, the normal correction is: **“I meant the return speed was 5 mph, not 6 mph.”**

| Additional GRPO steps | A content tokens | A reached EOS? | A generation seconds | B generation/scoring seconds | Correction proxy bits/token | Late-A proxy bits/token |
|---|---:|---|---:|---:|---:|---:|
| 0 | 180 | Yes | 23.59 | 41.31 | 0.06410 | 0.06891 |
| 77 | 334 | Yes | 47.66 | 46.10 | 0.03066 | 0.12057 |
| 251 | 2,048 | **No: safety cap** | 317.87 | Not attempted | Missing | Missing |

A generates to EOS, with a 2,048-token safety cap. Eligibility requires EOS and at least 75 A content tokens. A's EOS is removed before constructing the next user turn. Full B receives all of A and the correction. The correction baseline keeps all A and removes the correction text. The late-A baseline retains A tokens 1–74 and removes tokens 75 through the final content token, keeping the correction. All B contexts receive the same generated B prefix. B is generated up to 100 tokens, and the score averages positions **51–100 inclusive** (50 positions). Incomplete B windows are missing rather than padded.

The successful controller run took approximately **11m41s** wall time: **2m59s** downloading/loading models, **7m57s** generating/scoring, and **46s combined** sandbox provisioning, dependency installation, polling, transfer, and cleanup. Wall time is inferred from local launch/completion file timestamps, so overhead is approximate; earlier failed authentication/resource requests are not included.

Both eligible checkpoints reached all 100 B tokens. No valid score exists at step 251; do not plot its missing score as zero or infer a three-checkpoint training trend.

The right next step is to inspect why step 251 did not finish A, and decide how to handle long or looping responses before launching a full run. Raising the A cap alone would increase CPU cost and might not resolve looping. A single question cannot establish representative runtime or an expected full-run budget.

`pilot_results.json` contains exact A token IDs/text, all B tokens and per-position entropies for the eligible records, and timings. `pilot_config.json` records the protocol; `summary.json` records source hashes and completion. Code is in `daytona_pilot/late_a_pilot.py`; it expects the shared helpers, manifest, and dataset beside it in the remote staging directory, as uploaded by `controller.py`.

No API keys or account credentials are included. Set your own credentials locally to use the controller. The full experiment was not launched.
