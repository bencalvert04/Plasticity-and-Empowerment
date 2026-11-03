# Daytona Code For Ben

Run the corrected whole-A experiment from your computer using nine Daytona CPU
sandboxes. This folder contains a new runner and coordinator; historical
experiments and their saved source snapshots are unchanged. Uploading this code
does not start the experiment.

## Get the code and prepare your coordinator

From a terminal with Git and Python 3.11 or newer (matching the pinned package
requirements; the workers use Python 3.11):

```sh
git clone https://github.com/bencalvert04/Plasticity-and-Empowerment.git
cd Plasticity-and-Empowerment
python3 -m venv .venv-daytona
source .venv-daytona/bin/activate
python -m pip install -r daytona_for_ben/requirements.txt
```

If you already cloned the repository, update your checkout first instead of
cloning it again. All following commands run from the repository root.
The coordinator needs the analysis/Daytona dependencies only. To run the unit
tests locally, also install `python -m pip install -r requirements.txt`; this
installs PyTorch and Transformers locally but does not run inference.

Load your existing Daytona key without printing it or putting it in shell history:

```sh
export DAYTONA_API_KEY="$(python -c 'import getpass; print(getpass.getpass("Daytona API key: "))')"
```

Alternatively, save it privately in
`~/.config/adept-research/daytona_api_key` with permissions 600. Never commit or
send the key. Only the coordinator reads it; worker uploads do not include it.

## Check, then launch

```sh
# Prints the protocol; no API calls, sandbox creation, or inference.
python -m daytona_for_ben.controller --plan

# Optional local tests after installing requirements.txt.
python -m unittest tests.test_daytona_whole_a125 tests.test_sampled_rlvr

# Creates nine paid CPU sandboxes and starts the full experiment.
python -u -m daytona_for_ben.controller
```

On a Mac you can use `caffeinate -i python -u -m daytona_for_ben.controller`
for the launch command. Keep the coordinator computer awake and online; this
does not guarantee operation with the laptop lid closed. Inference runs remotely,
but status checks, incremental downloads, final validation and cleanup run locally.

Default output: `runs/qwen06b_original50_free_a125_whole_a_b75_125/`.
Use `--run-dir runs/ANOTHER_DESCRIPTIVE_NAME` for a separate experiment.
An existing registry is rejected to prevent accidental duplicate launches.
The plan command is not an account-authentication or quota check.

## Exact experiment

- First 50 original rows of `data/gsm8k200/gsm8k200_injections.parquet`;
  10 trials per question at pinned steps 0, 77 and 251: 1,500 attempts.
- Fresh A per checkpoint/trial, EOS or 125 generated tokens; then original
  correction as a new user message; fresh B, EOS or 125 generated tokens.
- No added system prompt, arithmetic-only note, or supplied B opening.
- Primary eligibility requires 125 non-EOS content tokens in both A and B.
  Early EOS is saved but excluded, without replacement. Short A skips B.
  Capped spans without EOS remain eligible.
- Full B context keeps Q, all A, correction J, and previous B tokens.
- Plasticity baseline retains A and an empty correction turn.
- Empowerment baseline removes ALL sampled A1–125, retaining empty assistant
  framing, Q, J, and exactly the same previous B tokens.
- Only full context samples B. Baselines use independent KV caches and the
  identical realized B history, including B1–74.
- Score B75–125 inclusive (51 positions, Python slice 74:125). Each score is
  baseline full-vocabulary entropy minus full-context entropy, averaged in bits/token.

The [detailed handoff](EXPERIMENT_OVERVIEW.md) includes checkpoint revisions,
seed formula and interpretation. Its implementation-status section reflects
the runnable files now included here.

## Resources and outputs

Three disjoint question shards per checkpoint (`source_row % 3`), 170/170/160
attempts. Each worker: 4 vCPU, 8 GiB RAM, 10 GiB disk; CPU torch 2.11.0,
Transformers 5.5.0, PyArrow 26.0.0; float32 weights/entropy, four threads,
temperature 1, no top-k/top-p truncation, Qwen3 nonthinking template.

The coordinator saves source snapshots/hashes, per-shard config, raw JSONL,
progress, logs and validation receipts. Final combined outputs are
`rollouts.parquet`, `question_means.parquet`, `summary.json`, and `COMPLETE`.
Means weight trials within question, then represented questions equally. Plotting
and confidence-interval computation are follow-up analysis, not produced by this
coordinator. Coverage must accompany the means; early-EOS selection can differ
across checkpoints. These are entropy-ablation proxies, not formal DI or
validated environment-control empowerment.

## Failure handling

Workers are provisioned before inference begins; provisioning failure triggers
cleanup of created workers. Completed outputs are downloaded and validated
before their sandbox is deleted. Proxy timeouts are retried. Failed jobs attempt
to download partial results and stop their sandbox for recovery. SDK 0.227.3
uses `auto_delete_interval=-1` to retain stopped sandboxes; zero would delete
them immediately. Stopped sandboxes may retain storage costs until removed.

If interrupted or a shard fails, inspect `status.json` and the Daytona dashboard.
Do not launch a fresh nine-sandbox run over the same registry or delete a sandbox
before downloading needed records. The worker supports `--resume` with an
identical config, but this coordinator does not provide automatic restart of a
failed launch. Recovery requires restoring the saved source snapshot and exact
per-shard config/JSONL to the worker. Missing attempts must retain their original
seeds; already saved attempts must not be resampled.
