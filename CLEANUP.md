# Supported workflow and retired scripts

The publication uses one inference runner, `sampled_rlvr_experiment.py`, plus
`verify_results.py`, `analyze_checkpoint_statistics.py`, and
`plot_checkpoint_means.py`. `experiment_config.py` supplies pinned checkpoints and
the bundled dataset. The generation and scoring functions were retained from
the available sampled-runner snapshot; paths, validation, and provenance were
made self-contained. The unresolved historical code hash is documented separately.

The following files were removed from the published main tree:

| Files | Reason |
|---|---|
| `core.py`, `experiments.py`, `test_core.py` | The old answer-level `history` baseline held the first-token distribution fixed. It did not isolate reasoning-to-answer influence. These scripts also used a different random-pause, ten-token correction protocol. Keeping parallel default runners would obscure what produced the current results. |
| `explore.py`, `requirements-visualization.txt` | Embedding clusters and UMAP were exploratory selection tools, not evidence for the current checkpoint comparison. |
| `prompts.json`, `injection_cases.example.json` | Inputs for the retired runners, not the dataset used for the published run. |

Other scripts present only in older branches or local checkouts are intentionally
not promoted into the current workflow:

| Scripts/artifacts | Reason |
|---|---|
| `directed_information_convergence_to_zero.py`, `effect_of_prompt_length_on_directed_information.py` and other early DI-named entropy scripts | Names imply a formal information quantity or convergence result that entropy ablations do not establish. Prompt-length sums also mix output-length and prediction effects. |
| `compare_rlvr_checkpoints.py` | Single-question greedy smoke test with EOS suppression. Its checkpoint constants now live in a configuration module without a second generation workflow. |
| `directed_information.py`, its tests and standalone instructions | Separate, unconverged finite-particle estimator. Diagnostic outputs and limitations are retained, but it is not the supported measurement pipeline. |
| `output_lengths.py`, `compare_lengths.py` | Length and mixed-split toy accuracy comparisons do not answer the current checkpoint question. |
| Legacy result dumps, process IDs, logs, smoke outputs, and redundant `checkpoint_means.parquet` | Stale or redundant output can be mistaken for current evidence. Primary raw results remain, and derived statistics/plots can be regenerated. |

Removed tracked files remain recoverable from Git history. Existing local
checkouts and uncommitted work were preserved. Older branches and open PRs are
historical proposals and may still contain retired code; they are not additional
supported workflows. Dataset records from PR #7 are incorporated with attribution
in [provenance](provenance/README.md).
