# Artifact provenance and limits

Published October 30, 2026. The raw sampled run, question summaries, and
`config.json` were copied without modification from the local completed
`plasticity-pythia-baseline/results/sampled_rlvr_50x20` run. Its configuration
records 50 questions, 20 rollouts, three pinned checkpoints, and seed 20261030.
The directory name is historical; the models are 1PP checkpoints, not Pythia.

`artifact_sha256.json` inventories the original evidence included here. It covers
raw results, dataset records, earlier diagnostic artifacts, and the source
snapshot. Generated analysis tables and plots are intentionally excluded because
they can be rebuilt. Run `python3 verify_results.py` from the repository root.

## Historical code identity is unresolved

The run configuration and both raw Parquet metadata blocks agree on runner SHA256:

```text
33fe5dc18356bce67dcc0a26fc95a437f997a40cc4048c10d40bc03f8703a1b0
```

The runner available in the source checkout when this publication was prepared
had SHA256:

```text
35673141760033744a66a565124403d77ee8ded2c1e8a5c6bf187323d262e5e6
```

That available file is preserved verbatim as
[sampled_runner_snapshot.py.txt](sampled_runner_snapshot.py.txt). It is an audit
snapshot, not a certified copy of the code that generated the run and not a
supported entry point. Its old import and sibling-checkout path are historical.
The matching original runner was not recovered during this audit. We do not
rewrite the run configuration to make the hashes agree.

The published runner is adapted from the available snapshot: checkpoint constants
now live in `experiment_config.py`, the dataset is bundled, new runs default to
`runs/`, input timing/IDs are validated, and environment/module hashes are recorded.
The generation and scoring functions are unchanged from that available snapshot.
This preserves a usable implementation, but does not close the historical gap.

Raw entropy arithmetic, stopping flags, all 3,000 rollout identifiers/seeds,
source fields, and all 150 question summaries verify independently. Recomputed
primary means, intervals, and tests match the previously saved numerical results.
These checks establish internal consistency, not exact historical model execution
or bit-for-bit rerun reproducibility. The original run did not record a complete
environment lockfile; dependency pins here describe the publication environment.

## Dataset and earlier evidence

The dataset and its source records come from
[dataset contribution #7](https://github.com/bencalvert04/Plasticity-and-Empowerment/pull/7),
commit `0d9c86fe1fc855412979287e8db3140c4ca58124`. The Parquet bytes match the actual
run input, SHA256 `571d941e89f43bfa3e62c2bf7f2dacea60501bdf02086ef5406acb8b4d81e9eb`.
The review document only has heading punctuation normalized in this publication.

Earlier Qwen source-prior estimates and the answer/injection pilot are preserved
in `results/earlier_diagnostics/` as diagnostic evidence. Their limitations and
why their runners are not supported here are recorded in [findings](../FINDINGS.md).
Their saved outputs are not pooled with the RLVR results.
