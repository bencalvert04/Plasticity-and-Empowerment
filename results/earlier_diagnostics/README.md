# Diagnostic evidence only

These files document limitations discovered in earlier experiments. They are
not results from the three-checkpoint sampled RLVR experiment and must not be
pooled with it.

- `directed_information.parquet` and `directed_information_summary.json` contain
  the separate Qwen source-prior run. Historical filenames and column labels are
  retained. Finite-particle convergence was not established; posterior collapse
  and source capping limit interpretation.
- `answer_pilot.json` contains the six-question Qwen answer pilot and the
  one-example correction controls. Only one question had usable answer scores,
  and that answer was wrong.

See [findings](../../FINDINGS.md) for the verified diagnostics and their scope.
The original file hashes are recorded in
[the artifact manifest](../../provenance/artifact_sha256.json). Their runners are
not supported entry points in this publication.
