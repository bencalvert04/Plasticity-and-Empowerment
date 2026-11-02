# Daytona CPU speed check

This directory contains the local controller and the remote one-question test runner. Completed outputs are tracked separately under `results/daytona_cpu_speed_check_late_a/`.

The controller stages the runner/helpers/pinned model manifest/dataset into a temporary 4-vCPU, 8-GiB, 10-GiB sandbox, installs pinned CPU dependencies, polls for completion, downloads results, and deletes the sandbox in a `finally` block. It also configures a remote idle timeout and deletion interval. It does not launch the full experiment.

The controller currently reads a local private key file at `~/.config/adept-research/daytona_api_key`. This file is outside the repository and is not published. Use your own organization key. Local controller logs/process records are operational files, not research results.

Run from the repository root with the Daytona SDK installed:

```sh
python daytona_pilot/controller.py
```

The speed check is completed; running this command creates another charged sandbox and reruns it. Read the results before deciding whether another test is needed.
