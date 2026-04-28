#!/usr/bin/env bash
# Self-hosted GitHub Actions runner — job-started hook (runs on the HOST).
#
# Configure on each runner machine, e.g. in the service environment:
#   ACTIONS_RUNNER_HOOK_JOB_STARTED=/path/to/llvm/devops/scripts/gha_runner_job_started_hook.sh
#
# GitHub runs this after a job is assigned to this runner but BEFORE the job
# container is created, so the check applies to the same host that will pull
# and run Docker (unlike a separate workflow job with the same runs-on labels,
# which may run on a different machine in the pool).
#
# Optional: UR_CI_MIN_FREE_DISK_MB (default 20480) — minimum free space on / in MiB.
#
# https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/running-scripts-before-or-after-a-job

set -euo pipefail

MIN_MB="${UR_CI_MIN_FREE_DISK_MB:-20480}"

if ! command -v df >/dev/null 2>&1; then
  echo "WARNING: df not available; skipping UR CI disk preflight"
  exit 0
fi

avail_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
avail_mb=$((avail_kb / 1024))

echo "gha_runner_job_started_hook: free on / = ${avail_mb} MiB (minimum ${MIN_MB} MiB)"

if [ "$avail_mb" -lt "$MIN_MB" ]; then
  echo "ERROR: insufficient disk space on runner host (${avail_mb} MiB free, need at least ${MIN_MB} MiB)"
  exit 1
fi

exit 0
