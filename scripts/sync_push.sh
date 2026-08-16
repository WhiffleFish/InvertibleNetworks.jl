#!/usr/bin/env bash
# Push local code (scripts, src, examples, project files) up to the server so it
# can be run there, since the repo is checked out at the same relative location
# on both sides.
#
# Usage:
#   scripts/sync_push.sh              # push current dir (if inside the repo), else the whole repo
#   scripts/sync_push.sh some/subdir   # push just that subdir (relative to repo root)
#   scripts/sync_push.sh --dry-run     # preview without copying
#   scripts/sync_push.sh --delete      # also remove remote files gone locally (careful: see excludes below)

set -euo pipefail

REMOTE_HOST="omak.colorado.edu"
REMOTE_ROOT="~/code/InvertibleNetworks.jl"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA_ARGS=()
SUBDIR=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) EXTRA_ARGS+=(--dry-run) ;;
    --delete)  EXTRA_ARGS+=(--delete) ;;
    *)         SUBDIR="$arg" ;;
  esac
done

if [[ -z "$SUBDIR" ]]; then
  case "$PWD" in
    "$LOCAL_ROOT"/*) SUBDIR="${PWD#"$LOCAL_ROOT"/}/" ;;
    "$LOCAL_ROOT")   SUBDIR="" ;;
    *)               SUBDIR="" ;;
  esac
fi

# Only push code, not local run artifacts or environment state that should stay
# host-specific:
#   - .git/                 version control lives on both sides independently
#   - Manifest.toml          the resolved env (esp. CUDA/GPU stack) differs per host
#   - LocalPreferences.toml  host-specific package preferences (e.g. GPU backend)
#   - output/, results/, checkpoints/, wandb/, *.jld2, *.bson  run artifacts, pull these back instead
#   - .CondaPkg/, .venv/     unrelated local envs, if present
RSYNC_ARGS=(
  -avzm
  --exclude='.git/'
  --exclude='Manifest.toml'
  --exclude='**/Manifest.toml'
  --exclude='LocalPreferences.toml'
  --exclude='**/LocalPreferences.toml'
  --exclude='output/'
  --exclude='results/'
  --exclude='checkpoints/'
  --exclude='wandb/'
  --exclude='*.jld2'
  --exclude='*.bson'
  --exclude='.CondaPkg/'
  --exclude='.venv/'
)
if (( ${#EXTRA_ARGS[@]} )); then
  RSYNC_ARGS+=("${EXTRA_ARGS[@]}")
fi

# Ensure the remote destination exists before syncing into it.
ssh "$REMOTE_HOST" "mkdir -p ${REMOTE_ROOT}/${SUBDIR%/}"

# A trailing slash on both paths syncs the directory contents, avoiding a
# nested <subdir>/<subdir> directory when the remote destination already exists.
rsync "${RSYNC_ARGS[@]}" \
  "${LOCAL_ROOT}/${SUBDIR%/}/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/${SUBDIR%/}/"
