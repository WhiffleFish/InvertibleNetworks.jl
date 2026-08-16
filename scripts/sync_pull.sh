#!/usr/bin/env bash
# Pull generated experiment artifacts from the server down into their local-equivalent
# paths, since the repo is checked out at the same relative location on both sides.
#
# Usage:
#   scripts/sync_pull.sh              # pull current dir (if inside the repo), else everything under results/
#   scripts/sync_pull.sh some/subdir   # pull just that subdir (relative to repo root)
#   scripts/sync_pull.sh --dry-run     # preview without copying
#   scripts/sync_pull.sh --delete      # also remove local files gone on remote

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
    "$LOCAL_ROOT")   SUBDIR="results/" ;;
    *)               SUBDIR="results/" ;;
  esac
fi

RSYNC_ARGS=(-avzm --exclude='.CondaPkg/' --exclude='*.jl')
if (( ${#EXTRA_ARGS[@]} )); then
  RSYNC_ARGS+=("${EXTRA_ARGS[@]}")
fi

# A trailing slash on both paths syncs the directory contents, avoiding a
# nested <subdir>/<subdir> directory when the local destination already exists.
rsync "${RSYNC_ARGS[@]}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/${SUBDIR%/}/" \
  "${LOCAL_ROOT}/${SUBDIR%/}/"
