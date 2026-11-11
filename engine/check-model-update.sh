#!/bin/sh
# Says whether upstream's copy of $MODEL_FILE differs from the pinned one. Never downloads; silent if offline.
# Usage: check-model-update.sh <path to the pinned file in the HF cache> <repo> <file>
local_sha=$(basename "$(readlink "$1")")
headers=$(curl -sfI --max-time 10 "https://huggingface.co/$2/resolve/main/$3" | tr -d '\r"')
remote_sha=$(echo "$headers" | awk -F': ' 'tolower($1)=="x-linked-etag" {print $2}')
commit=$(echo "$headers" | awk -F': ' 'tolower($1)=="x-repo-commit" {print $2}')
if [ -n "$local_sha" ] && [ -n "$remote_sha" ] && [ "$remote_sha" != "$local_sha" ]; then
    echo "NOTE: $2 has a revised $3 (revision $commit). Not downloaded; to switch, run download-model.sh with that revision and update MODEL_REVISION in compose.yaml." >&2
fi
