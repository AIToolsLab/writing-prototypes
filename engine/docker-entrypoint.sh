#!/bin/sh
# Runs the engine on $MODEL_FILE from $MODEL_REPO at $MODEL_REVISION in the Hugging Face cache, plus any extra arguments.
set -e
model="${HF_HOME:-/root/.cache/huggingface}/hub/models--$(echo "$MODEL_REPO" | sed "s|/|--|g")/snapshots/$MODEL_REVISION/$MODEL_FILE"
if [ ! -f "$model" ]; then
    echo "$MODEL_FILE from $MODEL_REPO (revision $MODEL_REVISION) isn't in the Hugging Face cache; run engine/download-model.sh on the host." >&2
    exit 1
fi
check-model-update.sh "$model" "$MODEL_REPO" "$MODEL_FILE" || true
exec writing-engine --model "$model" --host 0.0.0.0 --port 52409 "$@"
