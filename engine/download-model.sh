#!/bin/sh
# Downloads the engine's GGUF into the shared Hugging Face cache; a no-op if it's already there.
# Usage: engine/download-model.sh [repo] [file] [revision]   (defaults match compose.yaml)
# The revision is pinned because upstream re-uploads GGUFs under the same name; bump it deliberately.
set -e
repo=${1:-${MODEL_REPO:-unsloth/Qwen3.8-27B-GGUF}}
file=${2:-${MODEL_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}}
rev=${3:-${MODEL_REVISION:-ce331efc48d5722b4b8041f72793e530eae18d9b}}
export HF_HOME=${HF_HOME:-/storage/HuggingFace}
path=$(uvx --from 'huggingface_hub>=0.34' hf download "$repo" "$file" --revision "$rev" 2>/dev/null | tail -n1 | sed "s/^path=//")
echo "$path"
"$(dirname "$0")/check-model-update.sh" "$path" "$repo" "$file"
