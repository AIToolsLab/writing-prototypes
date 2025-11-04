# Custom Inference Backend Comparison

This document compares the llama.cpp and vLLM implementations for the writing prototypes project.

## Overview

The project requires two core capabilities:
1. **Next-token prediction with lookahead**: Get multiple possible completions (top-k next tokens), each with a preview of what comes next (lookahead tokens)
2. **Highlight edits**: Compare model predictions with user edits to suggest improvements

We've implemented both capabilities using two different backends: llama.cpp and vLLM.

## Architecture

### Files Structure

```
writing-prototypes/
├── llamacpp_inference.py      # llama.cpp implementation
├── vllm_inference.py           # vLLM implementation
├── inference_backend.py        # Unified interface
├── custom_llm_v2.py            # FastAPI server supporting both backends
├── compare_backends.py         # Comparison script
└── requirements-*.txt          # Dependencies for each backend
```

### Unified Interface

Both backends implement the `InferenceBackend` abstract class:

```python
class InferenceBackend(ABC):
    def get_next_token_predictions(
        self, prompt, original_doc, doc_in_progress, k=5, n_lookahead=2
    ) -> Tuple[List[str], List[float]]:
        """Get top-k next token predictions with lookahead."""
        pass

    def get_highlights(
        self, prompt, original_doc, updated_doc, k=5
    ) -> List[dict]:
        """Get edit suggestions by comparing predictions with actual tokens."""
        pass
```

## Implementation Details

### vLLM Implementation

**File**: `vllm_inference.py`

**Key Features**:
- Uses PagedAttention for efficient KV cache management
- Two-stage approach for next-token prediction:
  1. Get logprobs for next token → find top-k candidates
  2. Generate lookahead for each candidate in parallel
- Native support for prompt logprobs (perfect for highlights)
- Automatic cache sharing across parallel requests

**Next-Token Prediction Flow**:
```python
# Stage 1: Get top-k next tokens via logprobs
sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=k)
outputs = llm.generate([prompt], sampling_params)
top_k_tokens = extract_top_k_from_logprobs(outputs)

# Stage 2: Generate lookahead for each branch in parallel
lookahead_prompts = [prompt + token for token in top_k_tokens]
lookahead_params = SamplingParams(temperature=0.0, max_tokens=n_lookahead-1)
lookahead_outputs = llm.generate(lookahead_prompts, lookahead_params)
# vLLM automatically shares the cache for the common prefix!
```

**Highlights Flow**:
```python
# Use prompt_logprobs to get logprobs for each token in the document
sampling_params = SamplingParams(prompt_logprobs=k)
outputs = llm.generate([full_document], sampling_params)
# Extract logprobs for each token and compare with actual tokens
```

**Advantages**:
- ✅ Efficient cache sharing via PagedAttention
- ✅ Clean, high-level API
- ✅ Parallel generation for branches
- ✅ Built-in prompt logprobs
- ✅ Works with HuggingFace models directly
- ✅ Production-ready with high throughput

**Disadvantages**:
- ❌ Requires more GPU memory for the engine
- ❌ Beam search gives sequences, not exact top-k next tokens (though we work around this)
- ❌ Less control over low-level details

### llama.cpp Implementation

**File**: `llamacpp_inference.py`

**Key Features**:
- Direct control over model and tokenization
- Uses GGUF format (quantized models)
- Manual cache management

**Next-Token Prediction Flow**:
```python
# Evaluate prompt once to get next token logits
model.reset()
model.eval(prompt_tokens)
logits = model.scores[-1]
top_k_tokens = get_top_k(logits)

# Save the KV cache state after evaluating the prompt
saved_state = model.save_state()

# For each branch, restore the saved state (efficient!)
for token in top_k_tokens:
    model.load_state(saved_state)  # Restore KV cache
    # Generate lookahead from this branch
    for _ in range(n_lookahead-1):
        model.eval([next_token])
        next_token = argmax(model.scores[-1])
```

**Highlights Flow**:
```python
# Evaluate full document to get logits for each position
model.reset()
model.eval(full_document_tokens)
# Extract logits and compute top-k for each position
```

**Advantages**:
- ✅ Direct control over model
- ✅ GGUF format enables smaller models (quantization)
- ✅ Lower memory footprint per request
- ✅ Predictable behavior
- ✅ Efficient cache save/restore using `save_state()`/`load_state()`
- ✅ Good for edge deployment and resource-constrained environments

**Disadvantages**:
- ❌ Sequential processing of branches (not parallel like vLLM)
- ❌ Requires GGUF model format (need to convert HuggingFace models)
- ❌ More manual work for tokenization
- ❌ State save/load has some overhead compared to vLLM's PagedAttention

## Performance Comparison

### KV Cache Efficiency

**vLLM**: ⭐⭐⭐⭐⭐
- Automatically shares cache across parallel requests
- PagedAttention enables efficient memory usage
- Prompt computed **once**, reused for all k branches
- Batched parallel generation of lookahead branches

**llama.cpp**: ⭐⭐⭐⭐
- Uses `save_state()`/`load_state()` to share cache
- Prompt computed **once**, state restored for each branch
- Sequential processing (one branch at a time)
- Small overhead from state save/restore operations

### Code Cleanliness

**vLLM**: ⭐⭐⭐⭐⭐
- Clean, high-level API
- Natural fit for our use case
- Minimal manual work

**llama.cpp**: ⭐⭐⭐
- More manual work required
- Lower-level control
- Cache management is tricky

### Ease of Deployment

**vLLM**: ⭐⭐⭐⭐
- Works with HuggingFace models out of the box
- Production-ready
- Good documentation

**llama.cpp**: ⭐⭐⭐
- Requires GGUF conversion
- More setup steps
- Good for embedded/edge deployment

## Running the Backends

### vLLM

```bash
# Install
pip install -r requirements-vllm.txt

# Run server
python custom_llm_v2.py \
    --backend vllm \
    --model google/gemma-2-9b-it \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9

# Optional: Use beam search (alternative approach)
python custom_llm_v2.py \
    --backend vllm \
    --model google/gemma-2-9b-it \
    --use-beam-search
```

### llama.cpp

```bash
# Install
pip install -r requirements-llamacpp.txt

# Or with CUDA support:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Convert HuggingFace model to GGUF (if needed)
# See: https://github.com/ggerganov/llama.cpp#prepare-data--run

# Run server
python custom_llm_v2.py \
    --backend llamacpp \
    --model /path/to/model.gguf \
    --n-gpu-layers -1 \
    --context-size 8192
```

### Compare Both

```bash
python compare_backends.py \
    --llamacpp-model /path/to/model.gguf \
    --vllm-model google/gemma-2-2b-it
```

## API Endpoints

Both backends expose the same API:

### `GET /api/next_token`

Get top-k next token predictions with lookahead.

**Parameters**:
- `prompt`: Instruction for rewriting
- `original_doc`: Original document
- `doc_in_progress`: Partially rewritten document
- `k`: Number of predictions (default: 5)
- `n_lookahead`: Lookahead tokens (default: 2)

**Response**:
```json
{
  "next_tokens": ["jumps", " leaps", " hops", " bounds", " springs"],
  "scores": [-0.5, -1.2, -1.8, -2.1, -2.4]
}
```

### `GET /api/highlights`

Get edit suggestions.

**Parameters**:
- `prompt`: Instruction
- `doc`: Original document
- `updated_doc`: Edited version
- `k`: Number of alternatives per position (default: 5)

**Response**:
```json
{
  "highlights": [
    {
      "start": 0,
      "end": 3,
      "token": "The",
      "token_loss": 0.5,
      "most_likely_token": "A",
      "topk_tokens": ["A", "The", "This", "One", "Some"]
    },
    ...
  ]
}
```

## Recommendation

### For Production: **vLLM** ⭐⭐⭐⭐⭐

**Reasons**:
1. **Efficiency**: PagedAttention handles cache sharing automatically
2. **Clean API**: Natural fit for our use case
3. **Performance**: Parallel generation for branches (all k lookaheads at once)
4. **Built-in features**: `prompt_logprobs` perfect for highlights
5. **Ease of use**: Works with HuggingFace models directly
6. **Production-ready**: Designed for high-throughput serving

### For Edge/Embedded: **llama.cpp** ⭐⭐⭐⭐

**Reasons**:
1. **Smaller footprint**: GGUF quantization for efficient memory usage
2. **Good cache efficiency**: `save_state()`/`load_state()` works well
3. **Deployment flexibility**: Works on CPU, GPU, and embedded devices
4. **No heavy dependencies**: Lighter weight than vLLM
5. **Direct control**: More predictable behavior

**Trade-off**: Sequential processing (one branch at a time) vs. vLLM's parallel batching, so vLLM will be faster for k>1 lookahead branches.

## Future Improvements

### For llama.cpp:
1. ✅ ~~Implement cache save/restore~~ (DONE - uses `save_state()`/`load_state()`)
2. Explore parallel processing of branches (would require multiple model instances)
3. Consider using llama.cpp server mode with stateful sessions for even better cache reuse
4. Benchmark state save/load overhead vs. vLLM's PagedAttention

### For vLLM:
1. Fine-tune beam search parameters
2. Explore speculative decoding for even faster lookahead
3. Consider using vLLM's prefix caching feature

### For both:
1. Support longer lookahead (n_lookahead parameter)
2. Add temperature/sampling controls
3. Implement batched requests for multiple documents

## Testing

Run tests with:

```bash
# Test vLLM
python vllm_inference.py  # Runs example in __main__

# Test llama.cpp
python llamacpp_inference.py  # Runs example in __main__

# Compare both
python compare_backends.py \
    --llamacpp-model /path/to/model.gguf \
    --vllm-model google/gemma-2-2b-it
```

## Conclusion

Both implementations are now **production-ready with efficient cache management**!

**vLLM** has a slight edge due to:
- Parallel processing of lookahead branches (faster when k>1)
- Native `prompt_logprobs` support (cleaner highlights implementation)
- Works directly with HuggingFace models (no conversion needed)

**llama.cpp** is competitive and offers:
- Better for quantized models (GGUF format)
- Lower resource requirements
- Good for edge/embedded deployment
- Efficient cache reuse via `save_state()`/`load_state()`

**Recommendation**:
- **Use vLLM** for production servers with GPU resources and HuggingFace models
- **Use llama.cpp** for edge deployment, CPU inference, or when you need quantized models

Both are viable choices - the decision depends on your deployment constraints rather than fundamental capability differences.
