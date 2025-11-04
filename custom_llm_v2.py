"""
FastAPI server for custom inference using llama.cpp or vLLM backends.

This is a drop-in replacement for custom_llm.py that supports multiple backends.
Run with:
    python custom_llm_v2.py --backend vllm --model google/gemma-2-9b-it
    python custom_llm_v2.py --backend llamacpp --model /path/to/model.gguf

The API endpoints remain compatible with the original implementation.
"""

import argparse
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_backend import create_backend, InferenceBackend


# Global backend instance
backend: Optional[InferenceBackend] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and release on shutdown."""
    global backend

    print(f"Loading {args.backend} backend with model: {args.model}")
    start_time = time.time()

    # Create the backend based on CLI arguments
    if args.backend == "llamacpp":
        backend = create_backend(
            "llamacpp",
            model_path=args.model,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.context_size,
            verbose=args.verbose,
        )
    elif args.backend == "vllm":
        backend = create_backend(
            "vllm",
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            use_beam_search=args.use_beam_search,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    load_time = time.time() - start_time
    print(f"Backend loaded in {load_time:.2f}s")

    # Run timing tests
    print("\nRunning timing tests...")
    test_prompt = "Rewrite this more concisely."
    test_doc = "The quick brown fox jumps over the lazy dog."
    test_progress = "The fox"

    # Test next-token prediction
    start_time = time.time()
    predictions, scores = backend.get_next_token_predictions(
        test_prompt, test_doc, test_progress, k=5, n_lookahead=2
    )
    next_token_time = time.time() - start_time
    print(f"Next-token prediction (k=5, lookahead=2): {next_token_time:.3f}s")
    print(f"Predictions: {predictions}")

    # Test highlights
    start_time = time.time()
    highlights = backend.get_highlights(
        test_prompt, test_doc, "The fox jumps", k=5
    )
    highlights_time = time.time() - start_time
    print(f"Highlights: {highlights_time:.3f}s")
    print(f"Number of highlights: {len(highlights)}")

    yield

    # Cleanup
    print("\nShutting down backend...")
    del backend


# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Custom LLM Inference API v2",
        "backend": args.backend,
        "model": args.model,
    }


@app.get("/api/next_token")
def get_next_token_predictions(
    original_doc: str,
    prompt: str,
    doc_in_progress: str,
    k: Optional[int] = 5,
    n_lookahead: Optional[int] = 2,
):
    """
    Get top-k next token predictions with lookahead.

    Args:
        original_doc: The original document being rewritten
        prompt: The instruction/prompt for rewriting
        doc_in_progress: The partially rewritten document
        k: Number of predictions to return (default: 5)
        n_lookahead: Number of tokens to look ahead (default: 2)

    Returns:
        JSON with next_tokens list
    """
    predictions, scores = backend.get_next_token_predictions(
        prompt, original_doc, doc_in_progress, k, n_lookahead
    )
    return {"next_tokens": predictions, "scores": scores}


@app.get("/api/highlights")
def get_highlights(
    doc: str,
    prompt: Optional[str] = None,
    updated_doc: Optional[str] = "",
    k: Optional[int] = 5,
):
    """
    Get edit suggestions by comparing model predictions with actual tokens.

    Args:
        doc: Original document
        prompt: Instruction/prompt (optional)
        updated_doc: User's edited version
        k: Number of alternative tokens per position (default: 5)

    Returns:
        JSON with highlights list
    """
    if prompt is None:
        prompt = "Edit this document."

    highlights = backend.get_highlights(prompt, doc, updated_doc, k)
    return {"highlights": highlights}


@app.get("/api/gen_revisions")
def gen_revisions():
    """
    Generate complete revisions (disabled for now).

    This endpoint is disabled as it's not the focus of the current implementation.
    """
    return {"error": "This endpoint is disabled"}


# CLI argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run custom LLM inference server with llama.cpp or vLLM"
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["llamacpp", "vllm"],
        required=True,
        help="Inference backend to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (llamacpp) or name (vllm)"
    )

    # Common options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19570,
        help="Port to bind to (default: 19570)"
    )

    # llama.cpp specific options
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (llamacpp only, -1 for all)"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8192,
        help="Context window size (llamacpp only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (llamacpp only)"
    )

    # vLLM specific options
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vllm only)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (vllm only, 0-1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model sequence length (vllm only)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help="Data type (vllm only)"
    )
    parser.add_argument(
        "--use-beam-search",
        action="store_true",
        help="Use beam search instead of two-stage approach (vllm only)"
    )

    args = parser.parse_args()

    # Run the server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
