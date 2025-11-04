"""
Compare llama.cpp and vLLM backends side-by-side.

This script runs the same inference tasks on both backends and compares:
1. Performance (speed)
2. Output quality/consistency
3. Memory usage
4. Ease of use

Usage:
    python compare_backends.py \
        --llamacpp-model /path/to/model.gguf \
        --vllm-model google/gemma-2-2b-it
"""

import argparse
import time
import tracemalloc
from typing import List, Tuple
from tabulate import tabulate

from inference_backend import create_backend


def measure_inference(backend, name: str, test_cases: List[Tuple[str, str, str]]):
    """Run inference tests and measure performance."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print('='*60)

    results = []

    for i, (prompt, original_doc, doc_in_progress) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Original: {original_doc[:50]}...")
        print(f"  Progress: {doc_in_progress}")

        # Test next-token prediction
        tracemalloc.start()
        start_time = time.time()

        try:
            predictions, scores = backend.get_next_token_predictions(
                prompt, original_doc, doc_in_progress, k=5, n_lookahead=2
            )
            elapsed = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"\n  Results ({elapsed:.3f}s, peak memory: {peak / 1024 / 1024:.1f} MB):")
            for pred, score in zip(predictions, scores):
                print(f"    '{pred}': {score:.4f}")

            results.append({
                'test_case': i+1,
                'time': elapsed,
                'peak_memory_mb': peak / 1024 / 1024,
                'predictions': predictions,
                'scores': scores,
                'error': None,
            })

        except Exception as e:
            tracemalloc.stop()
            print(f"  ERROR: {str(e)}")
            results.append({
                'test_case': i+1,
                'time': None,
                'peak_memory_mb': None,
                'predictions': None,
                'scores': None,
                'error': str(e),
            })

    return results


def compare_results(llamacpp_results, vllm_results):
    """Compare results from both backends."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    # Performance comparison
    table_data = []
    for i, (lc_res, vllm_res) in enumerate(zip(llamacpp_results, vllm_results)):
        if lc_res['error'] is None and vllm_res['error'] is None:
            speedup = lc_res['time'] / vllm_res['time'] if vllm_res['time'] > 0 else 0
            table_data.append([
                i+1,
                f"{lc_res['time']:.3f}s",
                f"{vllm_res['time']:.3f}s",
                f"{speedup:.2f}x",
                f"{lc_res['peak_memory_mb']:.1f} MB",
                f"{vllm_res['peak_memory_mb']:.1f} MB",
            ])
        else:
            table_data.append([
                i+1,
                "ERROR" if lc_res['error'] else f"{lc_res['time']:.3f}s",
                "ERROR" if vllm_res['error'] else f"{vllm_res['time']:.3f}s",
                "-",
                "-",
                "-",
            ])

    print("\nPerformance:")
    print(tabulate(
        table_data,
        headers=["Test", "llama.cpp", "vLLM", "Speedup", "LC Memory", "vLLM Memory"],
        tablefmt="grid"
    ))

    # Output comparison
    print("\nOutput Comparison:")
    for i, (lc_res, vllm_res) in enumerate(zip(llamacpp_results, vllm_results)):
        if lc_res['error'] is None and vllm_res['error'] is None:
            print(f"\nTest {i+1}:")
            print("  llama.cpp predictions:", lc_res['predictions'])
            print("  vLLM predictions:     ", vllm_res['predictions'])

            # Check for overlap
            lc_set = set(lc_res['predictions'])
            vllm_set = set(vllm_res['predictions'])
            overlap = lc_set & vllm_set
            print(f"  Overlap: {len(overlap)}/{len(lc_set)} tokens")


def main():
    parser = argparse.ArgumentParser(description="Compare inference backends")
    parser.add_argument(
        "--llamacpp-model",
        type=str,
        help="Path to GGUF model for llama.cpp"
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        help="Model name for vLLM"
    )
    parser.add_argument(
        "--skip-llamacpp",
        action="store_true",
        help="Skip llama.cpp tests"
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip vLLM tests"
    )
    args = parser.parse_args()

    # Define test cases
    test_cases = [
        (
            "Rewrite this text to be more concise.",
            "The quick brown fox jumps over the lazy dog.",
            "The brown fox"
        ),
        (
            "Make this more formal.",
            "Hey, what's up? I was thinking we should meet up later.",
            "Good day,"
        ),
        (
            "Simplify this sentence.",
            "The implementation of the algorithm was characterized by excessive complexity.",
            "The algorithm was"
        ),
    ]

    # Run tests
    llamacpp_results = None
    vllm_results = None

    if not args.skip_llamacpp and args.llamacpp_model:
        print("\nLoading llama.cpp backend...")
        llamacpp_backend = create_backend(
            "llamacpp",
            model_path=args.llamacpp_model,
            n_gpu_layers=-1,
        )
        llamacpp_results = measure_inference(llamacpp_backend, "llama.cpp", test_cases)

    if not args.skip_vllm and args.vllm_model:
        print("\nLoading vLLM backend...")
        vllm_backend = create_backend(
            "vllm",
            model_name=args.vllm_model,
            tensor_parallel_size=1,
        )
        vllm_results = measure_inference(vllm_backend, "vLLM", test_cases)

    # Compare
    if llamacpp_results and vllm_results:
        compare_results(llamacpp_results, vllm_results)

    # Code cleanliness assessment
    print("\n" + "="*60)
    print("CODE CLEANLINESS ASSESSMENT")
    print("="*60)
    print("""
llama.cpp:
  Pros:
    - Direct control over model and cache
    - Lower-level, predictable behavior
    - Good for single requests
  Cons:
    - Manual cache management required
    - Sequential generation for branches (less efficient)
    - Need to handle tokenization carefully
    - Requires GGUF model format

vLLM:
  Pros:
    - Clean, high-level API
    - Automatic cache management (PagedAttention)
    - Efficient batching and parallel generation
    - Native beam search support
    - Built-in prompt logprobs
    - Works with HuggingFace models directly
  Cons:
    - Less control over low-level details
    - Beam search gives sequences, not exact top-k next tokens
    - Requires more memory for the engine

RECOMMENDATION:
  For the writing prototype use case, vLLM appears cleaner and more
  efficient. The two-stage approach (get logprobs, then generate lookahead)
  maps well to vLLM's API, and PagedAttention handles cache sharing
  automatically. The prompt_logprobs feature is perfect for highlights.
    """)


if __name__ == "__main__":
    main()
