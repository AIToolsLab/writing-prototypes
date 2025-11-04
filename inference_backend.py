"""
Unified inference backend that can use either llama.cpp or vLLM.

This module provides a common interface for both backends, making it easy
to switch between them or run both for comparison.
"""

from typing import List, Tuple, Optional, Literal
from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def get_next_token_predictions(
        self,
        prompt: str,
        original_doc: str,
        doc_in_progress: str,
        k: int = 5,
        n_lookahead: int = 2,
    ) -> Tuple[List[str], List[float]]:
        """Get top-k next token predictions with lookahead."""
        pass

    @abstractmethod
    def get_highlights(
        self,
        prompt: str,
        original_doc: str,
        updated_doc: str,
        k: int = 5,
    ) -> List[dict]:
        """Get edit suggestions by comparing predictions with actual tokens."""
        pass


class LlamaCppBackend(InferenceBackend):
    """Backend using llama.cpp."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        from llamacpp_inference import LlamaCppInference

        self.engine = LlamaCppInference(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def get_next_token_predictions(
        self,
        prompt: str,
        original_doc: str,
        doc_in_progress: str,
        k: int = 5,
        n_lookahead: int = 2,
    ) -> Tuple[List[str], List[float]]:
        return self.engine.get_next_token_predictions(
            prompt, original_doc, doc_in_progress, k, n_lookahead
        )

    def get_highlights(
        self,
        prompt: str,
        original_doc: str,
        updated_doc: str,
        k: int = 5,
    ) -> List[dict]:
        return self.engine.get_highlights(prompt, original_doc, updated_doc, k)


class VLLMBackend(InferenceBackend):
    """Backend using vLLM."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        use_beam_search: bool = False,
    ):
        from vllm_inference import VLLMInference

        self.engine = VLLMInference(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        self.use_beam_search = use_beam_search

    def get_next_token_predictions(
        self,
        prompt: str,
        original_doc: str,
        doc_in_progress: str,
        k: int = 5,
        n_lookahead: int = 2,
    ) -> Tuple[List[str], List[float]]:
        if self.use_beam_search:
            return self.engine.get_next_token_predictions_beam(
                prompt, original_doc, doc_in_progress, k, n_lookahead
            )
        else:
            return self.engine.get_next_token_predictions(
                prompt, original_doc, doc_in_progress, k, n_lookahead
            )

    def get_highlights(
        self,
        prompt: str,
        original_doc: str,
        updated_doc: str,
        k: int = 5,
    ) -> List[dict]:
        return self.engine.get_highlights(prompt, original_doc, updated_doc, k)


def create_backend(
    backend_type: Literal["llamacpp", "vllm"],
    **kwargs
) -> InferenceBackend:
    """
    Factory function to create an inference backend.

    Args:
        backend_type: Type of backend ("llamacpp" or "vllm")
        **kwargs: Backend-specific configuration

    Returns:
        An InferenceBackend instance

    Example:
        >>> # For llama.cpp
        >>> backend = create_backend(
        ...     "llamacpp",
        ...     model_path="/path/to/model.gguf",
        ...     n_gpu_layers=-1
        ... )

        >>> # For vLLM
        >>> backend = create_backend(
        ...     "vllm",
        ...     model_name="google/gemma-2-9b-it",
        ...     tensor_parallel_size=1
        ... )
    """
    if backend_type == "llamacpp":
        return LlamaCppBackend(**kwargs)
    elif backend_type == "vllm":
        return VLLMBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test inference backends")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["llamacpp", "vllm"],
        required=True,
        help="Backend to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (for llamacpp) or name (for vllm)"
    )
    args = parser.parse_args()

    # Create backend
    if args.backend == "llamacpp":
        backend = create_backend("llamacpp", model_path=args.model)
    else:
        backend = create_backend("vllm", model_name=args.model)

    # Test
    prompt = "Rewrite this text to be more concise."
    original_doc = "The quick brown fox jumps over the lazy dog."
    doc_in_progress = "The brown fox"

    print("Getting next token predictions...")
    predictions, scores = backend.get_next_token_predictions(
        prompt, original_doc, doc_in_progress, k=5, n_lookahead=2
    )

    print("\nPredictions:")
    for pred, score in zip(predictions, scores):
        print(f"  '{pred}': {score:.4f}")
