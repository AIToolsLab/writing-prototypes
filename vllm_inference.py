"""
Custom inference implementation using vLLM.

This module provides next-token prediction with lookahead using vLLM's
PagedAttention and efficient KV cache management. vLLM handles cache
sharing and batching internally, making it very clean for our use case.
"""

from typing import List, Tuple, Optional, Dict
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


class VLLMInference:
    """Inference engine using vLLM."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
    ):
        """
        Initialize vLLM model.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            dtype: Data type (auto, half, float16, bfloat16, float32)
        """
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def format_chat_prompt(
        self,
        prompt: str,
        original_doc: str,
        doc_in_progress: str
    ) -> str:
        """
        Format the prompt using the model's chat template.

        Args:
            prompt: System prompt or instruction
            original_doc: Original document
            doc_in_progress: Partially rewritten document

        Returns:
            Formatted prompt string
        """
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\nOriginal:\n{original_doc}\n\nRewrite:"
            },
            {
                "role": "assistant",
                "content": doc_in_progress
            }
        ]

        # Use the tokenizer's chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return formatted

    def get_next_token_predictions(
        self,
        prompt: str,
        original_doc: str,
        doc_in_progress: str,
        k: int = 5,
        n_lookahead: int = 2,
    ) -> Tuple[List[str], List[float]]:
        """
        Get top-k next token predictions with lookahead.

        vLLM's approach: We use a two-stage process:
        1. Get logprobs for the next token to find top-k candidates
        2. Generate lookahead for each candidate in parallel

        KEY ADVANTAGE: vLLM's PagedAttention automatically handles KV cache
        sharing across the parallel requests. The prompt is only computed once,
        and each lookahead branch efficiently reuses the cached computation.
        This is much more efficient than the llama.cpp approach which requires
        re-evaluating the prompt for each branch.

        Args:
            prompt: System prompt or instruction
            original_doc: Original document to rewrite
            doc_in_progress: Partially rewritten document
            k: Number of top predictions to return
            n_lookahead: Number of tokens to predict ahead for each branch

        Returns:
            Tuple of (decoded_sequences, scores)
        """
        # Format the input
        full_prompt = self.format_chat_prompt(prompt, original_doc, doc_in_progress)

        # Configure sampling to get top-k diverse continuations
        # We'll use a two-stage approach:
        # 1. First, get the logprobs for the next token to find top-k
        # 2. Then generate lookahead for each

        # Stage 1: Get next token logprobs
        sampling_params_logprobs = SamplingParams(
            temperature=0.0,  # Greedy for now
            max_tokens=1,
            logprobs=k,  # Return top-k logprobs
        )

        outputs = self.llm.generate(
            prompts=[full_prompt],
            sampling_params=sampling_params_logprobs,
            use_tqdm=False,
        )

        # Extract top-k tokens and their logprobs
        output = outputs[0]
        next_token_logprobs = output.outputs[0].logprobs[0]  # First token's logprobs

        # Sort by logprob to get top-k
        top_k_items = sorted(
            next_token_logprobs.items(),
            key=lambda x: x[1].logprob,
            reverse=True
        )[:k]

        # Stage 2: For each top-k token, generate lookahead
        # We'll do this in parallel by creating k prompts
        lookahead_prompts = []
        top_k_tokens = []
        top_k_logprobs = []

        for token_id, logprob_obj in top_k_items:
            # Decode the token
            token_str = self.tokenizer.decode([token_id])
            top_k_tokens.append(token_id)
            top_k_logprobs.append(logprob_obj.logprob)

            # Create a prompt with this token appended
            lookahead_prompts.append(full_prompt + token_str)

        # Generate lookahead for all branches in parallel
        sampling_params_lookahead = SamplingParams(
            temperature=0.0,  # Greedy for consistency
            max_tokens=n_lookahead - 1,  # We already have the first token
            logprobs=0,  # Don't need logprobs for lookahead
        )

        lookahead_outputs = self.llm.generate(
            prompts=lookahead_prompts,
            sampling_params=sampling_params_lookahead,
            use_tqdm=False,
        )

        # Combine first token + lookahead
        decoded_sequences = []
        for i, output in enumerate(lookahead_outputs):
            first_token = self.tokenizer.decode([top_k_tokens[i]])
            lookahead_text = output.outputs[0].text
            full_sequence = first_token + lookahead_text
            decoded_sequences.append(full_sequence)

        return decoded_sequences, top_k_logprobs

    def get_next_token_predictions_beam(
        self,
        prompt: str,
        original_doc: str,
        doc_in_progress: str,
        k: int = 5,
        n_lookahead: int = 2,
    ) -> Tuple[List[str], List[float]]:
        """
        Alternative implementation using beam search.

        This might be cleaner than the two-stage approach, but beam search
        doesn't give us exactly the top-k next tokens - it gives us the
        top-k sequences.

        Args:
            prompt: System prompt or instruction
            original_doc: Original document to rewrite
            doc_in_progress: Partially rewritten document
            k: Number of beams
            n_lookahead: Number of tokens to predict ahead

        Returns:
            Tuple of (decoded_sequences, scores)
        """
        full_prompt = self.format_chat_prompt(prompt, original_doc, doc_in_progress)

        # Use beam search
        sampling_params = SamplingParams(
            n=k,  # Number of output sequences
            best_of=k,  # Number of beams
            use_beam_search=True,
            max_tokens=n_lookahead,
            temperature=0.0,
        )

        outputs = self.llm.generate(
            prompts=[full_prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        output = outputs[0]
        decoded_sequences = [o.text for o in output.outputs]
        scores = [o.cumulative_logprob / len(o.token_ids) for o in output.outputs]

        return decoded_sequences, scores

    def get_highlights(
        self,
        prompt: str,
        original_doc: str,
        updated_doc: str,
        k: int = 5,
    ) -> List[dict]:
        """
        Get edit suggestions by computing logprobs for the updated document.

        This uses prompt logprobs to efficiently compute the probability
        of each token in the updated document.

        Args:
            prompt: System prompt or instruction
            original_doc: Original document
            updated_doc: User's edited version
            k: Number of alternative tokens to return per position

        Returns:
            List of highlight spans with token alternatives
        """
        # Format the full sequence
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\nOriginal:\n{original_doc}\n\nRewrite:"
            },
            {
                "role": "assistant",
                "content": updated_doc
            }
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Get logprobs for the entire sequence using prompt logprobs
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # We only care about prompt logprobs
            prompt_logprobs=k,  # Get top-k alternatives for each prompt token
        )

        outputs = self.llm.generate(
            prompts=[full_prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        output = outputs[0]

        # The prompt_logprobs give us the logprobs for each token in the prompt
        prompt_logprobs = output.prompt_logprobs

        # Find where the updated_doc starts in the sequence
        # This is a bit tricky - we need to tokenize to find the boundary
        tokenized_full = self.tokenizer.encode(full_prompt)
        tokenized_updated = self.tokenizer.encode(updated_doc, add_special_tokens=False)

        # Find the start of updated_doc in the full sequence
        # Simple approach: assume it's at the end
        updated_start = len(tokenized_full) - len(tokenized_updated)

        highlights = []
        char_offset = 0

        # Iterate through the updated_doc tokens
        for i, token_id in enumerate(tokenized_updated):
            position = updated_start + i

            if position < len(prompt_logprobs) and prompt_logprobs[position] is not None:
                logprobs_dict = prompt_logprobs[position]

                # Get the actual token's logprob
                if token_id in logprobs_dict:
                    token_logprob = logprobs_dict[token_id].logprob
                    token_loss = -float(token_logprob)
                else:
                    # Token not in top-k, assign high loss
                    token_loss = 10.0

                # Get top-k alternatives
                top_k_items = sorted(
                    logprobs_dict.items(),
                    key=lambda x: x[1].logprob,
                    reverse=True
                )[:k]

                top_k_tokens = [
                    self.tokenizer.decode([tid])
                    for tid, _ in top_k_items
                ]

                # Get actual token text
                token_text = self.tokenizer.decode([token_id])

                highlights.append({
                    'start': char_offset,
                    'end': char_offset + len(token_text),
                    'token': token_text,
                    'token_loss': token_loss,
                    'most_likely_token': top_k_tokens[0] if top_k_tokens else token_text,
                    'topk_tokens': top_k_tokens,
                })

                char_offset += len(token_text)

        return highlights


# Example usage and testing
if __name__ == "__main__":
    # Initialize with a model
    model_name = "google/gemma-2-2b-it"  # Smaller model for testing

    inference = VLLMInference(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # Test next-token prediction
    prompt = "Rewrite this text to be more concise."
    original_doc = "The quick brown fox jumps over the lazy dog."
    doc_in_progress = "The brown fox"

    print("=== Two-stage approach ===")
    predictions, logprobs = inference.get_next_token_predictions(
        prompt, original_doc, doc_in_progress, k=5, n_lookahead=2
    )

    print("Next token predictions:")
    for pred, logprob in zip(predictions, logprobs):
        print(f"  '{pred}': {logprob:.4f}")

    print("\n=== Beam search approach ===")
    predictions_beam, scores = inference.get_next_token_predictions_beam(
        prompt, original_doc, doc_in_progress, k=5, n_lookahead=2
    )

    print("Beam search predictions:")
    for pred, score in zip(predictions_beam, scores):
        print(f"  '{pred}': {score:.4f}")
