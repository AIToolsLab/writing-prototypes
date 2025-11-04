"""
Custom inference implementation using llama.cpp.

This module provides next-token prediction with lookahead using llama.cpp's
efficient KV cache management. The key advantage is that llama.cpp handles
cache management internally, so we can run multiple generations in rapid
sequence without manually managing cache duplication.
"""

from typing import List, Tuple, Optional
import numpy as np
from llama_cpp import Llama


class LlamaCppInference:
    """Inference engine using llama.cpp."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,  # -1 means offload all layers to GPU
        verbose: bool = False,
    ):
        """
        Initialize llama.cpp model.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            verbose: Whether to print verbose output
        """
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            logits_all=True,  # Enable logits for all tokens
        )

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        """Tokenize text."""
        return self.model.tokenize(text.encode('utf-8'), add_bos=add_bos)

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens."""
        return self.model.detokenize(tokens).decode('utf-8', errors='ignore')

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

        This implementation uses llama.cpp's save_state/load_state to efficiently
        share the KV cache across branches. The prompt is computed once and the
        state is saved, then restored for each branch.

        Args:
            prompt: System prompt or instruction
            original_doc: Original document to rewrite
            doc_in_progress: Partially rewritten document
            k: Number of top predictions to return
            n_lookahead: Number of tokens to predict ahead for each branch

        Returns:
            Tuple of (decoded_tokens, logits) where decoded_tokens includes
            the next token plus lookahead tokens
        """
        # Build the full prompt using chat template
        # For now, simple concatenation - we'll improve this
        full_prompt = f"{prompt}\n\nOriginal:\n{original_doc}\n\nRewrite:\n{doc_in_progress}"

        # Tokenize the prompt
        prompt_tokens = self.tokenize(full_prompt)

        # Evaluate the prompt once to fill KV cache
        self.model.reset()
        self.model.eval(prompt_tokens)

        # Get logits for the next token
        logits = self.model.scores[len(prompt_tokens) - 1]

        # Get top-k tokens
        top_k_indices = np.argsort(logits)[-k:][::-1]
        top_k_logits = [logits[i] for i in top_k_indices]

        # Save the state after evaluating the prompt
        # This captures the KV cache, logits, and model state
        saved_state = self.model.save_state()

        # Now for each top-k token, generate lookahead
        decoded_sequences = []

        for token_id in top_k_indices:
            # Restore the state (KV cache) from after the prompt
            # This is much more efficient than re-evaluating the prompt!
            self.model.load_state(saved_state)

            # Now generate lookahead from this branch
            lookahead_tokens = [token_id]
            for _ in range(n_lookahead - 1):
                # Evaluate the next token
                self.model.eval([lookahead_tokens[-1]])
                next_logits = self.model.scores[-1]
                next_token = int(np.argmax(next_logits))
                lookahead_tokens.append(next_token)

            # Decode the sequence
            decoded = self.detokenize(lookahead_tokens)
            decoded_sequences.append(decoded)

        return decoded_sequences, top_k_logits

    def get_highlights(
        self,
        prompt: str,
        original_doc: str,
        updated_doc: str,
        k: int = 5,
    ) -> List[dict]:
        """
        Get edit suggestions by comparing model predictions with actual tokens.

        This computes logprobs for the updated_doc and identifies positions
        where the model would predict a different token.

        Args:
            prompt: System prompt or instruction
            original_doc: Original document
            updated_doc: User's edited version
            k: Number of alternative tokens to return per position

        Returns:
            List of highlight spans with token alternatives
        """
        # Build the full prompt
        full_prompt = f"{prompt}\n\nOriginal:\n{original_doc}\n\nRewrite:\n{updated_doc}"

        # Tokenize
        full_tokens = self.tokenize(full_prompt)
        updated_doc_tokens = self.tokenize(updated_doc, add_bos=False)

        # Find where updated_doc starts in the full sequence
        # This is a simplified approach - in practice, we'd need more robust alignment
        updated_doc_start = len(full_tokens) - len(updated_doc_tokens)

        # Evaluate the full sequence
        self.model.reset()
        self.model.eval(full_tokens)

        # Collect highlights
        highlights = []
        char_offset = 0

        for i, token_id in enumerate(updated_doc_tokens):
            position = updated_doc_start + i

            if position > 0 and position < len(full_tokens):
                # Get logits at this position
                logits = self.model.scores[position - 1]

                # Compute log probability of actual token
                logprobs = logits - np.log(np.sum(np.exp(logits)))
                token_logprob = logprobs[token_id]
                token_loss = -float(token_logprob)

                # Get top-k alternatives
                top_k_indices = np.argsort(logits)[-k:][::-1]
                top_k_tokens = [self.detokenize([int(idx)]) for idx in top_k_indices]

                # Get actual token text
                token_text = self.detokenize([token_id])

                highlights.append({
                    'start': char_offset,
                    'end': char_offset + len(token_text),
                    'token': token_text,
                    'token_loss': token_loss,
                    'most_likely_token': top_k_tokens[0],
                    'topk_tokens': top_k_tokens,
                })

                char_offset += len(token_text)

        return highlights


def create_chat_prompt(prompt: str, original_doc: str, doc_in_progress: str) -> str:
    """
    Create a chat-formatted prompt.

    This is a simplified version - you'd want to use proper chat templates
    for each model type.
    """
    return f"""<start_of_turn>user
{prompt}

Original:
{original_doc}

Rewrite:
<end_of_turn>
<start_of_turn>model
{doc_in_progress}"""


# Example usage and testing
if __name__ == "__main__":
    # This would require a GGUF model file
    model_path = "/path/to/model.gguf"  # Update this path

    inference = LlamaCppInference(
        model_path=model_path,
        n_gpu_layers=-1,
    )

    # Test next-token prediction
    prompt = "Rewrite this text to be more concise."
    original_doc = "The quick brown fox jumps over the lazy dog."
    doc_in_progress = "The brown fox"

    predictions, logits = inference.get_next_token_predictions(
        prompt, original_doc, doc_in_progress, k=5
    )

    print("Next token predictions:")
    for pred, logit in zip(predictions, logits):
        print(f"  {pred}: {logit:.4f}")
