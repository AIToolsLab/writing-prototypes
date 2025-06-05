import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import custom_llm_inference
from transformers.cache_utils import DynamicCache

@pytest.fixture
def model_and_tokenizer():
    model_name = 'google/gemma-2-2b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16
    )
    return model, tokenizer

@pytest.fixture
def sample_inputs():
    doc = "The quick brown fox loves to jump over lazy dogs."
    prompt = "Rewrite this document to make more sense."
    doc_in_progress = "Sure, here's the document rewritten as requested:\n\nA fox,"
    return doc, prompt, doc_in_progress

def test_get_next_token_predictions(model_and_tokenizer, sample_inputs):
    model, tokenizer = model_and_tokenizer
    doc, prompt, doc_in_progress = sample_inputs

    predictions = custom_llm_inference.get_next_token_predictions_slow(
        model, tokenizer, doc, prompt, doc_in_progress=doc_in_progress, k=5
    )

    assert len(predictions) == 2  # Should return (token_texts, logits)
    assert len(predictions[0]) == 5  # Should return k=5 predictions
    assert predictions[1].shape[1] == model.config.vocab_size

def test_get_tokenized_chat(model_and_tokenizer, sample_inputs):
    model, tokenizer = model_and_tokenizer
    doc, prompt, _ = sample_inputs

    tokenized_chat = custom_llm_inference.get_tokenized_chat(tokenizer, prompt, doc)

    assert isinstance(tokenized_chat, torch.Tensor)
    assert tokenized_chat.dim() == 1
    assert tokenized_chat.dtype == torch.int64

def test_highlights(model_and_tokenizer, sample_inputs):
    model, tokenizer = model_and_tokenizer
    doc, prompt, updated_doc = sample_inputs

    highlights = custom_llm_inference.get_highlights_inner(
        model, tokenizer, doc, prompt, updated_doc=updated_doc, k=5
    )

    assert isinstance(highlights, list)
    assert len(highlights) > 0
    for h in highlights:
        assert h['start'] >= 0
        assert h['end'] >= h['start']
        assert isinstance(h['token'], str)
        assert isinstance(h['token_loss'], float)
        assert isinstance(h['most_likely_token'], str)
        assert isinstance(h['topk_tokens'], list)

def compare_lookahead_predictions(model, tokenizer, doc, prompt, doc_in_progress, k=5):
    """
    Extracts and compares the next token predictions between the fast method and slow method.
    Returns the differences between the two approaches for analysis.
    """
    # Get predictions from the fast method (using cache)
    fast_tokens, fast_logits = custom_llm_inference.get_next_token_predictions_inner(
        model, tokenizer, doc, prompt, doc_in_progress, k
    )

    # Get predictions from the slow method (recomputing for each token)
    slow_tokens, slow_logits = custom_llm_inference.get_next_token_predictions_slow(
        model, tokenizer, doc, prompt, doc_in_progress, k
    )

    # Compare the decoded tokens (this is what users will see)
    token_matches = [fast == slow for fast, slow in zip(fast_tokens, slow_tokens)]

    # Calculate the difference in logits for most likely next tokens
    fast_most_likely = fast_logits.argmax(dim=-1)
    slow_most_likely = slow_logits.argmax(dim=-1)
    logit_match = torch.eq(fast_most_likely, slow_most_likely).cpu().numpy()

    # Calculate numerical difference in logits
    logit_diff_norm = torch.linalg.vector_norm((fast_logits - slow_logits).to(torch.float32), dim=1).cpu().numpy()

    return {
        "fast_tokens": fast_tokens,
        "slow_tokens": slow_tokens,
        "token_matches": token_matches,
        "token_match_all": all(token_matches),
        "logit_match": logit_match,
        "logit_diff_norm": logit_diff_norm
    }

def test_lookahead_token_consistency(model_and_tokenizer, sample_inputs):
    """
    Test that demonstrates the potential issue with cache position indices
    when generating lookahead tokens.
    """
    model, tokenizer = model_and_tokenizer
    doc, prompt, doc_in_progress = sample_inputs

    results = compare_lookahead_predictions(model, tokenizer, doc, prompt, doc_in_progress)

    # Check if the tokens are the same
    assert results["token_match_all"], (
        f"Fast and slow methods produced different tokens.\n"
        f"Fast: {results['fast_tokens']}\n"
        f"Slow: {results['slow_tokens']}"
    )

    # Check if the most likely next tokens based on logits are the same
    assert all(results["logit_match"]), (
        f"Fast and slow methods predicted different most likely next tokens"
    )

    # Check that the logit differences are minimal
    # This might fail if there's a bug in the cache position indices
    assert all(diff < 1e-4 for diff in results["logit_diff_norm"]), (
        f"Significant difference in logits between fast and slow methods: {results['logit_diff_norm']}"
    )


def test_get_lookahead_sequences(model_and_tokenizer, sample_inputs):
    model, tokenizer = model_and_tokenizer
    doc, _, _ = sample_inputs
    hypotheses = tokenizer(doc, return_tensors='pt').input_ids

    assert len(hypotheses.shape) == 2 and hypotheses.shape[0] == 1, "Expected input shape (1, sequence_length)"

    sequences, next_token_logits = custom_llm_inference.get_lookahead_sequences(
        model, tokenizer, hypotheses, 5, model.device
    )

    assert sequences.dim() == 2  # Should be (batch_size, sequence_length)
    assert sequences.shape[0] == 5  # Should have k branches
    assert sequences.shape[1] == 3 # Third-next-token should be added

    # k predictions per token
    vocab_size = model.config.vocab_size
    assert next_token_logits.shape == (5, vocab_size), f"{next_token_logits.shape=}, {n_branch_tokens=}, {vocab_size=}"
