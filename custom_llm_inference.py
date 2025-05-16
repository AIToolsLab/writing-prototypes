import torch
from transformers.cache_utils import DynamicCache


def catch_and_report_memory_exceptions(func):
    """
    Decorator to catch and report memory exceptions.
    """
    def wrapper(*args, **kwargs):
        # https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
        torch.cuda.memory._record_memory_history()
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
            print(f"Memory error: {e}")
            # clear frames in the traceback to avoid memory leak
            import traceback, sys
            traceback.clear_frames(sys.exc_info()[2])
            raise e
    return wrapper


def get_tokenized_chat(tokenizer, prompt, doc):
    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n{doc}",
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
    return tokenized_chat


def tokenize_doc_in_progress(tokenizer, doc_in_progress):
    if len(doc_in_progress) == 0:
        # Some tokenizers give tensors of the wrong dtype if the input is empty
        return torch.empty(0, dtype=torch.int64)

    doc_in_progress_ids = tokenizer(
        doc_in_progress, return_tensors='pt')['input_ids'][0]

    # strip the first token, the "beginning of document" token
    # TODO: make this robust to switching models
    # since some models will use different special tokens
    doc_in_progress_ids = doc_in_progress_ids[1:]
    return doc_in_progress_ids


@catch_and_report_memory_exceptions
def get_highlights_inner(model, tokenizer, doc, prompt, updated_doc, k):
    tokenized_chat = get_tokenized_chat(tokenizer, prompt, doc)
    assert len(tokenized_chat.shape) == 1

    if updated_doc is None or len(updated_doc.strip()) == 0:
        updated_doc = doc
    updated_doc_ids = tokenize_doc_in_progress(tokenizer, updated_doc)

    joined_ids = torch.cat([tokenized_chat, updated_doc_ids])

    # Compute the next-token logits for the entire document
    with torch.no_grad():
        logits = model(joined_ids[None].to(model.device)).logits[0].cpu()
    
    highlights = []
    length_so_far = 0
    for idx in range(len(tokenized_chat), len(joined_ids)):
        probs = logits[idx - 1].softmax(dim=-1)
        token_id = joined_ids[idx]
        token = tokenizer.decode(token_id)
        token_loss = -probs[token_id].log().item()
        topk_tokens = probs.topk(k).indices.cpu().numpy().tolist()
        topk_tokens_decoded = tokenizer.batch_decode(topk_tokens, skip_special_tokens=True)
        highlights.append(dict(
            start=length_so_far,
            end=length_so_far + len(token),
            token=token,
            token_loss=token_loss,
            most_likely_token=topk_tokens_decoded[0],
            topk_tokens=topk_tokens_decoded,
        ))
        length_so_far += len(token)
    return highlights


@catch_and_report_memory_exceptions
def get_lookahead_sequences(model, tokenizer, hypotheses, n_branch_tokens, device):
    """
    For each of the n_branch_tokens next tokens, generate most-likely next tokens and append back on.
    """
    assert len(hypotheses.shape) == 2
    assert hypotheses.shape[0] == 1
    n_tokens_so_far = hypotheses.shape[1]
    past_key_values = DynamicCache()

    with torch.no_grad():
        model_outs_onestep = model(hypotheses, output_hidden_states=True, past_key_values=past_key_values)

    branch_tokens = model_outs_onestep.logits[0, -1].topk(n_branch_tokens).indices

    # split the cache into n_branch_tokens reps. We pretend we're doing a "Beam search"...
    past_key_values.reorder_cache(torch.zeros((n_branch_tokens,), dtype=torch.long, device=device))

    # Now call the model again, passing the kv cache, so we can continue generating.
    # Each of the n_branch_tokens next tokens will be considered as one sequence in a "batch".
    next_tokens_as_batch = branch_tokens.unsqueeze(1)
    assert next_tokens_as_batch.shape == (n_branch_tokens, 1)

    position_id_for_final_token = n_tokens_so_far
    cache_position = torch.full((1,), position_id_for_final_token, dtype=int, device=device)
    with torch.no_grad():
        model_outs = model(
            next_tokens_as_batch,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
            # the cache surprisingly doesn't know the position of the last token
            cache_position=cache_position
        )
    
    # Grab the single most likely token from each of the n_branch_tokens sequences
    next_token_logits = model_outs.logits[:, -1]
    vocab_size = model.config.vocab_size
    assert next_token_logits.shape == (n_branch_tokens, vocab_size), f"{next_token_logits.shape=}, {n_branch_tokens=}, {vocab_size=}"
    most_likely_token_ids = next_token_logits.argmax(dim=-1)

    # Stick them at the end of the branch tokens.
    assert most_likely_token_ids.shape == (n_branch_tokens,)
    lookahead_sequences = torch.cat([
        branch_tokens.unsqueeze(1),
        most_likely_token_ids.unsqueeze(1)
    ], dim=1)
    assert lookahead_sequences.shape == (n_branch_tokens, 2)
    return lookahead_sequences, next_token_logits


@catch_and_report_memory_exceptions
def get_next_token_predictions_inner(
        model, tokenizer, original_doc, prompt, doc_in_progress, k):

    tokenized_chat = get_tokenized_chat(tokenizer, prompt, original_doc)
    doc_in_progress_ids = tokenize_doc_in_progress(tokenizer, doc_in_progress)

    device = model.device

    joined_ids = torch.cat([tokenized_chat, doc_in_progress_ids])
    hypotheses = joined_ids[None].to(model.device)

    # Alternative approach: chat templates
    tokenized_chat = tokenizer.apply_chat_template([
        {"role": "user", "content": f"{prompt}\n\n{original_doc}"},
        {"role": "assistant", "content": doc_in_progress}
    ], tokenize=True, return_tensors="pt", continue_final_message=True).to(model.device)

    # Compare them
    if tokenized_chat.shape == hypotheses.shape and torch.all(tokenized_chat == hypotheses):
        print("Tokenized chat and hypotheses match")
    else:
        print("FAIL: Tokenized chat and hypotheses do not match!")
        print(f"{tokenized_chat=}")
        print(f"{hypotheses=}")

    lookahead_sequences, next_token_logits = get_lookahead_sequences(
        model, tokenizer, hypotheses, k, device)

    decoded_next_tokens = tokenizer.batch_decode(lookahead_sequences, skip_special_tokens=True)
    return decoded_next_tokens, next_token_logits


@catch_and_report_memory_exceptions
def get_next_token_predictions_slow(
        model, tokenizer, original_doc, prompt, doc_in_progress, k):

    tokenized_chat = get_tokenized_chat(tokenizer, prompt, original_doc)
    doc_in_progress_ids = tokenize_doc_in_progress(tokenizer, doc_in_progress)

    joined_ids = torch.cat([tokenized_chat, doc_in_progress_ids])
    hypotheses = joined_ids[None].to(model.device)

    # For each of the k next tokens, generate most-likely next tokens and append back on until we
    # reach a token with a space

    with torch.no_grad():
        model_outs = model(hypotheses, output_hidden_states=True)

    next_token_logits = model_outs.logits[0, -1]
    branch_tokens = next_token_logits.topk(k).indices

    # Slow mode: concat the branch tokens to the hypotheses.
    # Then call the model on the full sequence.
    # This is slow because the beginning of the sequence is re-processed each time.

    hypotheses_with_next_tokens = torch.cat([
        torch.repeat_interleave(hypotheses, k, dim=0),
        branch_tokens.unsqueeze(1)
    ], dim=1)
    assert hypotheses_with_next_tokens.shape == (k, len(joined_ids) + 1)

    with torch.no_grad():
        model_outs = model(hypotheses_with_next_tokens)
    
    # Grab the single most likely token from each of the k sequences
    next_token_logits = model_outs.logits[:, -1]
    vocab_size = model.config.vocab_size
    assert next_token_logits.shape == (k, vocab_size), f"{next_token_logits.shape=}, {k=}, {vocab_size=}"
    most_likely_token_ids = next_token_logits.argmax(dim=-1)

    # Stick them at the end of the branch tokens.
    assert most_likely_token_ids.shape == (k,)
    lookahead_sequences = torch.cat([
        branch_tokens.unsqueeze(1),
        most_likely_token_ids.unsqueeze(1)
    ], dim=1)
    assert lookahead_sequences.shape == (k, 2)

    decoded_next_tokens = tokenizer.batch_decode(lookahead_sequences, skip_special_tokens=True)
    return decoded_next_tokens, next_token_logits



def continue_messages_inner(model, tokenizer, messages, n_branch_tokens, n_future_tokens):
    # Note: we're ignoring n_future_tokens right now since the old implementation was buggy.
    device = model.device

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", continue_final_message=True).to(model.device)
    print(tokenizer.batch_decode(tokenized_chat, skip_special_tokens=False))

    lookahead_sequences, next_token_logits = get_lookahead_sequences(
        model, tokenizer, tokenized_chat, n_branch_tokens, device)

    generated_docs = tokenizer.batch_decode(lookahead_sequences, skip_special_tokens=True)
    return generated_docs