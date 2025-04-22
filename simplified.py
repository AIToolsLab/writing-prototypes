import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache

USE_GPU = torch.cuda.is_available()

@st.cache_resource
def load_model():
    import torch

    model_name = 'google/gemma-2-9b-it'

    dtype = torch.bfloat16 if USE_GPU else torch.float16

    llm = {
        'tokenizer': AutoTokenizer.from_pretrained(model_name),
        'model': AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if USE_GPU else "cpu",
            torch_dtype=dtype,
            attn_implementation='eager'
        )
    }
    llm['model'].eval()
    return llm



def type_assistant_response():
    if 'messages' not in st.session_state or st.button("Start a new conversation"):
        st.session_state['messages'] = [{"role": "user", "content": ""}]
        st.session_state['msg_in_progress'] = ""
    messages = st.session_state.messages

    def rewind_to(i):
        st.session_state.messages = st.session_state.messages[:i+1]
        st.session_state['msg_in_progress'] = st.session_state.messages[-1]['content']

    for i, message in enumerate(st.session_state.messages[:-1]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.button("Edit", on_click=rewind_to, args=(i,), key=f"rewind_to_{i}")

    # Display message-in-progress in chat message container
    last_role = messages[-1]["role"]
    with st.chat_message(last_role):
        label = "Your message" if last_role == "user" else "Assistant response"
        msg_in_progress = st.text_area(label, placeholder="Clicking the buttons below will update this field. You can also edit it directly; press Ctrl+Enter to apply changes.", height=300, key="msg_in_progress")
        if msg_in_progress is None:
            msg_in_progress = ""

        messages[-1]['content'] = msg_in_progress

        def append_token(word):
            messages[-1]['content'] = st.session_state['msg_in_progress'] = (
                msg_in_progress + word
            )
        
        allow_multi_word = st.checkbox("Allow multi-word predictions", value=False)

        response = continue_messages(
            messages=messages,
            n_branch_tokens=5,
            n_future_tokens=2
        )

        continuations = response['continuations']
        for i, (col, continuation) in enumerate(zip(st.columns(len(continuations)), continuations)):
            token = continuation['doc_text']
            with col:
                if not allow_multi_word and ' ' in token[1:]:
                    token = token[0] + token[1:].split(' ', 1)[0]

                # if not allow_multi_word:
                #     import re
                #     split_result = re.split(r'(\s+)', token, maxsplit=1)
                #     assert len(split_result) == 3
                #     before_ws, token, after_ws = split_result
                #     print(repr(split_result))
                #     if before_ws != '':
                #         token = before_ws
                token_display = show_token(token)
                st.button(token_display, on_click=append_token, args=(token,), key=i, use_container_width=True)
        
        def send_message():
            other_role = "assistant" if last_role == "user" else "user"
            st.session_state['messages'].append({"role": other_role, "content": ""})
            st.session_state['msg_in_progress'] = ""
        st.button("Send", on_click=send_message)
        
def show_token(token: str, escape_markdown=True) -> str:
    token_display = token.replace('\n', '↵').replace('\t', '⇥')
    if escape_markdown:
        for c in "\\`*_{}[]()#+-.!":
            token_display = token_display.replace(c, "\\" + c)
    return token_display


def continue_messages(messages, n_branch_tokens, n_future_tokens):

    messages = [{"role": m.role, "content": m.content} for m in messages]
    if len(messages) == 0:
        raise ValueError("At least one message must be provided.")

    llm = load_model()
    model = llm['model']
    tokenizer = llm['tokenizer']

    generated_docs = continue_messages_inner(model, tokenizer, messages, n_branch_tokens, n_future_tokens)

    return {
        'continuations': [dict(doc_text=doc) for doc in generated_docs]
    }


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


def continue_messages_inner(model, tokenizer, messages, n_branch_tokens, n_future_tokens):
    # Note: we're ignoring n_future_tokens right now since the old implementation was buggy.
    device = model.device

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", continue_final_message=True).to(model.device)
    print(tokenizer.batch_decode(tokenized_chat, skip_special_tokens=False))

    lookahead_sequences, next_token_logits = get_lookahead_sequences(
        model, tokenizer, tokenized_chat, n_branch_tokens, device)

    generated_docs = tokenizer.batch_decode(lookahead_sequences, skip_special_tokens=True)
    return generated_docs

type_assistant_response()

