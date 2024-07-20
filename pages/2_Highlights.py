import streamlit as st
import pandas as pd
import html

@st.cache_resource
def get_tokenizer(model_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name).from_pretrained(model_name)

@st.cache_resource
def get_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    print(f"Loaded model, {model.num_parameters():,d} parameters.")
    return model

def get_spans_local(prompt, doc, updated_doc):
    import torch
    
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)


    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n{doc}",
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
    assert len(tokenized_chat.shape) == 1

    if len(updated_doc.strip()) == 0:
        updated_doc = doc
    updated_doc_ids = tokenizer(updated_doc, return_tensors='pt')['input_ids'][0]
    joined_ids = torch.cat([tokenized_chat, updated_doc_ids[1:]])

    with torch.no_grad():
        logits = model(joined_ids[None].to(model.device)).logits[0].cpu()

    spans = []
    length_so_far = 0
    for idx in range(len(tokenized_chat), len(joined_ids)):
        probs = logits[idx - 1].softmax(dim=-1)
        token_id = joined_ids[idx]
        token = tokenizer.decode(token_id)
        token_loss = -probs[token_id].log().item()
        most_likely_token_id = probs.argmax()
        print(idx, token, token_loss, tokenizer.decode(most_likely_token_id))
        spans.append(dict(
            start=length_so_far,
            end=length_so_far + len(token),
            token=token,
            token_loss=token_loss,
            most_likely_token=tokenizer.decode(most_likely_token_id)
        ))
        length_so_far += len(token)
    return spans
