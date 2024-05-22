import streamlit as st
import pandas as pd
import html


prompt = st.text_area("Prompt", "Rewrite this document to be more clear and concise.")
doc = st.text_area("Document", "", help="Paste your document here")#"Revolutionize your paradigm-shifting synergy with our cutting-edge quantum cloud solutions!")
rewrite_in_progress = st.text_area("Rewrite in progress", key='rewrite_in_progress', value="")

if doc.strip() == "" and rewrite_in_progress.strip() == "":
    # Allow partial rewrites as a hack to enable autocomplete from the prompt
    st.stop()

def get_preds_api(prompt, original_doc, rewrite_in_progress, k=5):
    import requests
    response = requests.get("https://tools.kenarnold.org/api/next_token", params=dict(prompt=prompt, original_doc=original_doc, doc_in_progress=rewrite_in_progress, k=k))
    response.raise_for_status()
    return response.json()['next_tokens']

tokens = get_preds_api(prompt, doc, rewrite_in_progress)

def append_token(word):
    st.session_state['rewrite_in_progress'] = (
        st.session_state['rewrite_in_progress'] + word
    )
 
for col, token in zip(st.columns(len(tokens)), tokens):
    with col:
        st.button(token, on_click=append_token, args=(token,))
 