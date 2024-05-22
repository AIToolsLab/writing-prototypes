import streamlit as st
import pandas as pd
import html


prompt = st.text_area("Prompt", "Rewrite this document to be more clear and concise.")
doc = st.text_area("Document", "This is a document that I would like to have rewritten to be more concise.")
rewrite_in_progress = st.text_area("Rewrite in progress", key='rewrite_in_progress', value="")


def get_preds_api(prompt, original_doc, rewrite_in_progress, k=5):
    import requests
    response = requests.get("https://tools.kenarnold.org/api/next_token", params=dict(prompt=prompt, original_doc=original_doc, doc_in_progress=rewrite_in_progress, k=k))
    return response.json()['next_tokens']

tokens = get_preds_api(prompt, doc, rewrite_in_progress)

def append_token(word):
    st.session_state['rewrite_in_progress'] = (
        st.session_state['rewrite_in_progress'] + word
    )
 
for col, token in zip(st.columns(len(tokens)), tokens):
    with col:
        st.button(token, on_click=append_token, args=(token,))
 