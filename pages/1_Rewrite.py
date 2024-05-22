import streamlit as st
import pandas as pd
import html


prompt = st.text_area("Prompt", "Rewrite this document to be more clear and concise.", placeholder="Instructions for what the bot should do.")
doc = st.text_area("Document", "", placeholder="Paste your document here.")
st.button("Update document")
rewrite_in_progress = st.text_area("Rewrite in progress", key='rewrite_in_progress', value="", placeholder="Clicking the buttons below will update this field. You can also edit it directly; press Ctrl+Enter to apply changes.")

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
 
allow_multi_word = st.checkbox("Allow multi-word predictions", value=False)

for i, (col, token) in enumerate(zip(st.columns(len(tokens)), tokens)):
    with col:
        if not allow_multi_word and ' ' in token[1:]:
            token = token[0] + token[1:].split(' ', 1)[0]
        st.button(token, on_click=append_token, args=(token,), key=i)
 