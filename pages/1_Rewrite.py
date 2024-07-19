import streamlit as st
import requests


st.title("Rewrite with Predictive Text")

# pick a preset prompt or "other"
prompt_options = [
    "Rewrite this document to be ...",
    "Summarize this document in one sentence.",
    "Translate this document into Spanish.",
    "Other"
]
prompt = st.radio("Prompt", prompt_options, help="Instructions for what the bot should do.")
if prompt.startswith("Rewrite this document to be"):
    rewrite_adjs = ["clear and concise", "more detailed and engaging", "more formal and professional", "more casual and conversational", "more technical and precise", "more creative and imaginative", "more persuasive and compelling"]
    prompt = "Rewrite this document to be " + st.radio("to be ...", rewrite_adjs) + "."
elif prompt == "Other":
    prompt = st.text_area("Prompt", "Rewrite this document to be more clear and concise.")
st.write("Prompt:", prompt)
doc = st.text_area("Document", "", placeholder="Paste your document here.", height=300)
st.button("Update document")
rewrite_in_progress = st.text_area("Rewrite in progress", key='rewrite_in_progress', value="", placeholder="Clicking the buttons below will update this field. You can also edit it directly; press Ctrl+Enter to apply changes.", height=300)

if doc.strip() == "" and rewrite_in_progress.strip() == "":
    # Allow partial rewrites as a hack to enable autocomplete from the prompt
    st.stop()

def get_preds_api(prompt, original_doc, rewrite_in_progress, k=5):
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
 