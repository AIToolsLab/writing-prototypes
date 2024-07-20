import streamlit as st
import requests

def landing():
    st.title("Writing Tools Prototypes")
    st.markdown("Click one of the links below to see a prototype in action.")

    st.page_link(st.Page(rewrite_with_predictions), label="Rewrite with predictions", icon="📝")
    st.page_link(highlight_page, label="Highlight locations for possible edits", icon="🖍️")

    st.markdown("*Note*: These services send data to a remote server for processing. The server logs requests. Don't use sensitive or identifiable information on this page.")


def show_token(token):
    token_display = token.replace('\n', '↵').replace('\t', '⇥')
    if token_display.startswith("#"):
        token_display = "\\" + token_display
    return token_display


def get_prompt(default="Rewrite this document to be more clear and concise."):
    # pick a preset prompt or "other"
    with st.popover("Prompt options"):
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
        return prompt


def rewrite_with_predictions():
    st.title("Rewrite with Predictive Text")

    prompt = get_prompt()
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
            token_display = show_token(token)
            st.button(token_display, on_click=append_token, args=(token,), key=i, use_container_width=True)
    

def highlight_edits():
    st.title("Highlight locations for possible edits")
    
    import html
    prompt = get_prompt()
    st.write("Prompt:", prompt)
    doc = st.text_area("Document", placeholder="Paste your document here.")
    updated_doc = st.text_area("Updated Doc", placeholder="Your edited document. Leave this blank to use your original document.")


    response = requests.get("https://tools.kenarnold.org/api/highlights", params=dict(prompt=prompt, doc=doc, updated_doc=updated_doc))
    spans = response.json()['highlights']

    if len(spans) < 2:
        st.write("No spans found.")
        st.stop()

    highest_loss = max(span['token_loss'] for span in spans[1:])
    for span in spans:
        span['loss_ratio'] = span['token_loss'] / highest_loss

    html_out = ''
    for span in spans:
        is_different = span['token'] != span['most_likely_token']
        html_out += '<span style="color: {color}" title="{title}">{orig_token}</span>'.format(
            color="blue" if is_different else "black",
            title=html.escape(span["most_likely_token"]).replace('\n', ' '),
            orig_token=html.escape(span["token"]).replace('\n', '<br>')
        )
    html_out = f"<p style=\"background: white;\">{html_out}</p>"

    st.write(html_out, unsafe_allow_html=True)
    import pandas as pd
    st.write(pd.DataFrame(spans)[['token', 'token_loss', 'most_likely_token', 'loss_ratio']])


rewrite_page = st.Page(rewrite_with_predictions, title="Rewrite with predictions", icon="📝")
highlight_page = st.Page(highlight_edits, title="Highlight locations for possible edits", icon="🖍️")

# Manually specify the sidebar
page = st.navigation([
    st.Page(landing, title="Home", icon="🏠"),
    rewrite_page,
    highlight_page
])
page.run()
