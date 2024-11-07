import streamlit as st
import requests

def landing():
    st.title("Writing Tools Prototypes")
    st.markdown("Click one of the links below to see a prototype in action.")

    st.page_link(rewrite_page, label="Rewrite with predictions", icon="📝")
    st.page_link(highlight_page, label="Highlight locations for possible edits", icon="🖍️")
    st.page_link(generate_page, label="Generate revisions", icon="🔄")

    st.markdown("*Note*: These services send data to a remote server for processing. The server logs requests. Don't use sensitive or identifiable information on this page.")


def show_token(token):
    token_display = token.replace('\n', '↵').replace('\t', '⇥')
    # Escape Markdown
    for c in "\\`*_{}[]()#+-.!":
        token_display = token_display.replace(c, "\\" + c)
    return token_display


def get_prompt(*, include_generation_options, default="Rewrite this document to be more clear and concise."):
    # pick a preset prompt or "other"
    generation_options = [
        "Summarize this document in one sentence.",
        "Translate this document into Spanish.",
        "Write a concise essay according to this outline.",
        "Write a detailed essay according to this outline.",
    ]
    with st.popover("Prompt options"):
        prompt_options = [
            "Rewrite this document to be ...",
            *(generation_options if include_generation_options else []),
            "Other"
        ]
        prompt = st.radio("Prompt", prompt_options, help="Instructions for what the bot should do.")
        if prompt.startswith("Rewrite this document to be"):
            rewrite_adjs = ["clear and concise", "more detailed and engaging", "more formal and professional", "more casual and conversational", "more technical and precise", "more creative and imaginative", "more persuasive and compelling"]
            prompt = "Rewrite this document to be " + st.radio("to be ...", rewrite_adjs) + "."
        elif prompt == "Other":
            prompt = st.text_area("Prompt", "Rewrite this document to be more clear and concise.")
        return prompt


@st.cache_data
def get_preds_api(prompt, original_doc, rewrite_in_progress, k=5):
    response = requests.get("https://tools.kenarnold.org/api/next_token", params=dict(prompt=prompt, original_doc=original_doc, doc_in_progress=rewrite_in_progress, k=k))
    response.raise_for_status()
    return response.json()['next_tokens']


def rewrite_with_predictions():
    st.title("Rewrite with Predictive Text")

    prompt = get_prompt(include_generation_options=True)
    st.write("Prompt:", prompt)

    cols = st.columns(2)
    with cols[0]:
        doc = st.text_area("Document", "", placeholder="Paste your document here.", height=300)
        st.button("Update document")
    with cols[1]:
        rewrite_in_progress = st.text_area("Rewrite in progress", key='rewrite_in_progress', value="", placeholder="Clicking the buttons below will update this field. You can also edit it directly; press Ctrl+Enter to apply changes.", height=300)

    if doc.strip() == "" and rewrite_in_progress.strip() == "":
        # Allow partial rewrites as a hack to enable autocomplete from the prompt
        st.stop()

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
    

@st.cache_data
def get_highlights(prompt, doc, updated_doc):
    response = requests.get("https://tools.kenarnold.org/api/highlights", params=dict(prompt=prompt, doc=doc, updated_doc=updated_doc))
    return response.json()['highlights']


def highlight_edits():
    st.title("Highlight locations for possible edits")

    import html
    prompt = get_prompt(include_generation_options=False)
    st.write("Prompt:", prompt)
    cols = st.columns(2)
    with cols[0]:
        doc = st.text_area("Document", "Deep learning neural network technology advances are pretty cool if you are careful to use it in ways that don't take stuff from people.", height=300)
    with cols[1]:
        updated_doc = st.text_area("Updated Doc", placeholder="Your edited document. Leave this blank to use your original document.", height=300)

    spans = get_highlights(prompt, doc, updated_doc)

    if len(spans) < 2:
        st.write("No spans found.")
        st.stop()

    highest_loss = max(span['token_loss'] for span in spans[1:])
    for span in spans:
        span['loss_ratio'] = span['token_loss'] / highest_loss

    num_different = sum(span['token'] != span['most_likely_token'] for span in spans)
    loss_ratios_for_different = [span['loss_ratio'] for span in spans if span['token'] != span['most_likely_token']]
    loss_ratios_for_different.sort(reverse=True)

    if num_different == 0:
        st.write("No possible edits found.")
        st.stop()
    
    num_to_show = st.slider("Number of edits to show", 1, num_different, value=num_different // 2)
    min_loss = loss_ratios_for_different[num_to_show - 1]

    html_out = ''
    for span in spans:
        show = span['token'] != span['most_likely_token'] and span['loss_ratio'] >= min_loss
        hover = f'<span style="position: absolute; top: -10px; left: 5px; font-size: 10px; min-width:6em; line-height: 1; color: grey; transform-origin: left; transform: rotate(-15deg)">{span["most_likely_token"]}</span>'
        html_out += '<span style="position: relative; color: {color}" title="{title}">{hover}{orig_token}</span>'.format(
            color="blue" if show else "black",
            title=html.escape(span["most_likely_token"]).replace('\n', ' ') if show else '',
            orig_token=html.escape(span["token"]).replace('\n', '<br>'),
            hover=hover if show else ''
        )
    html_out = f"<p style=\"background: white; line-height: 2.5;\">{html_out}</p>"

    st.write(html_out, unsafe_allow_html=True)
    if st.checkbox("Show details"):
        import pandas as pd
        st.write(pd.DataFrame(spans)[['token', 'token_loss', 'most_likely_token', 'loss_ratio']])
        st.write("Token loss is the difference between the original token and the most likely token. The loss ratio is the token loss divided by the highest token loss in the document.")

def get_revised_docs(prompt, doc, n):
    response = requests.get("https://tools.kenarnold.org/api/gen_revisions", params=dict(prompt=prompt, doc=doc, n=n))
    return response.json()


def generate_revisions():
    st.title("Generate revised document")

    import html
    prompt = get_prompt(include_generation_options=False)
    st.write("Prompt:", prompt)
    doc = st.text_area("Document", "", height=300)

    revised_docs = get_revised_docs(prompt, doc, n=5)['revised_docs']

    tabs = st.tabs([f"Draft {i+1}" for i in range(len(revised_docs))])
    for i, tab in enumerate(tabs):
        with tab:
            st.write(revised_docs[i]['doc_text'])


rewrite_page = st.Page(rewrite_with_predictions, title="Rewrite with predictions", icon="📝")
highlight_page = st.Page(highlight_edits, title="Highlight locations for possible edits", icon="🖍️")
generate_page = st.Page(generate_revisions, title="Generate revisions", icon="🔄")

# Manually specify the sidebar
page = st.navigation([
    st.Page(landing, title="Home", icon="🏠"),
    highlight_page,
    rewrite_page,
    generate_page
])
page.run()
