import streamlit as st
import requests

API_SERVER = "https://tools.kenarnold.org/api"

def landing():
    st.title("Writing Tools Prototypes")
    st.markdown("Click one of the links below to see a prototype in action.")

    st.page_link(rewrite_page, label="Rewrite with predictions", icon="📝")
    st.page_link(highlight_page, label="Highlight locations for possible edits", icon="🖍️")
    st.page_link(generate_page, label="Generate revisions", icon="🔄")
    st.page_link(type_assistant_response_page, label="Type Assistant Response", icon="🔤")

    st.markdown("*Note*: These services send data to a remote server for processing. The server logs requests. Don't use sensitive or identifiable information on this page.")


def show_token(token: str, escape_markdown=True) -> str:
    token_display = token.replace('\n', '↵').replace('\t', '⇥')
    if escape_markdown:
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
    with st.popover("Edit Prompt"):
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
            prompt = st.text_area("Prompt", "Rewrite this document to be clear and concise.")
        return prompt


@st.cache_data
def get_preds_api(prompt, original_doc, rewrite_in_progress, k=5):
    response = requests.get("https://tools.kenarnold.org/api/next_token", params=dict(prompt=prompt, original_doc=original_doc, doc_in_progress=rewrite_in_progress, k=k))
    response.raise_for_status()
    return response.json()['next_tokens']


def rewrite_with_predictions():
    st.title("Rewrite with Predictive Text")

    cols = st.columns(2)
    with cols[0]:
        prompt = get_prompt(include_generation_options=True)
    with cols[1]:
        st.write("Prompt:", prompt)

    cols = st.columns(2)
    with cols[0]:
        doc = st.text_area("Document", "", placeholder="Paste your document here.", height=300)
        st.button("Update document")
    with cols[1]:
        rewrite_in_progress = st.text_area("Rewrite in progress", key='rewrite_in_progress', value="", placeholder="Clicking the buttons below will update this field. You can also edit it directly; press Ctrl+Enter to apply changes.", height=300)
        # strip spaces (but not newlines) to avoid a tokenization issue
        rewrite_in_progress = rewrite_in_progress.rstrip(' ')

    if doc.strip() == "" and rewrite_in_progress.strip() == "":
        # Allow partial rewrites as a hack to enable autocomplete from the prompt
        st.stop()

    tokens = get_preds_api(prompt, doc, rewrite_in_progress)

    def append_token(word):
        st.session_state['rewrite_in_progress'] = (
            rewrite_in_progress + word
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
    cols = st.columns([1, 4], vertical_alignment="center")
    with cols[0]:
        prompt = get_prompt(include_generation_options=False)
    with cols[1]:
        st.write("**Prompt**:", prompt)
    doc = st.text_area(
        "Document",
        "Deep learning neural network technology advances are pretty cool if you are careful to use it in ways that don't take stuff from people.",
        height=150
    )
    spans = get_highlights(prompt, doc, doc)

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
    
    output_container = st.container(border=True)
    
    with st.expander("Controls"):
        num_to_show = st.slider("Number of edits to show", 1, num_different, value=num_different // 2)
        show_alternatives = st.checkbox("Show alternatives", value=True)
        if show_alternatives:
            show_all_on_hover = st.checkbox("Show all alternatives on hover", value=False)
        else:
            show_all_on_hover = False
    min_loss = loss_ratios_for_different[num_to_show - 1]

    with output_container:
        highlights_component(spans, show_alternatives, min_loss, show_all_on_hover=show_all_on_hover)

    if st.checkbox("Show details"):
        import pandas as pd
        st.write(pd.DataFrame(spans)[['token', 'token_loss', 'most_likely_token', 'loss_ratio']])
        st.write("Token loss is the difference between the original token and the most likely token. The loss ratio is the token loss divided by the highest token loss in the document.")

def highlights_component(spans, show_alternatives, min_loss, show_all_on_hover=False):
    import streamlit.components.v1 as components
    import html

    html_out = ''
    for span in spans:
        show = span['token'] != span['most_likely_token'] and span['loss_ratio'] >= min_loss
        alternative_to_show = next(token for token in span['topk_tokens'] if token != span['token'])
        show_alternative = show and show_alternatives
        hover = f'<span class="alternative">{show_token(alternative_to_show, escape_markdown=False)}</span>'
        html_out += '<span class="{cls}">{hover}{orig_token}</span>'.format(
            cls="highlight" if show else "regular",
            orig_token=html.escape(span["token"]).replace('\n', '<br>'),
            hover=hover if show_all_on_hover or show_alternative else ''
        )
    html_out = f"""
    <style>
        p.highlights-container {{
            background: white;
            line-height: 2.5;
            color: #2C3E50;  /* Dark blue-grey for main text */
        }}
        p.highlights-container > span {{
            position: relative;
            padding: 2px 1px;
            border-radius: 3px;
        }}
        p.highlights-container > span.highlight {{
            background-color: #E8F5E9;  /* Very light green */
            border-bottom: 2px solid #81C784;  /* Medium green */
        }}
        p.highlights-container > span.regular {{
            color: #546E7A;  /* Muted blue-grey */
        }}
        p.highlights-container .alternative {{
            display: none;
        }}
        p.highlights-container > span:hover .alternative {{
            display: inline;
            position: absolute; 
            top: -24px; 
            left: 50%;
            transform: translateX(-50%);
            min-width: 6em; 
            text-align: center;
            line-height: 1.2; 
            color: #1976D2;  /* Clear blue */
            background: white;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
    <p class="highlights-container">{html_out}</p>
    """
    return st.html(html_out)
    

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

        response = requests.post(
            f"{API_SERVER}/continue_messages",
            json={
                "messages": messages,
                "n_branch_tokens": 5,
                "n_future_tokens": 2
            }
        )
        if response.status_code != 200:
            st.error("Error fetching response")
            st.write(response.text)
            st.stop()
        response.raise_for_status()
        response = response.json()

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
        
def show_internals():
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
        
        response = requests.post(
            f"{API_SERVER}/logprobs",
            json={
                "messages": messages,
                "n_branch_tokens": 5,
                "n_future_tokens": 2
            }
        )
        if response.status_code != 200:
            st.error("Error fetching response")
            st.write(response.text)
            st.stop()
        response.raise_for_status()
        response = response.json()

        logprobs = response['logprobs']
        # logprobs is a list of tokens:
        # {
        #     "token": "the",
        #     "logprobs": [{"the": -0.1, "a": -0.2, ...}]
        # }
        #st.write(logprobs)
        logprobs_component(logprobs)
        
        def send_message():
            other_role = "assistant" if last_role == "user" else "user"
            st.session_state['messages'].append({"role": other_role, "content": ""})
            st.session_state['msg_in_progress'] = ""
        st.button("Send", on_click=send_message)
        
def logprobs_component(logprobs):
    import html, json
    html_out = ''
    for i, entry in enumerate(logprobs):
        token = entry['token']
        if token is not None:
            token_to_show = html.escape(show_token(token, escape_markdown=False))
        else:
            token_to_show = html.escape("<empty>")
        html_out += f'<span style="border: 1px solid black;" onclick="showLogprobs({i})" title="Click to show logprobs for this token">{token_to_show}</span>'
    show_logprob_js = '''
    function showLogprobs(i) {
        const logprobs = allLogprobs[i].logprobs;
        const logprobsHtml = Object.entries(logprobs).map(([token, logprob]) => `<li>${token}: ${Math.exp(logprob)}</li>`).join('');
        const container = document.getElementById('logprobs-display');
        container.innerHTML = `<ul>${logprobsHtml}</ul>`;
    }
'''
    html_out = f"""
    <script>allLogprobs = {json.dumps(logprobs)};
    
    {show_logprob_js}</script>
    <style>
        p.logprobs-container {{
            background: white;
            line-height: 2.5;
            color: #2C3E50;  /* Dark blue-grey for main text */
        }}
        p.logprobs-container > span {{
            position: relative;
            padding: 2px 1px;
            border-radius: 3px;
        }}
    </style>
    <p class="logprobs-container">{html_out}</p>
    <div id="logprobs-display"></div>
    """
    #return st.html(html_out)
    import streamlit.components.v1 as components
    return components.html(html_out, height=200, scrolling=True)

rewrite_page = st.Page(rewrite_with_predictions, title="Rewrite with predictions", icon="📝")
highlight_page = st.Page(highlight_edits, title="Highlight locations for possible edits", icon="🖍️")
generate_page = st.Page(generate_revisions, title="Generate revisions", icon="🔄")
type_assistant_response_page = st.Page(type_assistant_response, title="Type Assistant Response", icon="🔤")
show_internals_page = st.Page(show_internals, title="Show Internals", icon="🔧")

# Manually specify the sidebar
page = st.navigation([
    st.Page(landing, title="Home", icon="🏠"),
    highlight_page,
    rewrite_page,
    generate_page,
    type_assistant_response_page,
    show_internals_page
])
page.run()
