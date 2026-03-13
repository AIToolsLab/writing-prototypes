import streamlit as st
import requests
import json

def show_token(token: str, escape_markdown=True) -> str:
    token_display = token.replace('\n', '↵').replace('\t', '⇥')
    if escape_markdown:
        for c in "\\`*_{}[]()#+-.!":
            token_display = token_display.replace(c, "\\" + c)
    return token_display


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
        
        def send_message():
            other_role = "assistant" if last_role == "user" else "user"
            st.session_state['messages'].append({"role": other_role, "content": ""})
            st.session_state['msg_in_progress'] = ""
        st.button("Send", on_click=send_message)

        # Make request to vLLM.
        st.write(messages)
        result = requests.post(
            "https://vllm.thoughtful-ai.com/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "Qwen/Qwen3.5-9B",
                "messages": messages,
                "max_tokens": 2,
                "logprobs": True,
                "continue_final_message": True,
                "add_generation_prompt": False,
                "top_logprobs": 5,
                "prompt_logprobs": 5,
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "echo": True
            }
        )
        result = result.json()
        prompt_logprobs = result['prompt_logprobs']
        logprobs = []
        for tok in prompt_logprobs[1:]: # first token has no logprobs
            # looks like
            tok_logprobs = {v['decoded_token']: v['logprob'] for v in tok.values()}
            logprobs.append({
                "token": next(iter(tok_logprobs.keys())),
                "logprobs": tok_logprobs
            })

        # Add the last token (set "token" to None)
        last_token_logprobs = result['choices'][0]['logprobs']['content'][0]['top_logprobs']
        logprobs.append(
            {
                "token": None,
                "logprobs": {tok["token"]: tok["logprob"] for tok in last_token_logprobs}
            }
        )


        #st.write(last_token_logprobs)
        st.write("Conversation so far as tokens (click to show logprobs):")
        logprobs_component(logprobs)
        
        
def logprobs_component(logprobs):
    # logprobs is a list of tokens:
    # {
    #     "token": "the",
    #     "logprobs": [{"the": -0.1, "a": -0.2, ...}]
    # }
    import html, json
    html_out = ''
    for i, entry in enumerate(logprobs):
        token = entry['token']
        if token is not None:
            token_to_show = html.escape(show_token(token, escape_markdown=False))
        else:
            token_to_show = html.escape("[____]")
        html_out += f'<span style="border: 1px solid black; display: inline-block;" onclick="showLogprobs({i})" title="Click to show logprobs for this token">{token_to_show}</span>'
    show_logprob_js = '''
const makeElt = (tag, attrs, children) => {
    const elt = document.createElement(tag);
    for (const [attr, val] of Object.entries(attrs)) {
        elt.setAttribute(attr, val);
    }
    for (const child of children) {
        if(typeof child === 'string') {
            elt.appendChild(document.createTextNode(child));
        } else {
            elt.appendChild(child);
        }
    }
    return elt;
}

function escapeToken(token) {
    return token.replace(/\\n/g, '↵').replace(/\\t/g, '⇥');
}
    function showLogprobs(i) {
        const logprobs = allLogprobs[i].logprobs;
        const container = document.getElementById('logprobs-display');
        container.innerHTML = '';
        container.appendChild(makeElt('ul', {}, Object.entries(logprobs).map(([token, logprob]) => makeElt('li', {}, `${escapeToken(token)}: ${Math.exp(logprob)}`))));
    }
'''
    html_out = f"""

    <style>
        p.logprobs-container {{
            background: white;
            line-height: 1.5;
        }}
        p.logprobs-container > span {{
            position: relative;
            padding: 2px 1px;
            border-radius: 3px;
        }}
    </style>
    <p class="logprobs-container">{html_out}</p>
    <div id="logprobs-display"></div>
    <script>allLogprobs = {json.dumps(logprobs)};
    
    {show_logprob_js}

//showLogprobs(allLogprobs.length - 1);
</script>
"""
    import streamlit.components.v1 as components
    return components.html(html_out, height=300, scrolling=True)

show_internals()
