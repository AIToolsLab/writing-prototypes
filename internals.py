# References for vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py

import streamlit as st
import requests
import json

placeholders_to_try = '#.?!@$%^&*()_+-=~`|;:"<>,./\\'

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
        st.session_state['placeholder_token'] = placeholders_to_try[0]
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

        # Unfortunately chat templates include things like this:
        #     {%- set content = render_content(message.content, true)|trim %}
        # so we can't include leading or trailing whitespace.
        # Can't do much about leading whitespace, but we can at least allow trailing whitespace by including a special token for it.
        # Unfortunately there's no single token that never gets joined with any other one, so we have to try a few different ones and see which one actually gets separated out by the tokenizer.


        messages[-1]['content'] = msg_in_progress + st.session_state.placeholder_token

        st.write(messages)

        def send_message():
            other_role = "assistant" if last_role == "user" else "user"
            st.session_state['messages'].append({"role": other_role, "content": ""})
            st.session_state['msg_in_progress'] = ""
        st.button("Send", on_click=send_message)

        token_ids_req = requests.post(
            "https://vllm.thoughtful-ai.com/tokenize",
            headers={"Content-Type": "application/json"},
            json={
                "model": "Qwen/Qwen3.5-9B",
                "messages": messages,
                "continue_final_message": True,
                "add_generation_prompt": False,
                "return_token_strs": True,
            }
        )
        token_ids_req = token_ids_req.json()
        token_ids = token_ids_req['tokens']
        token_strs = token_ids_req['token_strs']

        # completion given prompt token ids
        logprobs_request = requests.post(
            "https://vllm.thoughtful-ai.com/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "Qwen/Qwen3.5-9B",
                "prompt": token_ids,
                "max_tokens": 2,
                "logprobs": 5,
                "echo": True,
            }
        )
        logprobs_request = logprobs_request.json()
        complete_text = logprobs_request['choices'][0]['text']
        logprobs_part = logprobs_request['choices'][0]['logprobs']
        logprobs = []
        for i in range(len(token_ids)):
            if i == 0:
                # first token has no logprobs, but show the token string.
                logprobs.append({
                    "token": logprobs_part['tokens'][0],
                    "logprobs": None
                })
                continue

            top_logprobs = logprobs_part['top_logprobs'][i]
            logprobs.append({
                "token": logprobs_part['tokens'][i],
                "logprobs": {tok: logprob for tok, logprob in top_logprobs.items()}
            })
        #st.write(logprobs_part)

        if logprobs and logprobs[-1]['token'] == st.session_state.placeholder_token:
            logprobs[-1]['token'] = None
            # remove the placeholder token logprobs, since they aren't meaningful
            logprobs[-1]['logprobs'] = {tok: logprob for tok, logprob in logprobs[-1]['logprobs'].items() if tok != st.session_state.placeholder_token}
        else:
            st.warning("Expected the last token to be the placeholder token, but it wasn't. Logprobs may not display correctly.")
            if st.button("Try a different placeholder token"):
                current_index = placeholders_to_try.index(st.session_state.placeholder_token)
                next_index = (current_index + 1) % len(placeholders_to_try)
                st.session_state.placeholder_token = placeholders_to_try[next_index]
                st.rerun()

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

showLogprobs(allLogprobs.length - 1);
</script>
"""
    import streamlit.components.v1 as components
    return components.html(html_out, height=300, scrolling=True)

show_internals()
