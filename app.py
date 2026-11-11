import streamlit as st
import anthropic
from pathlib import Path

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
MODEL = "claude-sonnet-5-5"

docs = "\n\n".join(
    f"## {p.name}\n{p.read_text(encoding='utf-8')}"
    for p in Path("docs").glob("*.txt")
)

SYSTEM = f"""You are the internal help assistant for Brightline Marketing.
Answer only from the documents below. If the answer is not in them,
say you don't know and suggest asking the team lead. Keep answers short and clear.

DOCUMENTS:
{docs}"""

st.title("Brightline Help Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about our processes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with client.messages.stream(
            model=MODEL, max_tokens=1000, system=SYSTEM,
            messages=st.session_state.messages,
        ) as stream:
            reply = st.write_stream(stream.text_stream)
    st.session_state.messages.append({"role": "assistant", "content": reply})