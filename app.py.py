"""Streamlit UI for the Mini-RAG PDF Q&A Agent."""

import os
import streamlit as st
from dotenv import load_dotenv

from rag import build_index, answer_question

load_dotenv()

st.set_page_config(page_title="PDF Q&A Agent", page_icon="📄", layout="wide")

st.title("📄 PDF Q&A Agent")
st.caption("Upload a PDF, ask questions, get answers with page numbers.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Setup")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing `OPENAI_API_KEY`. Add it to your `.env` file or secrets.")
    else:
        st.success("OpenAI key detected ✅")

    st.divider()
    st.header("📂 Upload PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. PDF is split into chunks\n"
        "2. Chunks are embedded\n"
        "3. Your question retrieves top matches\n"
        "4. GPT-4o-mini answers with citations"
    )

# ---------- Session state ----------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ---------- Index PDF ----------
if uploaded and not st.session_state.indexed:
    with st.spinner("Indexing PDF..."):
        st.session_state.collection = build_index([uploaded])
        st.session_state.indexed = True
    st.success(f"Indexed {st.session_state.collection.count()} chunks.")

# ---------- Chat ----------
st.divider()
st.subheader("💬 Ask your document")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**[Source {i}]** — page {s['page']}")
                    st.caption(s["text"][:300] + "...")

# Input
question = st.chat_input("Ask a question about your PDF...")

if question:
    if not st.session_state.collection:
        st.warning("Please upload a PDF first.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(st.session_state.collection, question)
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📚 Sources ({len(result['sources'])})"):
                for i, s in enumerate(result["sources"], 1):
                    st.markdown(f"**[Source {i}]** — page {s['page']}")
                    st.caption(s["text"][:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })