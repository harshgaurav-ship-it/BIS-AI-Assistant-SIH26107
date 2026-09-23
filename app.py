"""Streamlit entry point for the BIS Intelligent Assistant prototype."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import load_embedding_model  # noqa: E402
from src.rag_pipeline import answer_question  # noqa: E402
from src.retriever import FaissRetriever  # noqa: E402

INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIRECTORY / "bis_seed.faiss"
CHUNKS_PATH = INDEX_DIRECTORY / "bis_seed_chunks.json"
INDEX_READY = INDEX_PATH.is_file() and CHUNKS_PATH.is_file()

st.set_page_config(
    page_title="BIS Intelligent Assistant",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the local search index...")
def get_retriever() -> FaissRetriever:
    return FaissRetriever.load(INDEX_PATH, CHUNKS_PATH)


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model():
    return load_embedding_model()


def render_sources(citations) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(f"[{citation.number}] [{citation.title}]({citation.url})")


st.title("BIS Intelligent Assistant")
st.caption("Smart India Hackathon 2026 \u00B7 SIH26107")

if not INDEX_READY:
    st.error(
        "The local search index has not been built yet. "
        "Run `python scripts/build_index.py` in the project folder, then restart this app."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Prototype status")
    st.write("**Current milestone:** Professional UI")
    st.write("**Knowledge base:** Local FAISS index ready")
    st.divider()

    st.subheader("Product / certification lookup")
    st.caption(
        "Optional \u2014 add a product name or Indian Standard number "
        "to focus the search."
    )
    product_hint = st.text_input(
        "Product name or IS number",
        key="product_hint",
        placeholder="e.g. IS 302 electric iron",
    )
    top_k = st.slider("Evidence chunks to consider", min_value=1, max_value=6, value=3)
    minimum_score = st.slider(
        "Minimum similarity score", min_value=0.0, max_value=1.0, value=0.55, step=0.05
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Information is presented as guidance, not as a legal or certification decision.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("citations", []))

question = st.chat_input("Ask about BIS standards, certification, or hallmarking...")

if question:
    full_question = question
    if product_hint:
        full_question = f"{question} (related product/standard: {product_hint})"

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking verified BIS sources..."):
            try:
                model = get_embedding_model()
                retriever = get_retriever()
                grounded_answer = answer_question(
                    full_question,
                    retriever,
                    model,
                    top_k=top_k,
                    minimum_score=minimum_score,
                )
            except Exception as error:  # noqa: BLE001 - surface pipeline failures in the demo UI
                st.error(f"Something went wrong while checking BIS sources: {error}")
                st.stop()

        if grounded_answer.supported:
            st.markdown(grounded_answer.answer)
        else:
            st.warning(grounded_answer.answer)

        st.caption(grounded_answer.advisory_notice)
        render_sources(grounded_answer.citations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": grounded_answer.answer,
            "citations": grounded_answer.citations,
        }
    )
