"""Streamlit entry point for the BIS Intelligent Assistant prototype."""

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).parent
INDEX_READY = (PROJECT_ROOT / "data" / "index" / "bis_seed.faiss").is_file()

st.set_page_config(
    page_title="BIS Intelligent Assistant",
    page_icon="🇮🇳",
    layout="wide",
)

st.title("BIS Intelligent Assistant")
st.caption("Smart India Hackathon 2026 · SIH26107")

st.success("Milestone 1 health check: the Streamlit application is running.")

if INDEX_READY:
    st.info(
        "Milestone 4: the local semantic-search index is ready. "
        "The conversational answer interface is the next milestone."
    )
else:
    st.info(
        "Milestone 4 code is ready, but the local semantic-search index has not "
        "been built yet. Run `python scripts/build_index.py` in the project folder."
    )

st.subheader("What this prototype will support")
st.markdown(
    """
    - Verified BIS information with source links
    - Product and certification guidance when supported by evidence
    - Laboratory and hallmarking information from approved sources
    - Safe responses when verified information is unavailable
    """
)

with st.sidebar:
    st.header("Prototype status")
    st.write("**Current milestone:** Semantic retrieval")
    if INDEX_READY:
        st.write("**Knowledge base:** Local FAISS index ready")
    else:
        st.write("**Knowledge base:** 16 reviewed facts; index not built yet")
    st.write("**Answer generation:** Not connected yet")
    st.divider()
    st.caption("Information will be presented as guidance, not as a legal or certification decision.")
