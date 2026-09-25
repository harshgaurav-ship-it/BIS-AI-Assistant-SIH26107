    """Streamlit entry point for the BIS Intelligent Assistant."""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import OllamaSettings, refine_with_ollama  # noqa: E402
from src.embeddings import load_embedding_model  # noqa: E402
from src.rag_pipeline import answer_question  # noqa: E402
from src.retriever import FaissRetriever  # noqa: E402

INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIRECTORY / "bis_seed.faiss"
CHUNKS_PATH = INDEX_DIRECTORY / "bis_seed_chunks.json"
INDEX_READY = INDEX_PATH.is_file() and CHUNKS_PATH.is_file()

FEEDBACK_LOG = PROJECT_ROOT / "data" / "feedback_log.csv"

EXAMPLE_QUESTIONS = [
    "What does BIS say about hallmarking?",
    "Which standard applies to electric irons?",
    "How do I get ISI certification for my product?",
]

st.set_page_config(
    page_title="BIS Sahayak — Intelligent Assistant",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theming — self-contained CSS, no extra dependencies required
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bis-navy: #0B2545;
        --bis-saffron: #FF7722;
        --bis-green: #128807;
    }
    .block-container { padding-top: 1.5rem; max-width: 900px; }
    .bis-header {
        display: flex; align-items: center; gap: 0.75rem;
        padding: 1rem 1.25rem; border-radius: 14px;
        background: linear-gradient(120deg, var(--bis-navy), #123a6b);
        color: white; margin-bottom: 1.25rem;
    }
    .bis-header h1 { font-size: 1.4rem; margin: 0; color: white; }
    .bis-header p { margin: 0; opacity: 0.85; font-size: 0.85rem; }
    .bis-badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; margin-right: 0.35rem;
    }
    .bis-badge-done { background: #e5f6e5; color: var(--bis-green); }
    .bis-badge-progress { background: #fff2e0; color: var(--bis-saffron); }
    .bis-citation-chip {
        display: inline-block; padding: 0.2rem 0.6rem; margin: 0.15rem 0.3rem 0.15rem 0;
        border-radius: 8px; background: #f0f3f8; border: 1px solid #d8e0ec;
        font-size: 0.8rem; text-decoration: none; color: var(--bis-navy);
    }
    .bis-confidence {
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; margin-bottom: 0.4rem;
    }
    .bis-confidence-high { background: #e5f6e5; color: var(--bis-green); }
    .bis-confidence-medium { background: #fff2e0; color: #a15c00; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bis-header">
        <div style="font-size:2rem;">🛡️</div>
        <div>
            <h1>BIS Sahayak</h1>
            <p>Smart India Hackathon 2026 · SIH26107 · Team Coder Crew —
            evidence-first answers on Indian Standards &amp; BIS services</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not INDEX_READY:
    st.error(
        "The local search index has not been built yet. "
        "Run `python scripts/build_index.py` in the project folder, "
        "then restart this app."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading the local search index...")
def get_retriever() -> FaissRetriever:
    return FaissRetriever.load(INDEX_PATH, CHUNKS_PATH)


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model():
    return load_embedding_model()


def log_feedback(question: str, supported: bool, rating: str) -> None:
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_LOG.is_file()
    with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["timestamp", "question", "supported", "rating"])
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), question, supported, rating]
        )


def render_confidence(minimum_score: float) -> None:
    high = minimum_score >= 0.55
    label = "High evidence match" if high else "Broader evidence match"
    css_class = "bis-confidence-high" if high else "bis-confidence-medium"
    st.markdown(
        f'<span class="bis-confidence {css_class}">{label} · threshold {minimum_score:.2f}</span>',
        unsafe_allow_html=True,
    )


def render_sources(citations) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    chips = "".join(
        f'<a class="bis-citation-chip" href="{c.url}" target="_blank">'
        f"[{c.number}] {c.title}</a>"
        for c in citations
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_message(index: int, message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and not message.get("supported", True):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])

        if message["role"] != "assistant":
            return

        if message.get("supported") and "minimum_score" in message:
            render_confidence(message["minimum_score"])
        if message.get("refined"):
            st.caption("Answer refined locally using retrieved evidence.")
        if message.get("advisory_notice"):
            st.caption(message["advisory_notice"])
        render_sources(message.get("citations", []))

        fcol1, fcol2, _ = st.columns([1, 1, 6])
        if fcol1.button("👍", key=f"fb_up_{index}"):
            log_feedback(message.get("question", ""), message.get("supported", False), "up")
            st.toast("Thanks for the feedback!")
        if fcol2.button("👎", key=f"fb_down_{index}"):
            log_feedback(message.get("question", ""), message.get("supported", False), "down")
            st.toast("Thanks — we'll use this to improve retrieval.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.header("Prototype status")
    st.markdown(
        """
        <span class="bis-badge bis-badge-done">Ingestion</span>
        <span class="bis-badge bis-badge-done">Retrieval</span>
        <span class="bis-badge bis-badge-done">Grounded answers</span>
        <span class="bis-badge bis-badge-progress">Hindi support</span>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("How it works"):
        st.markdown(
            "1. Your question is embedded and matched against a local FAISS "
            "index of reviewed BIS documents.\n"
            "2. Only evidence above the similarity threshold is used.\n"
            "3. The answer is generated strictly from that evidence, with "
            "numbered source citations.\n"
            "4. If nothing clears the bar, the assistant says so instead of "
            "guessing."
        )

    st.divider()
    st.subheader("Product / certification lookup")
    st.caption(
        "Optional — add a product name or Indian Standard number "
        "to focus the search."
    )
    product_hint = st.text_input(
        "Product name or IS number",
        key="product_hint",
        placeholder="e.g. IS 302 electric iron",
    )
    top_k = st.slider("Evidence chunks to consider", min_value=1, max_value=6, value=3)
    minimum_score = st.slider(
        "Minimum similarity score",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
    )

    st.divider()
    use_ollama = st.toggle(
        "Use local AI refinement",
        value=False,
        help=(
            "When enabled, a local Ollama model rewrites supported "
            "evidence into a shorter natural-language answer."
        ),
    )
    if use_ollama:
        st.caption("Only retrieved evidence is provided to the local LLM.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Information is presented as guidance, not as a legal "
        "or certification decision."
    )

# Quick-start example questions, shown only before the first message
if not st.session_state.messages:
    st.caption("Try asking:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example

for i, message in enumerate(st.session_state.messages):
    render_message(i, message)

question = st.chat_input("Ask about BIS standards, certification, or hallmarking...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    full_question = question
    if product_hint:
        full_question = f"{question} (related product/standard: {product_hint})"

    st.session_state.messages.append({"role": "user", "content": question})

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
            except Exception as error:  # noqa: BLE001
                st.error(f"Something went wrong while checking BIS sources: {error}")
                st.stop()

        final_answer = grounded_answer.answer
        refined = False
        if grounded_answer.supported and use_ollama:
            with st.spinner("Refining the evidence-based response..."):
                try:
                    final_answer = refine_with_ollama(
                        question=full_question,
                        grounded_answer=grounded_answer,
                        settings=OllamaSettings(),
                    )
                    refined = True
                except Exception:  # noqa: BLE001
                    final_answer = grounded_answer.answer
                    st.warning(
                        "Local AI refinement is unavailable. "
                        "Showing the evidence-only answer instead."
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "citations": grounded_answer.citations,
            "minimum_score": minimum_score,
            "question": full_question,
            "supported": grounded_answer.supported,
            "refined": refined,
            "advisory_notice": grounded_answer.advisory_notice,
        }
    )
    st.rerun()
