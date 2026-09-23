# BIS Intelligent Assistant

**Smart India Hackathon 2026 · Problem Statement SIH26107**
AI-powered Intelligent Assistant for Indian Standards & BIS Services
Team: **Coder Crew**

An evidence-first assistant that answers questions about BIS standards, product
certification, and hallmarking using only verified, retrieved source material —
never free-form model guessing. If the evidence doesn't clearly support an
answer, the assistant says so and points the user to the official BIS website
instead of hallucinating.

## What it does

- Answers questions using a locally built knowledge base of reviewed BIS
  source documents (FAQs, certification overviews, laboratory information)
- Every answer is backed by retrieved evidence and shows numbered, clickable
  source citations
- Refuses to answer when no evidence clears the similarity threshold, instead
  of guessing
- Optional product name / Indian Standard number field to sharpen retrieval
- Runs entirely locally — no data leaves the machine during retrieval or
  answer generation

## How it works

1. **Ingestion** (`scripts/build_index.py`) — reviewed BIS source documents are
   chunked, embedded with Sentence Transformers, and indexed in a local FAISS
   vector store (`src/chunker.py`, `src/embeddings.py`, `src/retriever.py`)
2. **Retrieval** — a user question is embedded and matched against the FAISS
   index to find the most semantically similar evidence chunks
   (`src/retriever.py`)
3. **Grounded answering** — only evidence that clears a minimum similarity
   score (default `0.55`) is used; if nothing clears the bar, the assistant
   returns a safe "insufficient evidence" message instead of an answer
   (`src/rag_pipeline.py`)
4. **Citations** — every supported answer carries numbered links back to its
   source documents (`src/citations.py`)
5. **Interface** — a Streamlit chat app with conversation history, a source
   panel, and a product/IS-number lookup field (`app.py`)

An optional local-LLM layer (`src/chatbot.py`) can rephrase grounded answers
more naturally via Ollama, while still only using the retrieved evidence as
its source of truth.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

python scripts/build_index.py       # builds the local FAISS index (first run only)
streamlit run app.py                # launches the chat interface
```

To ask a question from the terminal instead of the UI:

```bash
python scripts/ask_assistant.py "What does BIS say about hallmarking?"
```

## Milestone status

| # | Milestone | Status |
|---|---|---|
| 1 | Foundation — repo, venv, starter app, licensing | Done |
| 2 | Verified source set — reviewed BIS corpus | Done |
| 3 | Document ingestion — chunking, FAISS index | Done |
| 4 | Retrieval and citations | Done |
| 5 | Grounded assistant — evidence-only answers | Done |
| 6 | Professional UI — chat, source panel, product lookup | Done |
| 7 | Hindi and evaluation | Tested — see limitation below |
| 8 | Release readiness | In progress |

## Known limitation: Hindi / Hinglish queries

The current embedding model is English-only. Hindi and Hinglish questions
were tested and do not reliably clear the safety similarity threshold, so the
assistant correctly refuses to answer rather than guess from a weak match.

Lowering the threshold to force a Hindi answer was tested and rejected — it
also weakens protection against wrong or unrelated evidence for English
questions, undermining the project's core no-hallucination guarantee.

**Future work:** swap to a multilingual embedding model and rebuild the FAISS
index to support Hindi natively without compromising answer safety.

## Disclaimer

All responses are general guidance drawn from cited BIS sources. They are not
a certification or legal decision — users should verify current requirements
on the official BIS website before relying on this information.

See `LICENSES.md` for license information.
