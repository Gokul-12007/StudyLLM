# 📚 Study Notebook

A local, offline study assistant. Upload PDFs, get hybrid (keyword + semantic)
search over them, and chat with a local Ollama model about the material —
no cloud, no API keys, nothing leaves your machine.

Inspired by NotebookLM, built to run entirely offline on your own hardware.

## Features

- **Fully offline** — PDF parsing, embeddings, and chat all run locally via [Ollama](https://ollama.com)
- **Hybrid retrieval** — combines BM25 keyword search with vector similarity search for better recall than either alone
- **Projects & chats** — organize study material into projects, each with its own persistent chat history (SQLite, stored on disk — not in memory, survives restarts)
- **Model picker** — pick any chat/embedding model you already have pulled in Ollama, right from the sidebar
- **Incremental indexing** — adding a new PDF to a project only embeds that document, not the whole project again

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- At least one chat-capable model and one embedding model pulled:
  ```
  ollama pull qwen3:8b
  ollama pull nomic-embed-text
  ```

## Setup

```bash
git clone <this-repo>
cd StudyLLM
pip install -r requirements.txt
streamlit run sysmain.py
```

The app opens at `http://localhost:8501`. On first launch it checks that
Ollama is reachable and that you have at least one chat and one embedding
model installed — it'll tell you exactly what to pull if not.

## Usage

1. Create a **Project** in the sidebar (e.g. one per course or subject)
2. Open **Settings** in the sidebar and upload a PDF to that project
3. Start a **New Chat** and ask questions — answers are grounded in the
   PDF's content via retrieval, not the model's general knowledge
4. Chat history and projects persist across restarts automatically

## Project structure

```
sysmain.py            # Streamlit app (UI, retrieval, chat)
db.py                 # SQLite persistence: projects / chats / messages
.streamlit/config.toml  # dark theme
requirements.txt
```

Generated at runtime (not committed — see `.gitignore`):

```
study_materials/   # uploaded PDFs + parsed markdown
chroma_store/       # persisted vector index
notebook.db          # local chat history
```

## How it works

```
PDF → pymupdf4llm → Markdown → chunked
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
              BM25 (keyword)              Chroma + Ollama embeddings
                    │                              │
                    └──────────────┬──────────────┘
                          EnsembleRetriever (hybrid)
                                   │
                        retrieved context + question
                                   │
                          ChatOllama (local LLM)
                                   │
                                answer
```

## Notes

- Larger PDFs (100+ pages) take a while to embed on CPU-only setups — this only happens once per document, then it's cached.
- Windows: the app calls `SetThreadExecutionState` to keep the machine awake while running, so Ollama doesn't get suspended mid-session on sleep.
