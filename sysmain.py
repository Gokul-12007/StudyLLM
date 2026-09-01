import os
import sys
import ctypes
import atexit
import streamlit as st
import pymupdf4llm

import db

# ---------------------------------------------------------------------------
# Keep Windows awake while this app is running (Ollama can't serve requests
# if the whole machine is asleep). Releases automatically on exit.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002  # remove this flag if you don't want to keep the screen on too

    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )

    @atexit.register
    def _restore_sleep():
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STUDY_DIR = "study_materials"
CHROMA_DIR = "chroma_store"
DEFAULT_CTX = 8192  # Ollama's built-in default (2048) silently truncates RAG context

st.set_page_config(page_title="Study Notebook", layout="wide")
os.makedirs(STUDY_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
db.init_db()

# ---------------------------------------------------------------------------
# Dark, sidebar-driven look (Streamlit can't fully match a native app, but
# this gets the structure: project tree -> chat list -> chat pane)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { border-right: 1px solid #23262E; }
    div.stButton > button {
        text-align: left;
        justify-content: flex-start;
        background: transparent;
        border: 1px solid transparent;
        padding: 0.35rem 0.6rem;
    }
    div.stButton > button:hover {
        background: #1E212B;
        border: 1px solid #2C3040;
    }
    .project-active > button {
        background: #23263080 !important;
        border-left: 3px solid #7C8CF8 !important;
    }
    .chat-active > button {
        background: #1E212B !important;
        color: #7C8CF8 !important;
    }
    .sidebar-label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6B7080;
        margin: 0.9rem 0 0.2rem 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Detect installed Ollama models, split by capability
# ---------------------------------------------------------------------------
def get_ollama_models():
    import urllib.request
    import json

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
    except Exception:
        return None

    chat_models, embed_models = [], []
    for m in tags.get("models", []):
        caps = m.get("capabilities", [])
        if "embedding" in caps:
            embed_models.append(m["name"])
        elif "completion" in caps:
            chat_models.append(m["name"])
    return {"chat": sorted(chat_models), "embed": sorted(embed_models)}


def _default_index(options, preferred):
    for p in preferred:
        for i, o in enumerate(options):
            if o.startswith(p):
                return i
    return 0


def sanitize_name(name):
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "doc"
    if len(cleaned) < 3:
        cleaned = cleaned + ("0" * (3 - len(cleaned)))
    return cleaned[:512]


models = get_ollama_models()
if models is None:
    st.error("⚠️ Can't reach Ollama at localhost:11434. Is it running?")
    st.stop()
if not models["chat"]:
    st.error("⚠️ No chat-capable models found. Run: ollama pull qwen3:8b")
    st.stop()
if not models["embed"]:
    st.error("⚠️ No embedding models found. Run: ollama pull nomic-embed-text")
    st.stop()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("current_project_id", None)
st.session_state.setdefault("current_chat_id", None)

# ---------------------------------------------------------------------------
# Sidebar: Projects
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 📚 Study Notebook")

with st.sidebar.form("new_project_form", clear_on_submit=True):
    new_project_name = st.text_input("New project", placeholder="e.g. Thermodynamics")
    if st.form_submit_button("+ New Project") and new_project_name.strip():
        pid = db.create_project(new_project_name.strip())
        cid = db.create_chat(pid, "New chat")
        st.session_state.current_project_id = pid
        st.session_state.current_chat_id = cid
        st.rerun()

projects = db.list_projects()

st.sidebar.markdown('<div class="sidebar-label">Projects</div>', unsafe_allow_html=True)
for p in projects:
    is_active = p["id"] == st.session_state.current_project_id
    wrapper_class = "project-active" if is_active else ""
    st.sidebar.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
    if st.sidebar.button(f"📁 {p['name']}", key=f"proj_{p['id']}", use_container_width=True):
        st.session_state.current_project_id = p["id"]
        chats = db.list_chats(p["id"])
        st.session_state.current_chat_id = chats[0]["id"] if chats else None
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

project_id = st.session_state.current_project_id

# ---------------------------------------------------------------------------
# Sidebar: Chats (only once a project is selected)
# ---------------------------------------------------------------------------
if project_id:
    st.sidebar.markdown('<div class="sidebar-label">Chats</div>', unsafe_allow_html=True)
    if st.sidebar.button("+ New chat", key="new_chat_btn", use_container_width=True):
        cid = db.create_chat(project_id, "New chat")
        st.session_state.current_chat_id = cid
        st.rerun()

    for c in db.list_chats(project_id):
        is_active = c["id"] == st.session_state.current_chat_id
        wrapper_class = "chat-active" if is_active else ""
        st.sidebar.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.sidebar.button(f"💬 {c['title']}", key=f"chat_{c['id']}", use_container_width=True):
            st.session_state.current_chat_id = c["id"]
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: Settings (models + document upload), collapsed by default
# ---------------------------------------------------------------------------
with st.sidebar.expander("⚙️ Settings", expanded=False):
    CHAT_MODEL = st.selectbox(
        "Chat model", models["chat"], index=_default_index(models["chat"], ["qwen3:8b", "qwen3", "llama3.1"])
    )
    EMBED_MODEL = st.selectbox(
        "Embedding model", models["embed"], index=_default_index(models["embed"], ["nomic-embed-text"])
    )

    if project_id:
        st.markdown("**Documents in this project**")
        docs_meta = db.list_documents(project_id)
        for d in docs_meta:
            st.caption(f"📄 {d['filename']}")

        uploaded_file = st.file_uploader("Add a PDF", type=["pdf"], key=f"uploader_{project_id}")
        if uploaded_file:
            proj_dir = os.path.join(STUDY_DIR, project_id)
            os.makedirs(proj_dir, exist_ok=True)
            base_name, _ = os.path.splitext(uploaded_file.name)
            file_path = os.path.join(proj_dir, uploaded_file.name)
            md_path = os.path.join(proj_dir, base_name + ".md")

            already_added = any(d["filename"] == uploaded_file.name for d in docs_meta)
            if not already_added:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Parsing PDF..."):
                    md_text = pymupdf4llm.to_markdown(file_path)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_text)
                db.add_document(project_id, uploaded_file.name, file_path, md_path)
                st.success(f"Added {uploaded_file.name}")
                st.rerun()


# ---------------------------------------------------------------------------
# Retrieval engine, scoped to a project: combines every document currently
# attached to that project into one hybrid (BM25 + vector) retriever.
# ---------------------------------------------------------------------------
def project_signature(pid):
    parts = []
    for d in db.list_documents(pid):
        try:
            mtime = os.path.getmtime(d["md_path"])
        except OSError:
            mtime = 0
        parts.append(f"{d['id']}:{mtime}")
    return "|".join(sorted(parts))


@st.cache_resource(show_spinner="Indexing project documents...")
def setup_project_engine(pid, embed_model, _signature):
    docs_meta = db.list_documents(pid)
    if not docs_meta:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    for d in docs_meta:
        with open(d["md_path"], "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.create_documents([text], metadatas=[{"source": d["filename"]}])
        for i, ch in enumerate(chunks):
            ch.metadata["chunk_id"] = f"{d['id']}_{i}"
        all_chunks.extend(chunks)

    embeddings = OllamaEmbeddings(model=embed_model)
    collection = sanitize_name(f"{pid}_{embed_model}")
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Only embed chunks belonging to documents not already in this collection
    # (checked per-document via a probe id) so adding a new PDF to a project
    # doesn't re-embed everything that came before it.
    for d in docs_meta:
        probe_id = f"{d['id']}_0"
        existing = vector_store.get(ids=[probe_id])
        if existing["ids"]:
            continue
        doc_chunks = [c for c in all_chunks if c.metadata["chunk_id"].startswith(f"{d['id']}_")]
        if doc_chunks:
            ids = [c.metadata["chunk_id"] for c in doc_chunks]
            vector_store.add_documents(doc_chunks, ids=ids)

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 3

    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])


# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------
if not project_id:
    st.info("👈 Create or select a project in the sidebar to get started.")
    st.stop()

chat_id = st.session_state.current_chat_id
if not chat_id:
    st.info("👈 Start a new chat in the sidebar.")
    st.stop()

retriever = setup_project_engine(project_id, EMBED_MODEL, project_signature(project_id))

if retriever is None:
    st.info("📄 Add a PDF to this project (sidebar → Settings) before asking questions.")
    st.stop()

llm = ChatOllama(model=CHAT_MODEL, num_ctx=DEFAULT_CTX)
prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer isn't in the context, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for msg in db.list_messages(chat_id):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about your study material...")
if user_query:
    db.add_message(chat_id, "user", user_query)

    # First message in a chat becomes its title, like Codex threads.
    existing_messages = db.list_messages(chat_id)
    if len(existing_messages) == 1:
        db.rename_chat(chat_id, user_query[:60])

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)
    db.add_message(chat_id, "assistant", response)
    st.rerun()