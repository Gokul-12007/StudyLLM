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

st.set_page_config(page_title="Study Notebook", page_icon="📚", layout="wide")
os.makedirs(STUDY_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
db.init_db()

# ---------------------------------------------------------------------------
# Theme toggle — .streamlit/config.toml sets the process-level default
# (light, premium-minimalist), but Streamlit can't re-read that file
# mid-session. So the actual light/dark switch is driven by CSS custom
# properties, toggled from the sidebar and stored in session state.
# ---------------------------------------------------------------------------
st.session_state.setdefault("theme_mode", "light")

THEMES = {
    "light": {
        "bg-app": "#FAFAFA",
        "bg-panel": "#FFFFFF",
        "bg-panel-raised": "#F4F4F5",
        "bg-hover": "#F0F0F1",
        "border": "#E4E4E7",
        "border-soft": "#D4D4D8",
        "accent": "#18181B",
        "accent-soft": "#18181B14",
        "accent-strong": "#000000",
        "text-primary": "#18181B",
        "text-secondary": "#52525B",
        "text-muted": "#A1A1AA",
        "success": "#16A34A",
        "danger": "#DC2626",
        "btn-text": "#FFFFFF",
    },
    "dark": {
        "bg-app": "#0F1117",
        "bg-panel": "#151822",
        "bg-panel-raised": "#1B1F2B",
        "bg-hover": "#1E212B",
        "border": "#23262E",
        "border-soft": "#2C3040",
        "accent": "#E7E9F0",
        "accent-soft": "#E7E9F014",
        "accent-strong": "#FFFFFF",
        "text-primary": "#E7E9F0",
        "text-secondary": "#9BA0B3",
        "text-muted": "#6B7080",
        "success": "#4ADE80",
        "danger": "#F87171",
        "btn-text": "#0F1117",
    },
}

# ---------------------------------------------------------------------------
# Sidebar: Brand + theme toggle
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="app-brand">
        <div class="icon">📚</div>
        <div>
            <div class="title">Study Notebook</div>
            <div class="subtitle">offline · local · private</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_theme_choice = st.sidebar.radio(
    "Appearance",
    options=["light", "dark"],
    index=0 if st.session_state.theme_mode == "light" else 1,
    format_func=lambda x: "☀️ Light" if x == "light" else "🌙 Dark",
    horizontal=True,
    key="theme_radio",
)
st.session_state.theme_mode = _theme_choice
T = THEMES[st.session_state.theme_mode]

# ---------------------------------------------------------------------------
# CSS — premium minimalist theme (Vercel/Linear-style): soft rounded inputs,
# deliberate high-contrast buttons, bordered cards, no default Streamlit red.
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --bg-app: {T['bg-app']};
            --bg-panel: {T['bg-panel']};
            --bg-panel-raised: {T['bg-panel-raised']};
            --bg-hover: {T['bg-hover']};
            --border: {T['border']};
            --border-soft: {T['border-soft']};
            --accent: {T['accent']};
            --accent-soft: {T['accent-soft']};
            --accent-strong: {T['accent-strong']};
            --text-primary: {T['text-primary']};
            --text-secondary: {T['text-secondary']};
            --text-muted: {T['text-muted']};
            --success: {T['success']};
            --danger: {T['danger']};
            --btn-text: {T['btn-text']};
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 14px;
        }}

        html, body, [class*="css"] {{
            font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        code, pre, .stCode {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        .stApp {{
            background: var(--bg-app);
        }}

        /* ---------- Sidebar shell ---------- */
        section[data-testid="stSidebar"] {{
            background: var(--bg-panel);
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] > div {{ padding-top: 0.5rem; }}

        .app-brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.6rem 0.4rem 1rem 0.4rem;
            margin-bottom: 0.2rem;
            border-bottom: 1px solid var(--border);
        }}
        .app-brand .icon {{
            width: 34px;
            height: 34px;
            border-radius: 9px;
            background: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            flex-shrink: 0;
        }}
        .app-brand .title {{
            font-weight: 800;
            font-size: 1.02rem;
            color: var(--text-primary);
            line-height: 1.1;
        }}
        .app-brand .subtitle {{
            font-size: 0.7rem;
            color: var(--text-muted);
            letter-spacing: 0.02em;
        }}

        .sidebar-label {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin: 1.1rem 0 0.35rem 0.3rem;
        }}

        /* ---------- Theme radio, rendered as a compact pill switch ---------- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            background: var(--bg-panel-raised);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.2rem;
            gap: 0.2rem;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            border-radius: var(--radius-sm);
            padding: 0.15rem 0.5rem;
        }}

        /* ---------- Sidebar nav buttons (projects / chats) ---------- */
        div.stButton > button {{
            text-align: left;
            justify-content: flex-start;
            background: transparent;
            border: 1px solid transparent;
            border-radius: var(--radius-md);
            padding: 0.42rem 0.65rem;
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.86rem;
            transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
        }}
        div.stButton > button:hover {{
            background: var(--bg-hover);
            border: 1px solid var(--border-soft);
            color: var(--text-primary);
        }}
        div.stButton > button:focus:not(:active) {{ box-shadow: none; }}

        .project-active > div.stButton > button,
        .project-active button {{
            background: var(--accent-soft) !important;
            border-left: 3px solid var(--accent) !important;
            color: var(--text-primary) !important;
            font-weight: 600;
        }}
        .chat-active > div.stButton > button,
        .chat-active button {{
            background: var(--bg-hover) !important;
            color: var(--text-primary) !important;
            font-weight: 700;
        }}

        div[data-testid="stForm"] div.stButton > button,
        button[kind="secondaryFormSubmit"] {{
            background: var(--bg-panel-raised);
            border: 1px dashed var(--border-soft);
            justify-content: center;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        div[data-testid="stForm"] div.stButton > button:hover {{
            border-color: var(--accent);
            color: var(--text-primary);
        }}

        /* ---------- Premium minimalist input/button polish ---------- */
        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div {{
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border-soft) !important;
            background: var(--bg-panel-raised) !important;
            color: var(--text-primary) !important;
        }}

        .stButton > button {{
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
        }}
        /* Solid, deliberate "primary action" styling for main-pane buttons
           (sidebar nav buttons keep their transparent ghost style above). */
        div[data-testid="stAppViewContainer"] .main .stButton > button {{
            background-color: var(--accent) !important;
            color: var(--btn-text) !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
        }}
        div[data-testid="stAppViewContainer"] .main .stButton > button:hover {{
            background-color: var(--accent-strong) !important;
        }}

        /* Download button — same premium treatment */
        .stDownloadButton > button {{
            border-radius: var(--radius-sm) !important;
            font-weight: 600 !important;
            background-color: var(--accent) !important;
            color: var(--btn-text) !important;
            border: none !important;
            padding: 0.45rem 1.3rem !important;
        }}
        .stDownloadButton > button:hover {{
            background-color: var(--accent-strong) !important;
        }}

        /* ---------- Document chip list ---------- */
        .doc-chip {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.5rem;
            border-radius: var(--radius-sm);
            background: var(--bg-panel-raised);
            border: 1px solid var(--border);
            margin-bottom: 0.35rem;
            font-size: 0.78rem;
            color: var(--text-secondary);
        }}
        .doc-chip .dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--success);
            flex-shrink: 0;
        }}

        /* ---------- File uploader ---------- */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: var(--bg-panel-raised);
            border: 1.5px dashed var(--border-soft);
            border-radius: var(--radius-md);
        }}
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: var(--accent);
        }}

        /* ---------- Main pane header ---------- */
        .chat-header {{
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            padding-bottom: 0.6rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid var(--border);
        }}
        .chat-header .crumb {{
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 500;
        }}
        .chat-header .crumb .sep {{ margin: 0 0.35rem; color: var(--border-soft); }}
        .chat-header h2 {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
        }}

        /* ---------- Empty state cards ---------- */
        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 0.6rem;
            padding: 4rem 2rem;
            margin-top: 2rem;
            border: 1px dashed var(--border-soft);
            border-radius: var(--radius-lg);
            background: var(--bg-panel);
        }}
        .empty-state .icon {{ font-size: 2.4rem; opacity: 0.85; }}
        .empty-state .headline {{ font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }}
        .empty-state .hint {{ font-size: 0.85rem; color: var(--text-muted); max-width: 320px; }}

        /* ---------- Chat bubbles (user turns) ---------- */
        [data-testid="stChatMessage"] {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
        }}
        [data-testid="stChatMessageContent"] p {{
            color: var(--text-primary);
            line-height: 1.55;
        }}

        /* Answer card label — sits above the bordered st.container(border=True)
           that wraps each assistant response. */
        .answer-label {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--border-soft) !important;
            border-radius: var(--radius-lg) !important;
            background: var(--bg-panel-raised);
        }}

        [data-testid="stChatInput"] textarea {{
            background: var(--bg-panel-raised) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: var(--radius-lg) !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stChatInput"] textarea:focus {{ border-color: var(--accent) !important; }}

        [data-testid="stSelectbox"] > div > div {{
            background: var(--bg-panel-raised);
            border: 1px solid var(--border-soft);
            border-radius: var(--radius-md);
        }}

        section[data-testid="stSidebar"] details {{
            background: var(--bg-panel-raised);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            margin-top: 1rem;
        }}
        section[data-testid="stSidebar"] summary {{
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-primary);
        }}

        div[data-testid="stAlert"] {{ border-radius: var(--radius-md); }}
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
with st.sidebar.form("new_project_form", clear_on_submit=True):
    new_project_name = st.text_input("New project", placeholder="e.g. Thermodynamics", label_visibility="collapsed")
    if st.form_submit_button("＋ New Project", use_container_width=True) and new_project_name.strip():
        pid = db.create_project(new_project_name.strip())
        cid = db.create_chat(pid, "New chat")
        st.session_state.current_project_id = pid
        st.session_state.current_chat_id = cid
        st.rerun()

projects = db.list_projects()
st.sidebar.markdown('<div class="sidebar-label">Projects</div>', unsafe_allow_html=True)

if not projects:
    st.sidebar.markdown(
        '<div style="font-size:0.78rem; color:var(--text-muted); padding:0.2rem 0.4rem;">'
        'No projects yet — create one above.</div>',
        unsafe_allow_html=True,
    )

for p in projects:
    is_active = p["id"] == st.session_state.current_project_id
    wrapper_class = "project-active" if is_active else ""
    st.sidebar.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
    if st.sidebar.button(f"📁  {p['name']}", key=f"proj_{p['id']}", use_container_width=True):
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
    if st.sidebar.button("＋ New chat", key="new_chat_btn", use_container_width=True):
        cid = db.create_chat(project_id, "New chat")
        st.session_state.current_chat_id = cid
        st.rerun()

    for c in db.list_chats(project_id):
        is_active = c["id"] == st.session_state.current_chat_id
        wrapper_class = "chat-active" if is_active else ""
        st.sidebar.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.sidebar.button(f"💬  {c['title']}", key=f"chat_{c['id']}", use_container_width=True):
            st.session_state.current_chat_id = c["id"]
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: Settings (models + document upload), collapsed by default
# ---------------------------------------------------------------------------
with st.sidebar.expander("⚙️  Settings", expanded=False):
    CHAT_MODEL = st.selectbox(
        "Chat model", models["chat"], index=_default_index(models["chat"], ["qwen3:8b", "qwen3", "llama3.1"])
    )
    EMBED_MODEL = st.selectbox(
        "Embedding model", models["embed"], index=_default_index(models["embed"], ["nomic-embed-text"])
    )


def render_pdf_upload_box(pid):
    """PDF upload lives on the main page, not tucked in the sidebar. Always
    visible above the chat once a project is selected — both before the
    first document is added and for adding more later."""
    docs_meta = db.list_documents(pid)

    with st.container(border=True):
        st.markdown(
            '<div class="answer-label">📄 Study material</div>',
            unsafe_allow_html=True,
        )

        if docs_meta:
            for d in docs_meta:
                st.markdown(
                    f'<div class="doc-chip"><span class="dot"></span>{d["filename"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No PDFs added to this project yet.")

        uploaded_file = st.file_uploader(
            "Add a PDF", type=["pdf"], key=f"uploader_{pid}", label_visibility="collapsed"
        )
        if uploaded_file:
            # Re-check against fresh DB state right before writing — docs_meta
            # captured at the top of this function can be stale by the time
            # the uploader widget fires, since Streamlit keeps the file
            # attached across unrelated reruns (theme toggle, new chat, etc).
            current_docs = db.list_documents(pid)
            already_added = any(d["filename"] == uploaded_file.name for d in current_docs)

            if not already_added:
                proj_dir = os.path.join(STUDY_DIR, pid)
                os.makedirs(proj_dir, exist_ok=True)

                safe_name = sanitize_name(uploaded_file.name)
                base_name, _ = os.path.splitext(safe_name)
                file_path = os.path.join(proj_dir, safe_name)
                md_path = os.path.join(proj_dir, base_name + ".md")

                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    with st.spinner(f"Parsing {uploaded_file.name}..."):
                        md_text = pymupdf4llm.to_markdown(file_path)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_text)
                    db.add_document(pid, uploaded_file.name, file_path, md_path)
                except Exception as e:
                    st.error(f"Couldn't add {uploaded_file.name}: {e}")
                    # Clean up any partial file so a retry doesn't collide.
                    for p in (file_path, md_path):
                        if os.path.exists(p):
                            os.remove(p)
                else:
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
        if not os.path.exists(d["md_path"]):
            st.error(f"Missing parsed text for {d['filename']} — try re-uploading it.")
            return None
        with open(d["md_path"], "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.create_documents([text], metadatas=[{"source": d["filename"]}])
        for i, ch in enumerate(chunks):
            ch.metadata["chunk_id"] = f"{d['id']}_{i}"
        all_chunks.extend(chunks)

    if not all_chunks:
        st.error("Parsed PDFs produced no readable text.")
        return None

    try:
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
    except Exception as e:
        st.error(f"Couldn't index this project's documents: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------
if not project_id:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🗂️</div>
            <div class="headline">Create or select a project</div>
            <div class="hint">Use the sidebar to start a new project — one per course
            or subject works well. Everything stays on your machine.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

chat_id = st.session_state.current_chat_id
if not chat_id:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">💬</div>
            <div class="headline">Start a new chat</div>
            <div class="hint">Use "＋ New chat" in the sidebar to begin asking
            questions about this project's material.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

active_project = next((p for p in projects if p["id"] == project_id), None)
active_chat = next((c for c in db.list_chats(project_id) if c["id"] == chat_id), None)

st.markdown(
    f"""
    <div class="chat-header">
        <span class="crumb">📁 {active_project['name'] if active_project else ''}<span class="sep">/</span></span>
        <h2>💬 {active_chat['title'] if active_chat else 'Chat'}</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Always visible above the chat — before the first PDF is added and for
# adding more material once the conversation is underway.
render_pdf_upload_box(project_id)

retriever = setup_project_engine(project_id, EMBED_MODEL, project_signature(project_id))

if retriever is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📄</div>
            <div class="headline">Add a PDF to get started</div>
            <div class="hint">Upload a PDF in the box above before asking
            questions about this project.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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


def render_answer_card(text, dl_key):
    """Assistant responses render inside a bordered card with a label and
    a download button, instead of a bare markdown wall in the chat bubble."""
    with st.container(border=True):
        st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(text)
        st.download_button(
            "⬇ Download",
            data=text,
            file_name=f"answer_{dl_key}.md",
            mime="text/markdown",
            key=f"dl_{chat_id}_{dl_key}",
        )


# ---------------------------------------------------------------------------
# Message history — keyed by position in the list rather than a message id,
# since db.py's row shape for messages isn't assumed here.
# ---------------------------------------------------------------------------
for idx, msg in enumerate(db.list_messages(chat_id)):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="📘"):
            render_answer_card(msg["content"], idx)

user_query = st.chat_input("Ask about your study material...")

if user_query:
    db.add_message(chat_id, "user", user_query)

    # First message in a chat becomes its title, like Codex threads.
    existing_messages = db.list_messages(chat_id)
    if len(existing_messages) == 1:
        db.rename_chat(chat_id, user_query[:60])

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_query)
        db.add_message(chat_id, "assistant", response)
        render_answer_card(response, len(db.list_messages(chat_id)) - 1)

    st.rerun()
