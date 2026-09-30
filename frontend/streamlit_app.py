import sys
import os

# Add project root to sys.path for Streamlit Cloud module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import json
import shutil

# Direct Service Imports
from backend.app.database import SessionLocal, init_db
from backend.app.models.database_models import User, Document, Conversation, Message
from backend.app.services.document_service import DocumentService
from backend.app.services.nlp_service import NLPService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store import FAISSVectorStore
from backend.app.services.rag_service import RAGService
from backend.app.services.memory_service import MemoryService
from backend.app.config import settings

# Page Setup
st.set_page_config(
    page_title="PersonalAI — Personalized RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database on startup
init_db()

# Initialize Persistent Services
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()
if "embedding_service" not in st.session_state:
    st.session_state.embedding_service = EmbeddingService()
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService(vector_store=st.session_state.vector_store)

# Helper function to obtain DB session
def get_db():
    return SessionLocal()

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper DB Functions
def fetch_user_profile():
    db = get_db()
    try:
        user = db.query(User).filter(User.id == 1).first()
        if not user:
            user = User(
                id=1,
                name="Shivam",
                education="B.Tech CSE",
                skills="Python, SQL, Generative AI, React, FastAPI",
                interests="RAG, LLMs, Backend Engineering",
                response_style="Concise, technical, and structured"
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()

def fetch_documents():
    db = get_db()
    try:
        return db.query(Document).order_by(Document.upload_date.desc()).all()
    finally:
        db.close()

def fetch_conversations():
    db = get_db()
    try:
        return MemoryService.get_conversations(db)
    finally:
        db.close()

# Sidebar Navigation
st.sidebar.title("🤖 PersonalAI")
st.sidebar.caption("Personalized RAG Portfolio Assistant")

navigation = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat",
        "📁 Documents & Analytics",
        "👤 Profile",
        "📄 Resume Analyzer",
        "🎯 Job Description Matcher",
        "📜 Chat History",
        "ℹ️ About & AI Concepts"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Target Role**: Fresher GenAI Engineer")
st.sidebar.markdown("**Tech Stack**: Python | FastAPI | LangChain | FAISS | OpenAI | SQLite | Streamlit | Pandas | NumPy")

# ----------------------------------------------------
# 1. 💬 CHAT TAB
# ----------------------------------------------------
if navigation == "💬 Chat":
    st.markdown('<div class="main-header">💬 PersonalAI RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your uploaded documents with personalized AI context.</div>', unsafe_allow_html=True)

    # Session State Initialization
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Chat controls
    col_conv, col_new = st.columns([4, 1])
    conversations = fetch_conversations()
    
    with col_conv:
        conv_options = {"New Conversation": None}
        for c in conversations:
            conv_options[f"{c.title} (ID: {c.id})"] = c.id
        
        selected_label = st.selectbox(
            "Select Conversation Session",
            list(conv_options.keys()),
            index=0
        )
        selected_id = conv_options[selected_label]

        if selected_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_id
            if selected_id:
                db = get_db()
                try:
                    conv = MemoryService.get_conversation(db, selected_id)
                    if conv:
                        msgs = []
                        for m in conv.messages:
                            sources_list = []
                            if m.sources_json:
                                try:
                                    sources_list = json.loads(m.sources_json)
                                except Exception:
                                    pass
                            msgs.append({
                                "role": m.role,
                                "content": m.content,
                                "sources": sources_list
                            })
                        st.session_state.chat_messages = msgs
                finally:
                    db.close()
            else:
                st.session_state.chat_messages = []

    with col_new:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_chat_id = None
            st.session_state.chat_messages = []
            st.rerun()

    use_rag_toggle = st.checkbox("🔍 Enable RAG Vector Retrieval (FAISS)", value=True)

    # Display Chat History
    st.markdown("---")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Document Sources"):
                    for idx, src in enumerate(msg["sources"], 1):
                        page_str = f" | Page {src['page_number']}" if src.get("page_number") else ""
                        st.markdown(f"**Source {idx}:** `{src['document_name']}` (Chunk {src['chunk_id']}{page_str})")
                        st.caption(src['content'])

    # Chat Input
    if user_query := st.chat_input("Ask a question about your documents (e.g., 'What skills are in my resume?')..."):
        st.chat_message("user").write(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query, "sources": []})

        with st.spinner("Searching FAISS index and generating response..."):
            db = get_db()
            try:
                if not st.session_state.current_chat_id:
                    new_conv = MemoryService.create_conversation(db, title=user_query[:30])
                    st.session_state.current_chat_id = new_conv.id

                conv_id = st.session_state.current_chat_id
                MemoryService.add_message(db, conversation_id=conv_id, role="user", content=user_query)

                if use_rag_toggle:
                    answer, sources = st.session_state.rag_service.query_rag(db, question=user_query)
                else:
                    answer = st.session_state.rag_service.llm_service.generate_completion(user_query)
                    sources = []

                MemoryService.add_message(
                    db,
                    conversation_id=conv_id,
                    role="assistant",
                    content=answer,
                    sources=sources
                )

                st.chat_message("assistant").write(answer)
                if sources:
                    with st.expander("📚 View Document Sources"):
                        for idx, src in enumerate(sources, 1):
                            page_str = f" | Page {src['page_number']}" if src.get("page_number") else ""
                            st.markdown(f"**Source {idx}:** `{src['document_name']}` (Chunk {src['chunk_id']}{page_str})")
                            st.caption(src['content'])

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
            finally:
                db.close()

# ----------------------------------------------------
# 2. 📁 DOCUMENTS & ANALYTICS TAB
# ----------------------------------------------------
elif navigation == "📁 Documents & Analytics":
    st.markdown('<div class="main-header">📁 Document Management & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload documents and view Pandas-powered token/chunk analytics.</div>', unsafe_allow_html=True)

    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"]
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Document"):
            with st.spinner("Extracting text, counting tokens, chunking & creating FAISS embeddings..."):
                os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
                file_path = os.path.join(settings.UPLOADS_DIR, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                ext = uploaded_file.name.split(".")[-1].lower()
                raw_text = DocumentService.extract_text_from_file(file_path, ext)
                clean_text = NLPService.clean_text(raw_text)

                if clean_text:
                    char_cnt = NLPService.count_characters(clean_text)
                    word_cnt = NLPService.count_words(clean_text)
                    token_cnt = NLPService.count_tokens(clean_text)

                    chunks = NLPService.chunk_text(
                        clean_text,
                        chunk_size=settings.DEFAULT_CHUNK_SIZE,
                        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
                    )

                    db = get_db()
                    try:
                        doc_model = Document(
                            filename=uploaded_file.name,
                            file_type=ext.upper(),
                            file_path=file_path,
                            chunk_count=len(chunks),
                            char_count=char_cnt,
                            word_count=word_cnt,
                            token_count=token_cnt
                        )
                        db.add(doc_model)
                        db.commit()
                        db.refresh(doc_model)

                        chunk_texts = [c["content"] for c in chunks]
                        embeddings = st.session_state.embedding_service.get_embeddings_batch(chunk_texts)

                        chunk_metadatas = [
                            {
                                "document_id": doc_model.id,
                                "filename": uploaded_file.name,
                                "chunk_id": c["chunk_id"],
                                "content": c["content"],
                                "char_count": c["char_count"],
                                "word_count": c["word_count"],
                                "token_count": c["token_count"]
                            }
                            for c in chunks
                        ]

                        st.session_state.vector_store.add_chunks(embeddings, chunk_metadatas)
                        st.success(f"Successfully processed `{uploaded_file.name}`!")
                        st.rerun()
                    finally:
                        db.close()
                else:
                    st.error("Uploaded document contains no readable text.")

    st.markdown("---")

    st.subheader("📊 Document & Token Analytics (Pandas Summary)")
    docs = fetch_documents()
    doc_dicts = [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "chunk_count": d.chunk_count,
            "char_count": d.char_count,
            "word_count": d.word_count,
            "token_count": d.token_count
        }
        for d in docs
    ]
    stats = DocumentService.generate_pandas_document_analytics(doc_dicts)

    if stats["total_documents"] > 0:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Documents", stats["total_documents"])
        m2.metric("Total Chunks", stats["total_chunks"])
        m3.metric("Total Words", f"{stats['total_words']:,}")
        m4.metric("Approx. Tokens", f"{stats['total_tokens']:,}")
        m5.metric("Avg Chunk Size", f"{stats['average_chunk_length']} chars")
    else:
        st.info("No documents uploaded yet. Upload a document to see token analytics.")

    st.subheader("📄 Uploaded Documents")
    if docs:
        df_docs = pd.DataFrame(doc_dicts)
        st.dataframe(df_docs, use_container_width=True)

        col_del, col_sum = st.columns(2)
        with col_del:
            doc_to_delete = st.selectbox("Select document to delete", [f"{d.filename} (ID: {d.id})" for d in docs])
            if st.button("🗑️ Delete Selected Document"):
                doc_id = int(doc_to_delete.split("ID: ")[1].strip(")"))
                db = get_db()
                try:
                    doc = db.query(Document).filter(Document.id == doc_id).first()
                    if doc:
                        st.session_state.vector_store.delete_document_chunks(doc_id)
                        if os.path.exists(doc.file_path):
                            try:
                                os.remove(doc.file_path)
                            except Exception:
                                pass
                        db.delete(doc)
                        db.commit()
                        st.success("Document and its FAISS embeddings deleted.")
                        st.rerun()
                finally:
                    db.close()

        with col_sum:
            doc_to_sum = st.selectbox("Select document to summarize", [f"{d.filename} (ID: {d.id})" for d in docs])
            sum_type = st.radio("Summary Type", ["short", "detailed", "key_points"], horizontal=True)
            if st.button("✨ Generate Summary"):
                doc_id = int(doc_to_sum.split("ID: ")[1].strip(")"))
                db = get_db()
                try:
                    doc = db.query(Document).filter(Document.id == doc_id).first()
                    if doc and os.path.exists(doc.file_path):
                        raw_text = DocumentService.extract_text_from_file(doc.file_path, doc.file_type)
                        summary_text = st.session_state.rag_service.summarize_document_text(raw_text, summary_type=sum_type)
                        st.subheader("Summary Result")
                        st.write(summary_text)
                finally:
                    db.close()

# ----------------------------------------------------
# 3. 👤 PROFILE TAB
# ----------------------------------------------------
elif navigation == "👤 Profile":
    st.markdown('<div class="main-header">👤 User Profile Personalization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalization settings stored in SQLite DB and injected into RAG prompts.</div>', unsafe_allow_html=True)

    profile_obj = fetch_user_profile()

    with st.form("profile_form"):
        name = st.text_input("Name", value=profile_obj.name)
        education = st.text_input("Education", value=profile_obj.education or "B.Tech CSE")
        skills = st.text_area("Technical Skills", value=profile_obj.skills or "Python, SQL, Generative AI, React, FastAPI")
        interests = st.text_area("Interests / Specialization", value=profile_obj.interests or "RAG, LLMs, Backend Engineering")
        response_style = st.selectbox(
            "Preferred AI Response Style",
            [
                "Concise, technical, and structured",
                "Detailed with step-by-step code examples",
                "Executive high-level summary",
                "Conversational and simple"
            ],
            index=0
        )

        submitted = st.form_submit_button("💾 Save Profile")
        if submitted:
            db = get_db()
            try:
                user = db.query(User).filter(User.id == 1).first()
                if not user:
                    user = User(id=1)
                    db.add(user)
                user.name = name
                user.education = education
                user.skills = skills
                user.interests = interests
                user.response_style = response_style
                db.commit()
                st.success("Profile saved successfully to SQLite database!")
            finally:
                db.close()

# ----------------------------------------------------
# 4. 📄 RESUME ANALYZER TAB
# ----------------------------------------------------
elif navigation == "📄 Resume Analyzer":
    st.markdown('<div class="main-header">📄 AI Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract structured candidate profile insights and analyze skills/projects.</div>', unsafe_allow_html=True)

    docs = fetch_documents()
    input_mode = st.radio("Select Input Source", ["Select Uploaded Document", "Paste Resume Text Direct"])

    selected_doc_id = None
    raw_resume_text = ""

    if input_mode == "Select Uploaded Document":
        if docs:
            doc_option = st.selectbox("Choose document", [f"{d.filename} (ID: {d.id})" for d in docs])
            selected_doc_id = int(doc_option.split("ID: ")[1].strip(")"))
        else:
            st.warning("No uploaded documents found. Please upload a resume first or paste raw text below.")
    else:
        raw_resume_text = st.text_area("Paste Resume Text Here", height=200, placeholder="Paste your resume content here...")

    custom_q = st.text_input("Custom Analysis Focus (Optional)", "Summarize key skills, education, and projects")

    if st.button("🔍 Analyze Resume"):
        resume_content = ""
        if selected_doc_id:
            db = get_db()
            try:
                doc = db.query(Document).filter(Document.id == selected_doc_id).first()
                if doc and os.path.exists(doc.file_path):
                    resume_content = DocumentService.extract_text_from_file(doc.file_path, doc.file_type)
            finally:
                db.close()
        elif raw_resume_text.strip():
            resume_content = raw_resume_text

        if not resume_content.strip():
            st.warning("Please provide resume text by selecting an uploaded document or pasting text.")
        else:
            with st.spinner("Analyzing resume structure with AI..."):
                db = get_db()
                try:
                    data = st.session_state.rag_service.analyze_resume_text(
                        db,
                        resume_text=resume_content,
                        question=custom_q
                    )
                    st.subheader("📋 Analysis Results")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🎓 Education")
                        for ed in data.get("education", []):
                            st.markdown(f"- {ed}")
                        
                        st.markdown("### 🛠️ Core Skills")
                        for sk in data.get("skills", []):
                            st.markdown(f"- {sk}")

                    with c2:
                        st.markdown("### 💻 Technologies")
                        for tech in data.get("technologies", []):
                            st.markdown(f"- `{tech}`")

                        st.markdown("### 🚀 Projects")
                        for proj in data.get("projects", []):
                            st.markdown(f"- {proj}")

                    st.markdown("---")
                    st.markdown("### 📝 Professional Summary & Insight")
                    st.write(data.get("summary", ""))
                finally:
                    db.close()

# ----------------------------------------------------
# 5. 🎯 JOB DESCRIPTION MATCHER TAB
# ----------------------------------------------------
elif navigation == "🎯 Job Description Matcher":
    st.markdown('<div class="main-header">🎯 Job Description Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare your candidate profile/resume against target job descriptions.</div>', unsafe_allow_html=True)

    jd_text = st.text_area("Paste Job Description (JD) Text Here", height=180, placeholder="We are looking for a Generative AI Engineer skilled in Python, FastAPI, LangChain, FAISS, SQL...")

    docs = fetch_documents()
    selected_doc_id = None
    if docs:
        doc_option = st.selectbox("Optionally select uploaded resume to include", ["None (Use SQLite User Profile Only)"] + [f"{d.filename} (ID: {d.id})" for d in docs])
        if "ID: " in doc_option:
            selected_doc_id = int(doc_option.split("ID: ")[1].strip(")"))

    if st.button("🚀 Analyze Fit & Match Skills"):
        if not jd_text.strip():
            st.warning("Please paste a Job Description first.")
        else:
            with st.spinner("Performing AI Gap Analysis..."):
                resume_ctx = ""
                if selected_doc_id:
                    db = get_db()
                    try:
                        doc = db.query(Document).filter(Document.id == selected_doc_id).first()
                        if doc and os.path.exists(doc.file_path):
                            resume_ctx = DocumentService.extract_text_from_file(doc.file_path, doc.file_type)
                    finally:
                        db.close()

                db = get_db()
                try:
                    data = st.session_state.rag_service.analyze_job_description(
                        db,
                        job_description=jd_text,
                        resume_context=resume_ctx
                    )
                    st.subheader("🎯 Skill Gap Analysis Results")

                    col_match, col_miss = st.columns(2)
                    with col_match:
                        st.success("### ✅ Matching Skills Found")
                        for s in data.get("matching_skills", []):
                            st.markdown(f"- **{s}**")

                    with col_miss:
                        st.warning("### ⚠️ Missing / Gap Skills")
                        for s in data.get("missing_skills", []):
                            st.markdown(f"- **{s}**")

                    st.markdown("---")
                    st.markdown("### 📚 Suggested Preparation Topics for Interview")
                    for topic in data.get("preparation_topics", []):
                        st.markdown(f"1. {topic}")

                    st.markdown("---")
                    st.markdown("### 💡 Overall Recommendation")
                    st.write(data.get("recommendation_summary", ""))
                finally:
                    db.close()

# ----------------------------------------------------
# 6. 📜 CHAT HISTORY TAB
# ----------------------------------------------------
elif navigation == "📜 Chat History":
    st.markdown('<div class="main-header">📜 Conversation Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect and manage all saved SQLite chat sessions.</div>', unsafe_allow_html=True)

    conversations = fetch_conversations()
    if conversations:
        for conv in conversations:
            with st.expander(f"💬 {conv.title} — ID: {conv.id} ({str(conv.created_at)[:19]})"):
                db = get_db()
                try:
                    conv_detail = MemoryService.get_conversation(db, conv.id)
                    if conv_detail:
                        for m in conv_detail.messages:
                            st.markdown(f"**{m.role.capitalize()}:** {m.content}")
                            if m.sources_json:
                                try:
                                    srcs = json.loads(m.sources_json)
                                    st.caption(f"Sources used: {len(srcs)} chunks")
                                except Exception:
                                    pass
                            st.divider()
                finally:
                    db.close()

                if st.button(f"🗑️ Delete Session {conv.id}", key=f"del_{conv.id}"):
                    db = get_db()
                    try:
                        MemoryService.delete_conversation(db, conv.id)
                        st.rerun()
                    finally:
                        db.close()
    else:
        st.info("No past chat sessions stored in SQLite database.")

# ----------------------------------------------------
# 7. ℹ️ ABOUT & AI CONCEPTS TAB
# ----------------------------------------------------
elif navigation == "ℹ️ About & AI Concepts":
    st.markdown('<div class="main-header">ℹ️ About PersonalAI & Fresher AI Concepts</div>', unsafe_allow_html=True)
    
    st.subheader("🏗️ Core Architecture & Data Pipeline")
    st.code("""
    Document (PDF, DOCX, TXT, MD)
       ↓
    Text Extraction (pypdf, python-docx)
       ↓
    Text Cleaning & Normalization (NLP Service)
       ↓
    Recursive Chunking (configurable size & overlap)
       ↓
    Embeddings (OpenAI text-embedding-3-small)
       ↓
    FAISS Vector Database (IndexFlatIP cosine similarity)
       ↓
    Relevant Context Retrieval (Top-K similarity search)
       ↓
    Prompt Construction (System instructions + Profile + Context + Query)
       ↓
    LLM (OpenAI GPT-3.5-Turbo)
       ↓
    Answer + Verified Source Citations
    """, language="text")

    st.subheader("📚 Fresher Interview Key Concepts")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. What is RAG?**
        Retrieval-Augmented Generation combines an external information retrieval step (FAISS) with a Generative LLM to answer questions using custom private documents without retraining or fine-tuning.

        **2. Why FAISS?**
        FAISS (Facebook AI Similarity Search) is an ultra-fast vector index for searching embedding similarities locally without external network overhead or costly SaaS vector DBs.

        **3. Tokenization & Embeddings**
        Tokenization converts text into numerical token IDs. Embeddings convert token IDs into continuous high-dimensional vector representations capturing semantic meaning.
        """)
    with c2:
        st.markdown("""
        **4. Self-Attention Mechanism**
        Self-attention in Transformers allows tokens to weight relationships with all other tokens in a sequence dynamically, understanding context and long-range dependencies.

        **5. Hallucination Control in RAG**
        Controlled via system prompt instructions requiring the model to rely solely on retrieved context and explicitly declare when information is unavailable.
        """)
