import streamlit as st
import requests
import pandas as pd
import json
import os

# Backend API base URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api")

st.set_page_config(
    page_title="PersonalAI — Personalized RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Helper functions for API calls
def get_user_profile():
    try:
        res = requests.get(f"{API_URL}/profile")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {
        "name": "Shivam",
        "education": "B.Tech CSE",
        "skills": "Python, SQL, Generative AI, React, FastAPI",
        "interests": "RAG, LLMs, Backend Engineering",
        "response_style": "Concise, technical, and structured"
    }

def get_documents():
    try:
        res = requests.get(f"{API_URL}/documents")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def get_document_stats():
    try:
        res = requests.get(f"{API_URL}/documents/stats")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

def get_conversations():
    try:
        res = requests.get(f"{API_URL}/chats")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

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
    conversations = get_conversations()
    
    with col_conv:
        conv_options = {"New Conversation": None}
        for c in conversations:
            conv_options[f"{c['title']} (ID: {c['id']})"] = c['id']
        
        selected_label = st.selectbox(
            "Select Conversation Session",
            list(conv_options.keys()),
            index=0
        )
        selected_id = conv_options[selected_label]

        if selected_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_id
            if selected_id:
                try:
                    res = requests.get(f"{API_URL}/chats/{selected_id}")
                    if res.status_code == 200:
                        st.session_state.chat_messages = res.json().get("messages", [])
                except Exception:
                    st.session_state.chat_messages = []
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
        # Display user message immediately
        st.chat_message("user").write(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query, "sources": []})

        with st.spinner("Searching FAISS index and generating response..."):
            try:
                payload = {
                    "conversation_id": st.session_state.current_chat_id,
                    "message": user_query,
                    "use_rag": use_rag_toggle
                }
                res = requests.post(f"{API_URL}/chats", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.current_chat_id = data["conversation_id"]
                    
                    st.chat_message("assistant").write(data["content"])
                    if data.get("sources"):
                        with st.expander("📚 View Document Sources"):
                            for idx, src in enumerate(data["sources"], 1):
                                page_str = f" | Page {src['page_number']}" if src.get("page_number") else ""
                                st.markdown(f"**Source {idx}:** `{src['document_name']}` (Chunk {src['chunk_id']}{page_str})")
                                st.caption(src['content'])

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": data["content"],
                        "sources": data.get("sources", [])
                    })
                else:
                    st.error(f"Error from server: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend server: {str(e)}")

# ----------------------------------------------------
# 2. 📁 DOCUMENTS & ANALYTICS TAB
# ----------------------------------------------------
elif navigation == "📁 Documents & Analytics":
    st.markdown('<div class="main-header">📁 Document Management & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload documents and view Pandas-powered token/chunk analytics.</div>', unsafe_allow_html=True)

    # Document Upload Section
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"]
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Document"):
            with st.spinner("Extracting text, counting tokens, chunking & creating FAISS embeddings..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{API_URL}/documents/upload", files=files)
                    if res.status_code == 201:
                        st.success(f"Successfully processed `{uploaded_file.name}`!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.markdown("---")

    # Document Statistics Dashboard (Pandas metrics)
    st.subheader("📊 Document & Token Analytics (Pandas Summary)")
    stats = get_document_stats()
    if stats and stats["total_documents"] > 0:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Documents", stats["total_documents"])
        m2.metric("Total Chunks", stats["total_chunks"])
        m3.metric("Total Words", f"{stats['total_words']:,}")
        m4.metric("Approx. Tokens", f"{stats['total_tokens']:,}")
        m5.metric("Avg Chunk Size", f"{stats['average_chunk_length']} chars")
    else:
        st.info("No documents uploaded yet. Upload a document to see token analytics.")

    # Uploaded Documents Table
    st.subheader("📄 Uploaded Documents")
    docs = get_documents()
    if docs:
        df_docs = pd.DataFrame(docs)
        df_display = df_docs[["id", "filename", "file_type", "chunk_count", "char_count", "word_count", "token_count", "upload_date"]]
        st.dataframe(df_display, use_container_width=True)

        col_del, col_sum = st.columns(2)
        with col_del:
            doc_to_delete = st.selectbox("Select document to delete", [f"{d['filename']} (ID: {d['id']})" for d in docs])
            if st.button("🗑️ Delete Selected Document"):
                doc_id = int(doc_to_delete.split("ID: ")[1].strip(")"))
                res = requests.delete(f"{API_URL}/documents/{doc_id}")
                if res.status_code == 204:
                    st.success("Document and its FAISS embeddings deleted.")
                    st.rerun()

        with col_sum:
            doc_to_sum = st.selectbox("Select document to summarize", [f"{d['filename']} (ID: {d['id']})" for d in docs])
            sum_type = st.radio("Summary Type", ["short", "detailed", "key_points"], horizontal=True)
            if st.button("✨ Generate Summary"):
                doc_id = int(doc_to_sum.split("ID: ")[1].strip(")"))
                with st.spinner("Generating LLM document summary..."):
                    res = requests.post(f"{API_URL}/documents/summarize", json={"document_id": doc_id, "summary_type": sum_type})
                    if res.status_code == 200:
                        st.subheader("Summary Result")
                        st.write(res.json()["summary"])

# ----------------------------------------------------
# 3. 👤 PROFILE TAB
# ----------------------------------------------------
elif navigation == "👤 Profile":
    st.markdown('<div class="main-header">👤 User Profile Personalization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalization settings stored in SQLite DB and injected into RAG prompts.</div>', unsafe_allow_html=True)

    profile = get_user_profile()

    with st.form("profile_form"):
        name = st.text_input("Name", value=profile.get("name", "Shivam"))
        education = st.text_input("Education", value=profile.get("education", "B.Tech CSE"))
        skills = st.text_area("Technical Skills", value=profile.get("skills", "Python, SQL, Generative AI, React, FastAPI"))
        interests = st.text_area("Interests / Specialization", value=profile.get("interests", "RAG, LLMs, Backend Engineering"))
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
            payload = {
                "name": name,
                "education": education,
                "skills": skills,
                "interests": interests,
                "response_style": response_style
            }
            res = requests.put(f"{API_URL}/profile", json=payload)
            if res.status_code == 200:
                st.success("Profile saved successfully to SQLite database!")

# ----------------------------------------------------
# 4. 📄 RESUME ANALYZER TAB
# ----------------------------------------------------
elif navigation == "📄 Resume Analyzer":
    st.markdown('<div class="main-header">📄 AI Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract structured candidate profile insights and analyze skills/projects.</div>', unsafe_allow_html=True)

    docs = get_documents()
    resume_docs = [d for d in docs if "resume" in d["filename"].lower() or d["file_type"] in ["PDF", "DOCX", "TXT"]]

    input_mode = st.radio("Select Input Source", ["Select Uploaded Document", "Paste Resume Text Direct"])

    selected_doc_id = None
    raw_resume_text = ""

    if input_mode == "Select Uploaded Document":
        if docs:
            doc_option = st.selectbox("Choose document", [f"{d['filename']} (ID: {d['id']})" for d in docs])
            selected_doc_id = int(doc_option.split("ID: ")[1].strip(")"))
        else:
            st.warning("No uploaded documents found. Please upload a resume first or use raw text.")
    else:
        raw_resume_text = st.text_area("Paste Resume Text Here", height=200)

    custom_q = st.text_input("Custom Analysis Focus (Optional)", "Summarize key skills, education, and projects")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing resume structure with AI..."):
            payload = {
                "document_id": selected_doc_id,
                "resume_text": raw_resume_text,
                "question": custom_q
            }
            res = requests.post(f"{API_URL}/resume/analyze", json=payload)
            if res.status_code == 200:
                data = res.json()
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
            else:
                st.error(f"Analysis failed: {res.text}")

# ----------------------------------------------------
# 5. 🎯 JOB DESCRIPTION MATCHER TAB
# ----------------------------------------------------
elif navigation == "🎯 Job Description Matcher":
    st.markdown('<div class="main-header">🎯 Job Description Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare your candidate profile/resume against target job descriptions.</div>', unsafe_allow_html=True)

    jd_text = st.text_area("Paste Job Description (JD) Text Here", height=180, placeholder="We are looking for a Generative AI Engineer skilled in Python, FastAPI, LangChain, FAISS, SQL...")

    docs = get_documents()
    selected_doc_id = None
    if docs:
        doc_option = st.selectbox("Optionally select uploaded resume to include", ["None (Use SQLite User Profile Only)"] + [f"{d['filename']} (ID: {d['id']})" for d in docs])
        if "ID: " in doc_option:
            selected_doc_id = int(doc_option.split("ID: ")[1].strip(")"))

    if st.button("🚀 Analyze Fit & Match Skills"):
        if not jd_text.strip():
            st.warning("Please paste a Job Description first.")
        else:
            with st.spinner("Performing AI Gap Analysis..."):
                payload = {
                    "job_description": jd_text,
                    "resume_document_id": selected_doc_id
                }
                res = requests.post(f"{API_URL}/job/analyze", json=payload)
                if res.status_code == 200:
                    data = res.json()
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

# ----------------------------------------------------
# 6. 📜 CHAT HISTORY TAB
# ----------------------------------------------------
elif navigation == "📜 Chat History":
    st.markdown('<div class="main-header">📜 Conversation Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect and manage all saved SQLite chat sessions.</div>', unsafe_allow_html=True)

    conversations = get_conversations()
    if conversations:
        for conv in conversations:
            with st.expander(f"💬 {conv['title']} — ID: {conv['id']} ({conv['created_at'][:19]})"):
                res = requests.get(f"{API_URL}/chats/{conv['id']}")
                if res.status_code == 200:
                    messages = res.json().get("messages", [])
                    for m in messages:
                        st.markdown(f"**{m['role'].capitalize()}:** {m['content']}")
                        if m.get("sources"):
                            st.caption(f"Sources used: {len(m['sources'])} chunks")
                        st.divider()

                if st.button(f"🗑️ Delete Session {conv['id']}", key=f"del_{conv['id']}"):
                    requests.delete(f"{API_URL}/chats/{conv['id']}")
                    st.rerun()
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
