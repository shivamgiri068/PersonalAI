# System Architecture & Technical Specification

This document details the architectural design, component flow, database schemas, and data pipelines for **PersonalAI — Personalized RAG Assistant**.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    subgraph Frontend Layer [Streamlit UI]
        UI[Streamlit Multi-Tab Frontend]
    end

    subgraph Backend Layer [FastAPI REST APIs]
        API[FastAPI Router /api]
        DOC_SVC[Document Service]
        NLP_SVC[NLP Service]
        EMB_SVC[Embedding Service]
        VECTOR_SVC[FAISS Vector Store]
        RAG_SVC[RAG Service]
        LLM_SVC[OpenAI LLM Service]
        MEM_SVC[Memory Service]
    end

    subgraph Storage Layer
        DB[(SQLite Database personal_ai.db)]
        FAISS_DB[(FAISS Vector Index faiss_index.bin)]
        UPLOADS[(Physical Uploads data/uploads/)]
    end

    UI -->|HTTP Requests| API
    API --> DOC_SVC
    API --> RAG_SVC
    API --> MEM_SVC
    DOC_SVC --> NLP_SVC
    DOC_SVC --> UPLOADS
    RAG_SVC --> EMB_SVC
    RAG_SVC --> VECTOR_SVC
    RAG_SVC --> LLM_SVC
    EMB_SVC -->|OpenAI Embeddings| FAISS_DB
    MEM_SVC --> DB
    DOC_SVC --> DB
```

---

## 2. Document Ingestion Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit UI
    participant FastAPI (/api/documents/upload)
    participant DocumentService
    participant NLPService
    participant EmbeddingService
    participant FAISS Vector Store
    participant SQLite DB

    User->>Streamlit UI: Upload Document (PDF/DOCX/TXT/MD)
    Streamlit UI->>FastAPI (/api/documents/upload): Multipart POST
    FastAPI->>DocumentService: Extract text & Save file to disk
    DocumentService-->>FastAPI: Raw Text
    FastAPI->>NLPService: Clean text & Chunk (500 chars, 50 overlap)
    NLPService-->>FastAPI: Chunks + Token Stats
    FastAPI->>SQLite DB: Insert Document Record
    SQLite DB-->>FastAPI: Document ID
    FastAPI->>EmbeddingService: Batch Embed Chunks
    EmbeddingService-->>FastAPI: 1536-d Vectors
    FastAPI->>FAISS Vector Store: Add Vectors & Metadata to IndexFlatIP
    FAISS Vector Store-->>FastAPI: Index Updated & Saved to disk
    FastAPI-->>Streamlit UI: HTTP 201 Created (Document Response)
```

---

## 3. RAG Query Execution & Hallucination Guardrail Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit UI
    participant FastAPI (/api/chats)
    participant SQLite DB
    participant EmbeddingService
    participant FAISS Vector Store
    participant LLMService

    User->>Streamlit UI: Ask Question ("What skills are in my resume?")
    Streamlit UI->>FastAPI (/api/chats): POST ChatMessageRequest
    FastAPI->>SQLite DB: Save User Message
    FastAPI->>SQLite DB: Fetch User Profile (ID=1)
    FastAPI->>EmbeddingService: Embed Question -> Query Vector
    EmbeddingService-->>FastAPI: Query Vector (1536-d)
    FastAPI->>FAISS Vector Store: Search Top-4 Nearest Chunks
    FAISS Vector Store-->>FastAPI: Top-4 Chunks + Metadata + Scores
    FastAPI->>LLMService: Construct Personalized RAG System Prompt
    LLMService->>OpenAI API: ChatCompletion (gpt-3.5-turbo, temp=0.2)
    OpenAI API-->>LLMService: Completion Text
    FastAPI->>SQLite DB: Save Assistant Response & Sources
    FastAPI-->>Streamlit UI: Return Answer + Expandable Source Citations
```

---

## 4. Relational Database Schema (SQLite + SQLAlchemy)

```mermaid
erDiagram
    USERS {
        int id PK
        string name
        string education
        string skills
        string interests
        string response_style
        datetime created_at
        datetime updated_at
    }

    DOCUMENTS {
        int id PK
        string filename
        string file_type
        string file_path
        datetime upload_date
        int chunk_count
        int char_count
        int word_count
        int token_count
    }

    CONVERSATIONS {
        int id PK
        string title
        datetime created_at
        datetime updated_at
    }

    MESSAGES {
        int id PK
        int conversation_id FK
        string role
        text content
        text sources_json
        datetime created_at
    }

    CONVERSATIONS ||--o{ MESSAGES : "contains"
```
