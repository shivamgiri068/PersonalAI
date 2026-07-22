import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.config import settings
from backend.app.models.database_models import Document
from backend.app.models.domain_schemas import DocumentResponse, DocumentStatsResponse, DocumentSummaryRequest, DocumentSummaryResponse
from backend.app.services.document_service import DocumentService
from backend.app.services.nlp_service import NLPService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store import FAISSVectorStore
from backend.app.services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["Documents"])
vector_store = FAISSVectorStore()
embedding_service = EmbeddingService()
rag_service = RAGService(vector_store=vector_store)

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md", "markdown"}
MAX_FILE_SIZE_MB = 15

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF, DOCX, TXT, or Markdown document.
    Extracts text, computes NLP stats (chars, words, tokens), splits into chunks,
    generates embeddings, stores in FAISS, and persists document metadata in SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '.{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOADS_DIR, file.filename)

    # Save file to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Check file size limit
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB")

    # Extract raw text
    try:
        raw_text = DocumentService.extract_text_from_file(file_path, ext)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {str(e)}")

    clean_text = NLPService.clean_text(raw_text)
    if not clean_text:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="Document appears to be empty or contains no readable text.")

    # Calculate tokenization & NLP stats
    char_count = NLPService.count_characters(clean_text)
    word_count = NLPService.count_words(clean_text)
    token_count = NLPService.count_tokens(clean_text)

    # Split into chunks
    chunks = NLPService.chunk_text(
        clean_text,
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
    )

    # Create Document DB record first to get ID
    doc_model = Document(
        filename=file.filename,
        file_type=ext.upper(),
        file_path=file_path,
        chunk_count=len(chunks),
        char_count=char_count,
        word_count=word_count,
        token_count=token_count
    )
    db.add(doc_model)
    db.commit()
    db.refresh(doc_model)

    # Generate Embeddings & Index into FAISS
    chunk_texts = [c["content"] for c in chunks]
    embeddings = embedding_service.get_embeddings_batch(chunk_texts)

    chunk_metadatas = [
        {
            "document_id": doc_model.id,
            "filename": file.filename,
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "char_count": c["char_count"],
            "word_count": c["word_count"],
            "token_count": c["token_count"]
        }
        for c in chunks
    ]

    vector_store.add_chunks(embeddings, chunk_metadatas)

    return doc_model

@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """Retrieve list of all uploaded documents."""
    return db.query(Document).order_by(Document.upload_date.desc()).all()

@router.get("/stats", response_model=DocumentStatsResponse)
def get_document_stats(db: Session = Depends(get_db)):
    """
    Computes tabular document analytics & summaries using Pandas.
    """
    documents = db.query(Document).all()
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
        for d in documents
    ]
    return DocumentService.generate_pandas_document_analytics(doc_dicts)

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document record from SQLite, remove file from disk, and strip chunks from FAISS index.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from FAISS index
    vector_store.delete_document_chunks(document_id)

    # Remove from disk
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass

    # Remove from DB
    db.delete(doc)
    db.commit()
    return None

@router.post("/summarize", response_model=DocumentSummaryResponse)
def summarize_document(req: DocumentSummaryRequest, db: Session = Depends(get_db)):
    """
    Summarize document text (short summary, detailed summary, or key points).
    """
    doc = db.query(Document).filter(Document.id == req.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Physical document file missing on server")

    raw_text = DocumentService.extract_text_from_file(doc.file_path, doc.file_type)
    summary_text = rag_service.summarize_document_text(raw_text, summary_type=req.summary_type)

    return DocumentSummaryResponse(
        document_id=doc.id,
        filename=doc.filename,
        summary_type=req.summary_type,
        summary=summary_text
    )
