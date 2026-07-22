import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models.database_models import Document
from backend.app.models.domain_schemas import JobAnalyzeRequest, JobAnalyzeResponse
from backend.app.services.document_service import DocumentService
from backend.app.services.rag_service import RAGService

router = APIRouter(prefix="/job", tags=["Job Description Analyzer"])
rag_service = RAGService()

@router.post("/analyze", response_model=JobAnalyzeResponse)
def analyze_job_description(req: JobAnalyzeRequest, db: Session = Depends(get_db)):
    """
    Compares candidate profile/resume against job description.
    Returns matching skills, missing skills, mentioned technologies, and preparation topics.
    """
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty")

    resume_context = ""
    if req.resume_document_id:
        doc = db.query(Document).filter(Document.id == req.resume_document_id).first()
        if doc and os.path.exists(doc.file_path):
            resume_context = DocumentService.extract_text_from_file(doc.file_path, doc.file_type)

    analysis_result = rag_service.analyze_job_description(
        db,
        job_description=req.job_description,
        resume_context=resume_context
    )

    return JobAnalyzeResponse(**analysis_result)
