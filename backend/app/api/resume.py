import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models.database_models import Document
from backend.app.models.domain_schemas import ResumeAnalyzeRequest, ResumeAnalyzeResponse
from backend.app.services.document_service import DocumentService
from backend.app.services.rag_service import RAGService

router = APIRouter(prefix="/resume", tags=["Resume Analyzer"])
rag_service = RAGService()

@router.post("/analyze", response_model=ResumeAnalyzeResponse)
def analyze_resume(req: ResumeAnalyzeRequest, db: Session = Depends(get_db)):
    """
    Analyzes candidate resume to extract skills, technologies, projects, and education.
    Accepts either a document_id or raw resume text.
    """
    resume_text = ""
    if req.document_id:
        doc = db.query(Document).filter(Document.id == req.document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Resume document not found")
        if not os.path.exists(doc.file_path):
            raise HTTPException(status_code=404, detail="Resume physical file missing")
        resume_text = DocumentService.extract_text_from_file(doc.file_path, doc.file_type)
    elif req.resume_text and req.resume_text.strip():
        resume_text = req.resume_text
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either document_id of uploaded resume or raw resume_text"
        )

    analysis_result = rag_service.analyze_resume_text(
        db,
        resume_text=resume_text,
        question=req.question
    )

    return ResumeAnalyzeResponse(**analysis_result)
