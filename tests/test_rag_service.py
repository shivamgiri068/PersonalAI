import tempfile
import shutil
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models.database_models import User
from backend.app.services.rag_service import RAGService
from backend.app.services.vector_store import FAISSVectorStore

def test_rag_query_execution():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSession()

    temp_dir = tempfile.mkdtemp()
    try:
        user = User(id=1, name="Shivam", education="B.Tech CSE", skills="Python, RAG", interests="AI", response_style="Concise")
        db.add(user)
        db.commit()

        vector_store = FAISSVectorStore(vector_dir=temp_dir, dimension=1536)
        rag_service = RAGService(vector_store=vector_store)

        answer, sources = rag_service.query_rag(db, question="What skills are listed in my profile?")
        assert isinstance(answer, str)
        assert isinstance(sources, list)
    finally:
        db.close()
        shutil.rmtree(temp_dir)

def test_summarization_and_resume_analysis():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSession()

    try:
        user = User(id=1, name="Shivam")
        db.add(user)
        db.commit()

        rag_service = RAGService()
        summary = rag_service.summarize_document_text("FastAPI and Streamlit build high performance GenAI tools.")
        assert len(summary) > 0

        resume_result = rag_service.analyze_resume_text(db, "Experience in Python, SQL, FAISS, and LangChain.")
        assert "skills" in resume_result
        assert "projects" in resume_result
    finally:
        db.close()
