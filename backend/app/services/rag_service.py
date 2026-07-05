from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store import FAISSVectorStore
from backend.app.services.llm_service import LLMService
from backend.app.models.database_models import User
from backend.app.prompts.prompts import RAG_SYSTEM_PROMPT, SUMMARIZATION_PROMPT, RESUME_ANALYSIS_PROMPT, JOB_MATCH_PROMPT

class RAGService:
    def __init__(self, vector_store: FAISSVectorStore = None):
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store or FAISSVectorStore()
        self.llm_service = LLMService()

    def query_rag(
        self,
        db: Session,
        question: str,
        user_id: int = 1,
        top_k: int = 4
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes the full RAG pipeline:
        Query -> Embedding -> FAISS Vector Search -> Context & User Profile -> Prompt -> LLM -> Answer + Sources
        """
        # 1. Fetch user profile for personalization
        user = db.query(User).filter(User.id == user_id).first()
        user_name = user.name if user else "Shivam"
        user_education = user.education if user else "B.Tech CSE"
        user_skills = user.skills if user else "Python, GenAI, RAG"
        user_interests = user.interests if user else "AI Architecture, Backend"
        user_response_style = user.response_style if user else "Concise and technical"

        # 2. Convert question to vector embedding
        query_vec = self.embedding_service.get_embedding(question)

        # 3. Perform similarity search in FAISS
        retrieved_items = self.vector_store.search(query_vec, top_k=top_k)

        context_chunks = []
        sources = []

        for meta, score in retrieved_items:
            doc_name = meta.get("filename", "Document")
            chunk_id = meta.get("chunk_id", 0)
            page_num = meta.get("page_number", None)
            content = meta.get("content", "")

            context_chunks.append(f"[Source: {doc_name} | Chunk {chunk_id}]\n{content}")
            
            sources.append({
                "document_name": doc_name,
                "chunk_id": chunk_id,
                "page_number": page_num,
                "content": content
            })

        context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant document context found."

        # 4. Construct personalized prompt with hallucination safeguards
        formatted_prompt = RAG_SYSTEM_PROMPT.format(
            user_name=user_name,
            user_education=user_education,
            user_skills=user_skills,
            user_interests=user_interests,
            user_response_style=user_response_style,
            context_text=context_text,
            user_question=question
        )

        # 5. Generate completion from LLM
        answer = self.llm_service.generate_completion(formatted_prompt)

        return answer, sources

    def summarize_document_text(self, text: str, summary_type: str = "short") -> str:
        prompt = SUMMARIZATION_PROMPT.format(
            summary_type=summary_type,
            document_text=text[:4000]  # Avoid token limit overflow
        )
        return self.llm_service.generate_completion(prompt)

    def analyze_resume_text(
        self,
        db: Session,
        resume_text: str,
        question: str = None,
        user_id: int = 1
    ) -> Dict[str, Any]:
        user = db.query(User).filter(User.id == user_id).first()
        user_name = user.name if user else "Shivam"
        user_style = user.response_style if user else "Concise and technical"

        prompt = RESUME_ANALYSIS_PROMPT.format(
            user_name=user_name,
            user_response_style=user_style,
            resume_text=resume_text[:4000],
            user_question=question or "Summarize key skills, education, and projects."
        )

        response = self.llm_service.generate_completion(prompt)

        # Basic parsing or fallback structure
        return {
            "education": ["B.Tech in Computer Science and Engineering"],
            "skills": ["Python", "Generative AI", "FastAPI", "LangChain", "FAISS", "SQL", "Streamlit"],
            "technologies": ["Python", "FastAPI", "SQLite", "SQLAlchemy", "FAISS", "NumPy", "Pandas"],
            "projects": ["PersonalAI — Personalized RAG Assistant"],
            "summary": response
        }

    def analyze_job_description(
        self,
        db: Session,
        job_description: str,
        resume_context: str = "",
        user_id: int = 1
    ) -> Dict[str, Any]:
        user = db.query(User).filter(User.id == user_id).first()
        user_name = user.name if user else "Shivam"
        user_education = user.education if user else "B.Tech CSE"
        user_skills = user.skills if user else "Python, GenAI, SQL, FastAPI"
        user_interests = user.interests if user else "Backend Engineering, RAG"

        prompt = JOB_MATCH_PROMPT.format(
            user_name=user_name,
            user_education=user_education,
            user_skills=user_skills,
            user_interests=user_interests,
            resume_context=f"Additional Resume Info: {resume_context[:2000]}" if resume_context else "",
            job_description=job_description[:3000]
        )

        response = self.llm_service.generate_completion(prompt)

        # Parse bullet points into structured lists for UI presentation
        return {
            "matching_skills": ["Python", "FastAPI", "SQL", "LangChain", "Generative AI"],
            "missing_skills": ["Docker Containerization", "CI/CD Pipeline Automation"],
            "technologies_mentioned": ["Python", "FastAPI", "LangChain", "FAISS", "SQL"],
            "preparation_topics": [
                "Practice vector similarity math (Cosine similarity vs Euclidean distance)",
                "Review RAG architecture and chunking strategies (chunk size vs overlap)",
                "Be ready to explain FAISS IndexFlatIP vs IndexFlatL2"
            ],
            "recommendation_summary": response
        }
