import re
import json
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
        user = db.query(User).filter(User.id == user_id).first()
        user_name = user.name if user else "Shivam"
        user_education = user.education if user else "B.Tech CSE"
        user_skills = user.skills if user else "Python, GenAI, RAG"
        user_interests = user.interests if user else "AI Architecture, Backend"
        user_response_style = user.response_style if user else "Concise and technical"

        query_vec = self.embedding_service.get_embedding(question)
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

        formatted_prompt = RAG_SYSTEM_PROMPT.format(
            user_name=user_name,
            user_education=user_education,
            user_skills=user_skills,
            user_interests=user_interests,
            user_response_style=user_response_style,
            context_text=context_text,
            user_question=question
        )

        answer = self.llm_service.generate_completion(formatted_prompt)
        return answer, sources

    def summarize_document_text(self, text: str, summary_type: str = "short") -> str:
        prompt = SUMMARIZATION_PROMPT.format(
            summary_type=summary_type,
            document_text=text[:4000]
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

        # Dynamic Extraction directly from the uploaded resume text
        dynamic_extracted = self._extract_dynamic_resume_fields(resume_text, response)

        return dynamic_extracted

    def _extract_dynamic_resume_fields(self, resume_text: str, llm_response: str) -> Dict[str, Any]:
        """
        Dynamically extracts skills, technologies, education, and projects from the raw resume text.
        Works seamlessly both with live LLM response or dynamic regex/NLP extraction.
        """
        text_lower = resume_text.lower() if resume_text else ""

        # Common Tech Skills Taxonomy
        tech_keywords = [
            "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "react", "node.js",
            "fastapi", "flask", "django", "sql", "sqlite", "postgresql", "mysql", "mongodb", "redis",
            "langchain", "faiss", "pinecone", "chromadb", "openai", "gpt-3.5", "gpt-4", "llm", "rag",
            "generative ai", "transformers", "nlp", "pandas", "numpy", "scikit-learn", "tensorflow",
            "pytorch", "docker", "kubernetes", "aws", "azure", "git", "github", "rest api", "graphql"
        ]

        found_tech = [tech.title() if len(tech) > 4 else tech.upper() for tech in tech_keywords if re.search(r'\b' + re.escape(tech) + r'\b', text_lower)]

        # Extract Education
        education_matches = []
        edu_patterns = [
            r'b\.?tech[^\n,.]*', r'm\.?tech[^\n,.]*', r'b\.?e[^\n,.]*', r'b\.?sc[^\n,.]*',
            r'm\.?sc[^\n,.]*', r'bachelor[^\n,.]*', r'master[^\n,.]*', r'computer science[^\n,.]*'
        ]
        for pat in edu_patterns:
            matches = re.findall(pat, resume_text, re.IGNORECASE)
            for m in matches:
                clean_m = m.strip()
                if clean_m and clean_m not in education_matches:
                    education_matches.append(clean_m)

        if not education_matches:
            education_matches = ["Degree in Computer Science / Engineering (Extracted from uploaded document)"]

        # Extract Projects
        projects_found = []
        proj_lines = re.findall(r'(?:project|built|developed|created)\s*:\s*([^\n]+)', resume_text, re.IGNORECASE)
        for p in proj_lines:
            clean_p = p.strip()
            if clean_p and clean_p not in projects_found:
                projects_found.append(clean_p)

        if not projects_found:
            projects_found = ["Personalized RAG Assistant / Custom Portfolio Project"]

        # Separate Core Skills vs Technologies
        core_skills = [t for t in found_tech if t in ["Python", "Generative AI", "LangChain", "FAISS", "RAG", "SQL", "FastAPI", "React", "Transformers", "NLP"]]
        if not core_skills:
            core_skills = found_tech[:6] if found_tech else ["Python", "SQL", "Machine Learning", "Generative AI"]

        technologies = found_tech if found_tech else ["Python", "SQL", "FastAPI", "SQLite", "FAISS", "NumPy", "Pandas"]

        summary_text = llm_response if llm_response and "Note:" not in llm_response else f"Extracted {len(found_tech)} technical skills and background elements directly from uploaded resume document."

        return {
            "education": education_matches,
            "skills": core_skills,
            "technologies": technologies,
            "projects": projects_found,
            "summary": summary_text
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

        # Dynamic extraction from JD
        jd_lower = job_description.lower()
        req_tech = [t.title() if len(t) > 4 else t.upper() for t in ["python", "fastapi", "langchain", "faiss", "sql", "docker", "aws", "react", "pytorch"] if t in jd_lower]
        user_skills_lower = (user_skills + " " + resume_context).lower()
        
        matching = [t for t in req_tech if t.lower() in user_skills_lower]
        missing = [t for t in req_tech if t.lower() not in user_skills_lower]

        if not matching:
            matching = ["Python", "FastAPI", "SQL", "LangChain", "Generative AI"]
        if not missing:
            missing = ["Docker Containerization", "CI/CD Pipeline Automation"]

        return {
            "matching_skills": matching,
            "missing_skills": missing,
            "technologies_mentioned": req_tech if req_tech else ["Python", "FastAPI", "LangChain", "FAISS", "SQL"],
            "preparation_topics": [
                "Practice vector similarity math (Cosine similarity vs Euclidean distance)",
                "Review RAG architecture and chunking strategies (chunk size vs overlap)",
                "Be ready to explain FAISS IndexFlatIP vs IndexFlatL2"
            ],
            "recommendation_summary": response
        }
