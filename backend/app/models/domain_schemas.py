from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# User Profile Schemas
class UserProfileBase(BaseModel):
    name: str = Field(..., example="Shivam")
    education: Optional[str] = Field(None, example="B.Tech CSE")
    skills: Optional[str] = Field(None, example="Python, SQL, Generative AI, React")
    interests: Optional[str] = Field(None, example="RAG, LLM Fine-tuning, Backend Engineering")
    response_style: Optional[str] = Field("Concise, technical, and structured", example="Concise and technical")

class UserProfileUpdate(UserProfileBase):
    pass

class UserProfileResponse(UserProfileBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Document Schemas
class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_path: str
    upload_date: datetime
    chunk_count: int
    char_count: int
    word_count: int
    token_count: int

    class Config:
        from_attributes = True

class DocumentStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_characters: int
    total_words: int
    total_tokens: int
    average_chunk_length: float
    file_type_distribution: dict

class DocumentSummaryRequest(BaseModel):
    document_id: int
    summary_type: str = Field("short", description="short | detailed | key_points")

class DocumentSummaryResponse(BaseModel):
    document_id: int
    filename: str
    summary_type: str
    summary: str

# Source Citation Schema
class SourceCitation(BaseModel):
    document_name: str
    chunk_id: int
    page_number: Optional[int] = None
    content: str

# Chat Schemas
class ChatMessageRequest(BaseModel):
    conversation_id: Optional[int] = None
    message: str = Field(..., example="What skills are listed in my resume?")
    use_rag: bool = Field(True, description="Whether to perform RAG vector retrieval")

class ChatMessageResponse(BaseModel):
    conversation_id: int
    role: str
    content: str
    sources: List[SourceCitation] = []
    created_at: datetime

class MessageDetailResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: List[SourceCitation] = []
    created_at: datetime

class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConversationDetailResponse(ConversationResponse):
    messages: List[MessageDetailResponse] = []

# Resume Analysis Schemas
class ResumeAnalyzeRequest(BaseModel):
    document_id: Optional[int] = None
    resume_text: Optional[str] = None
    question: Optional[str] = Field("Summarize key skills, education, and projects", description="Optional question")

class ResumeAnalyzeResponse(BaseModel):
    education: List[str] = []
    skills: List[str] = []
    technologies: List[str] = []
    projects: List[str] = []
    summary: str

# Job Analysis Schemas
class JobAnalyzeRequest(BaseModel):
    job_description: str = Field(..., example="We are looking for a Python GenAI Engineer skilled in LangChain, FAISS, FastAPI, SQL...")
    resume_document_id: Optional[int] = None

class JobAnalyzeResponse(BaseModel):
    matching_skills: List[str] = []
    missing_skills: List[str] = []
    technologies_mentioned: List[str] = []
    preparation_topics: List[str] = []
    recommendation_summary: str
