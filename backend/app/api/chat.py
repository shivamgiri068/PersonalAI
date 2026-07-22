import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models.domain_schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationResponse,
    ConversationDetailResponse,
    MessageDetailResponse,
    SourceCitation
)
from backend.app.services.memory_service import MemoryService
from backend.app.services.rag_service import RAGService
from backend.app.services.vector_store import FAISSVectorStore

router = APIRouter(prefix="/chats", tags=["Chat"])
vector_store = FAISSVectorStore()
rag_service = RAGService(vector_store=vector_store)

@router.post("", response_model=ChatMessageResponse)
def chat_with_assistant(
    req: ChatMessageRequest,
    db: Session = Depends(get_db)
):
    """
    Main RAG Chat endpoint. Accepts query, retrieves FAISS context, applies user profile personalization,
    calls OpenAI LLM, and persists user & assistant messages in SQLite chat history.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    # Get or create conversation session
    if req.conversation_id:
        conversation = MemoryService.get_conversation(db, req.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation session not found")
        conversation_id = conversation.id
    else:
        new_conv = MemoryService.create_conversation(db, title="New Conversation")
        conversation_id = new_conv.id

    # Add user message to SQLite history
    MemoryService.add_message(db, conversation_id=conversation_id, role="user", content=req.message)

    # Perform RAG retrieval and answer generation
    if req.use_rag:
        answer, sources = rag_service.query_rag(db, question=req.message)
    else:
        # Direct LLM call without document retrieval
        answer = rag_service.llm_service.generate_completion(req.message)
        sources = []

    # Format source citations
    formatted_sources = [
        SourceCitation(
            document_name=s["document_name"],
            chunk_id=s["chunk_id"],
            page_number=s.get("page_number"),
            content=s["content"]
        )
        for s in sources
    ]

    sources_dict = [s.dict() for s in formatted_sources]

    # Save assistant message to SQLite
    assistant_msg = MemoryService.add_message(
        db,
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=sources_dict
    )

    return ChatMessageResponse(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=formatted_sources,
        created_at=assistant_msg.created_at
    )

@router.get("", response_model=List[ConversationResponse])
def get_conversations(db: Session = Depends(get_db)):
    """Retrieve list of past conversation sessions."""
    return MemoryService.get_conversations(db)

@router.get("/{chat_id}", response_model=ConversationDetailResponse)
def get_conversation_detail(chat_id: int, db: Session = Depends(get_db)):
    """Get complete conversation session details and message history."""
    conv = MemoryService.get_conversation(db, chat_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation session not found")

    messages_detail = []
    for m in conv.messages:
        sources_list = []
        if m.sources_json:
            try:
                raw = json.loads(m.sources_json)
                sources_list = [SourceCitation(**item) for item in raw]
            except Exception:
                pass
        
        messages_detail.append(
            MessageDetailResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=sources_list,
                created_at=m.created_at
            )
        )

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages_detail
    )

@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(chat_id: int, db: Session = Depends(get_db)):
    """Delete a conversation session and all its messages."""
    success = MemoryService.delete_conversation(db, chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    return None
