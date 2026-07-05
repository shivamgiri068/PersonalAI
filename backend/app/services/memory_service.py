import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.app.models.database_models import Conversation, Message
from backend.app.models.domain_schemas import SourceCitation

class MemoryService:
    @staticmethod
    def create_conversation(db: Session, title: str = "New Conversation") -> Conversation:
        conversation = Conversation(title=title)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    @staticmethod
    def get_conversations(db: Session) -> List[Conversation]:
        return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()

    @staticmethod
    def get_conversation(db: Session, conversation_id: int) -> Optional[Conversation]:
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()

    @staticmethod
    def delete_conversation(db: Session, conversation_id: int) -> bool:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            db.delete(conversation)
            db.commit()
            return True
        return False

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: int,
        role: str,
        content: str,
        sources: List[Dict[str, Any]] = None
    ) -> Message:
        sources_json = json.dumps(sources) if sources else None
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources_json=sources_json
        )
        db.add(message)
        
        # Update conversation title if it's the first message
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            if conversation.title == "New Conversation" and role == "user":
                conversation.title = content[:30] + ("..." if len(content) > 30 else "")
            db.add(conversation)

        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_recent_chat_history(db: Session, conversation_id: int, limit: int = 6) -> List[Dict[str, str]]:
        """
        Retrieves the last `limit` messages for context window inclusion.
        Prevents sending unlimited chat history to the LLM.
        """
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.desc()).limit(limit).all()
        
        # Reverse to chronological order
        messages.reverse()
        
        return [{"role": msg.role, "content": msg.content} for msg in messages]
