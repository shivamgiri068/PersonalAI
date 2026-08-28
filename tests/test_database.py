import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models.database_models import User, Document, Conversation, Message

def test_database_models_crud():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    try:
        # Create User
        user = User(
            id=1,
            name="Shivam",
            education="B.Tech CSE",
            skills="Python, SQL, RAG",
            interests="GenAI",
            response_style="Concise"
        )
        db.add(user)
        db.commit()

        fetched_user = db.query(User).filter(User.id == 1).first()
        assert fetched_user.name == "Shivam"

        # Create Document
        doc = Document(
            filename="test.pdf",
            file_type="PDF",
            file_path="/data/test.pdf",
            chunk_count=3,
            char_count=1000,
            word_count=150,
            token_count=200
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        assert doc.id is not None
        assert doc.chunk_count == 3

        # Create Conversation & Message
        conv = Conversation(title="Test Chat")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        msg = Message(
            conversation_id=conv.id,
            role="user",
            content="Hello AI"
        )
        db.add(msg)
        db.commit()

        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello AI"
    finally:
        db.close()
