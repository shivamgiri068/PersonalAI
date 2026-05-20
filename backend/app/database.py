import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

# Ensure data directory exists
data_dir = os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", ""))
if data_dir and not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for obtaining database sessions in API endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables and create a default user profile if none exists."""
    from backend.app.models.database_models import User
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == 1).first()
        if not user:
            default_user = User(
                id=1,
                name="Shivam",
                education="B.Tech CSE",
                skills="Python, SQL, Generative AI, Machine Learning, FastApi, React",
                interests="RAG, LLM Fine-tuning, Backend Engineering, Vector Databases",
                response_style="Concise, technical, and structured with clear code examples"
            )
            db.add(default_user)
            db.commit()
    finally:
        db.close()
