import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from backend.app.models.database_models import User
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == 1).first()
        if not user:
            default_user = User(
                id=1,
                name="SHIVAM GIRI",
                education="B.Tech - Computer Science & Engineering, Pranveer Singh Institute of Technology, Kanpur (2022 - 2026, CGPA: 7.2)",
                skills="Java, JavaScript, SQL, Node.js, Express.js, HTML, CSS, MySQL, MongoDB, Git, GitHub, Linux, JWT, WebSockets, REST APIs, Python, FastAPI, LangChain, FAISS, Generative AI, RAG",
                interests="Generative AI, Full Stack Development, Backend Engineering, Data Structures & Algorithms, Vector Databases",
                response_style="Concise, technical, and structured with clear code examples"
            )
            db.add(default_user)
            db.commit()
    finally:
        db.close()
