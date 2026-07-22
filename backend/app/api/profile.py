from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models.database_models import User
from backend.app.models.domain_schemas import UserProfileResponse, UserProfileUpdate

router = APIRouter(prefix="/profile", tags=["User Profile"])

@router.get("", response_model=UserProfileResponse)
def get_user_profile(db: Session = Depends(get_db)):
    """Retrieve candidate profile stored in SQLite."""
    user = db.query(User).filter(User.id == 1).first()
    if not user:
        # Create default profile
        user = User(
            id=1,
            name="Shivam",
            education="B.Tech CSE",
            skills="Python, SQL, Generative AI, React, FastAPI",
            interests="RAG, LLMs, Backend Engineering",
            response_style="Concise, technical, and structured"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

@router.put("", response_model=UserProfileResponse)
def update_user_profile(profile_data: UserProfileUpdate, db: Session = Depends(get_db)):
    """Update candidate profile attributes in SQLite."""
    user = db.query(User).filter(User.id == 1).first()
    if not user:
        user = User(id=1)
        db.add(user)

    user.name = profile_data.name
    user.education = profile_data.education
    user.skills = profile_data.skills
    user.interests = profile_data.interests
    user.response_style = profile_data.response_style

    db.commit()
    db.refresh(user)
    return user
