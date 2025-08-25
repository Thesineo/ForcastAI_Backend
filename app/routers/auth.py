from datetime import datetime, timedelta
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt  # PyJWT

from app.core.config import settings
from app.db.db import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserOut

router = APIRouter(prefix="/api/auth", tags=["Auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = settings.JWT_ALGORITHM

def create_token(data: Dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=ALGORITHM)

@router.post("/signup", response_model=UserOut)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    # Ensure email format is valid (EmailStr in schema does this)
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = pwd_context.hash(payload.password)
    user = User(username=payload.username, email=EmailStr(payload.email), hashed_password=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token({"sub": str(user.id)})
    return UserOut(id=user.id, username=user.username, email=user.email, token=token)

@router.post("/signin", response_model=UserOut)
def signin(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not pwd_context.verify(payload.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_token({"sub": str(user.id)})
    return UserOut(id=user.id, username=user.username, email=user.email, token=token)
