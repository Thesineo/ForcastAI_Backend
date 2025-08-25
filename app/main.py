from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.db import Base, engine
from app.routers import chat as chat_router
from app.routers import auth as auth_router

app = FastAPI()

# CORS (adjust for your frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://your-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
if engine is not None:
    Base.metadata.create_all(bind=engine)

# Routers
app.include_router(chat_router.router, prefix="/api")
app.include_router(auth_router.router)  # already prefixed with /api/auth

@app.get("/health")
def health():
    return {"ok": True}
