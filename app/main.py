from fastapi import FastAPI
from app.routers import analyze, chat
from app.db.db import engine, Base 
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title= "Predective AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # or ["http://localhost:5173"] if using Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(chat.router, prefix="/v1", tags=["chat"])


@app.get("/health")
def health():
    return {"ok": True}
Base.metadata.create_all(bind=engine)




