from pydantic import BaseModel
from typing import Dict

class ChatRequest(BaseModel):
    ticker: str
    question: str

class ChatResponse(BaseModel):
    ticker: str
    question: str
    reply: str
    analytics: Dict[str, str]

