"""
Application Configuration
"""

from pydantic_settings import BaseSettings  # ← Changed this line
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    #Alpha Vantage
    ALPHA_VANTAGE_API_KEY: str | None = None
    # Database
    DATABASE_URL: str | None = None
    
    # Redis
    REDIS_URL: str | None = None
    
    # JWT
    JWT_SECRET: str = "change-me"  # default; overridden by env in prod
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    
    # OpenAI (NEW)
    OPENAI_API_KEY: str | None = None
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
