from pydantic_settings import BaseSettings
from pydantic import Field
import os 


class Settings(BaseSettings):
    # Alpha Vantage API (Primary)
    ALPHA_VANTAGE_API_KEY: str = ""
    ""
    
    # Other existing settings...
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/"
    
    class Config:
        env_file = ".env"

settings = Settings()


