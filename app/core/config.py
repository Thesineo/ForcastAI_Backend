from pydantic_settings import BaseSettings
from pydantic import Field
import os 


class Settings(BaseSettings):
    # Alpha Vantage API (Primary)
    ALPHA_VANTAGE_API_KEY: str = ""
    ""
    
    # Other existing settings...
    DATABASE_URL: str | None = None
    REDIS_URL: str | None = None
    JWT_SECRET: str = "change-me"  # default; overridden by env in prod
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    class Config:
        env_file = ".env"

settings = Settings()


