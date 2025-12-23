from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    openai_api_key: str
    admin_api_key: str
    allowed_origins: List[str] = ["http://localhost:8000"]
    default_llm: str = "gpt-4o-mini"
    vision_llm: str = "gpt-4o"

    class Config:
        env_file = ".env"

settings = Settings()
