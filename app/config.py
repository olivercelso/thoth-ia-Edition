from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Alteramos de 'str' para 'Optional[str] = None'
    # Isso impede o erro de validação se a chave não estiver no ambiente do sistema
    openai_api_key: Optional[str] = None
    
    # Outras configurações (se houver)
    model_name: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"

settings = Settings()