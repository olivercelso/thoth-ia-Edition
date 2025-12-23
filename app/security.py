from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

# Define que a chave deve vir no cabeçalho "X-API-KEY"
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    """Valida se a chave enviada bate com a chave do administrador."""
    if api_key_header == settings.admin_api_key:
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas ou ausentes (X-API-KEY incorreta)"
    )