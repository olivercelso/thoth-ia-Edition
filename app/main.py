from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import shutil
import os

# Imports internos
from app.config import settings
from app.security import verify_api_key
from app.rag_module import ingest_pdf, ask_rag
from app.vision_module import analyze_image
from app.excel_engine import fill_excel_from_pdf

# Configuração do Limitador (Segurança)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="NeuralManager API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuração de CORS (Quem pode acessar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de Dados
class ChatMessage(BaseModel):
    message: str

# --- ROTAS ---

@app.get("/")
def health_check():
    return {"status": "online", "system": "NeuralManager"}

@app.post("/upload-document")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,  # <--- ADICIONADO PARA O SLOWAPI
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Recebe um PDF e salva no cérebro da IA."""
    try:
        # Salvar arquivo temporariamente
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Processar
        result = await ingest_pdf(temp_filename)
        
        # Limpar
        os.remove(temp_filename)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,  # <--- ADICIONADO PARA O SLOWAPI
    body: ChatMessage,
    api_key: str = Depends(verify_api_key)
):
    """Conversa com a base de conhecimento (RAG)."""
    response = await ask_rag(body.message)
    return {"response": response}

@app.post("/vision")
@limiter.limit("10/minute")
async def vision_analysis(
    request: Request,  # <--- ADICIONADO PARA O SLOWAPI
    file: UploadFile = File(...),
    prompt: str = "Descreva esta imagem em detalhes técnicos.",
    api_key: str = Depends(verify_api_key)
):
    """Analisa imagens usando GPT-4o."""
    content = await file.read()
    result = await analyze_image(content, prompt)
    return {"analysis": result}

@app.post("/fill-excel")
@limiter.limit("5/minute")
async def automation_excel(
    request: Request,  # <--- ADICIONADO PARA O SLOWAPI
    pdf_file: UploadFile = File(...),
    excel_template: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Preenche Excel baseado em PDF."""
    try:
        # Salvar arquivos temporários
        pdf_path = f"temp_{pdf_file.filename}"
        xlsx_path = f"temp_{excel_template.filename}"
        
        with open(pdf_path, "wb") as f:
            f.write(await pdf_file.read())
        with open(xlsx_path, "wb") as f:
            f.write(await excel_template.read())
            
        # Executar Motor
        output_path = fill_excel_from_pdf(pdf_path, xlsx_path)
        
        # Limpeza (Opcional, cuidado para não apagar o output antes de enviar)
        # os.remove(pdf_path)
        # os.remove(xlsx_path)
        
        return FileResponse(
            output_path, 
            filename="Relatorio_Preenchido.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))