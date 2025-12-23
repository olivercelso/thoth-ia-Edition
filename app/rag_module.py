import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# --- CORREÇÃO AQUI: Usando langchain_core ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings

# Configuração do Banco Vetorial (Persistente)
CHROMA_PATH = "chroma_db"

def get_vectorstore():
    """Retorna a instância do banco vetorial ChromaDB."""
    # Garante que a chave da OpenAI está definida
    if not settings.openai_api_key:
        raise ValueError("OpenAI API Key não encontrada no .env")
        
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    return Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

async def ingest_pdf(file_path: str) -> str:
    """Lê um PDF, quebra em pedaços e salva no banco vetorial."""
    try:
        # 1. Carregar o PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 2. Quebrar em pedaços (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # 3. Salvar no Banco Vetorial
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        
        return f"Sucesso: {len(chunks)} fragmentos indexados."
    except Exception as e:
        return f"Erro na ingestão: {str(e)}"

async def ask_rag(query: str) -> str:
    """Busca contexto no banco e responde a pergunta."""
    try:
        vectorstore = get_vectorstore()
        
        # 1. Buscar documentos relevantes (Retrieval)
        results = vectorstore.similarity_search(query, k=3)
        
        # Se não achou nada, avisa
        if not results:
             return "Não encontrei informações nos documentos para responder a isso."
             
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # 2. Gerar resposta com LLM (Generation)
        # Usamos o modelo definido na config (gpt-4o-mini para economia)
        llm = ChatOpenAI(
            model_name=settings.default_llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0
        )
        
        messages = [
            SystemMessage(content="Você é um assistente útil. Responda à pergunta baseando-se APENAS no contexto fornecido abaixo."),
            HumanMessage(content=f"Contexto:\n{context_text}\n\nPergunta: {query}")
        ]
        
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"Erro no RAG: {str(e)}"