import streamlit as st
import asyncio
import os

# --- CORREÇÃO OBRIGATÓRIA PARA STREAMLIT CLOUD (LINUX) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------------------------------------

# Importando a lógica DIRETAMENTE (Sem API/Requests)
from app.config import settings
from app.rag_module import ingest_pdf, ask_rag
from app.vision_module import analyze_image
from app.excel_engine import fill_excel_from_pdf

# Configuração da Página
st.set_page_config(page_title="Thoth IA Cloud", layout="wide", page_icon="☁️")
st.title("☁️ Thoth IA - Versão Cloud")

# Validação de Segurança (Chave OpenAI)
# No Streamlit Cloud, as senhas vêm de st.secrets, não do .env
if not st.secrets.get("OPENAI_API_KEY"):
    st.error("⚠️ A chave da OpenAI não foi configurada nos Secrets do Streamlit.")
    st.stop()
else:
    # Forçar a configuração da chave para o sistema interno
    settings.openai_api_key = st.secrets["OPENAI_API_KEY"]

# Menu
menu = st.sidebar.radio("Módulos:", ["💬 Chat (RAG)", "👁️ Visão", "📊 Excel"])

# --- MÓDULO 1: CHAT ---
if menu == "💬 Chat (RAG)":
    st.header("Chat com Documentos")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.sidebar.button("Processar"):
        with st.spinner("Indexando..."):
            # Salvar temporariamente
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Chamar função Async
            result = asyncio.run(ingest_pdf("temp.pdf"))
            st.sidebar.success(result)
            os.remove("temp.pdf")

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Pergunte algo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = asyncio.run(ask_rag(prompt))
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- MÓDULO 2: VISÃO ---
elif menu == "👁️ Visão":
    st.header("Análise Visual")
    img = st.file_uploader("Imagem", type=["jpg", "png"])
    prompt = st.text_input("Prompt", "O que há nesta imagem?")
    
    if img and st.button("Ver"):
        st.image(img)
        with st.spinner("Analisando..."):
            res = asyncio.run(analyze_image(img.getvalue(), prompt))
            st.write(res)

# --- MÓDULO 3: EXCEL ---
elif menu == "📊 Excel":
    st.header("Automação Excel")
    pdf = st.file_uploader("PDF Dados", type="pdf")
    xlsx = st.file_uploader("Excel Modelo", type="xlsx")
    
    if st.button("Gerar") and pdf and xlsx:
        with st.spinner("Processando..."):
            # Salvar temps
            with open("temp_input.pdf", "wb") as f: f.write(pdf.getbuffer())
            with open("temp_template.xlsx", "wb") as f: f.write(xlsx.getbuffer())
            
            # Processar
            output = fill_excel_from_pdf("temp_input.pdf", "temp_template.xlsx")
            
            with open(output, "rb") as f:
                st.download_button("Baixar", f, "Relatorio.xlsx")