import streamlit as st
import requests

# Configurações
API_URL = "http://127.0.0.1:8000"
# IMPORTANTE: A chave abaixo deve ser igual à ADMIN_API_KEY do seu arquivo .env
API_KEY = "minha-senha-super-secreta" 
HEADERS = {"X-API-KEY": API_KEY}

st.set_page_config(page_title="Thoth IA", layout="wide", page_icon="🦉")
st.title("🦉 Thoth IA - Sistema Integrado")

# Menu Lateral
menu = st.sidebar.radio("Escolha o Módulo:", ["💬 Chat com Documentos", "👁️ Visão Computacional", "📊 Automação Excel"])

# --- 1. CHAT (RAG) ---
if menu == "💬 Chat com Documentos":
    st.header("Chat Inteligente (RAG)")
    
    # Upload
    uploaded_file = st.sidebar.file_uploader("Adicionar PDF ao Conhecimento", type="pdf")
    if uploaded_file and st.sidebar.button("Processar Arquivo"):
        with st.spinner("Lendo e indexando..."):
            files = {"file": uploaded_file}
            try:
                res = requests.post(f"{API_URL}/upload-document", files=files, headers=HEADERS)
                if res.status_code == 200:
                    st.sidebar.success("Sucesso! O arquivo agora faz parte da memória.")
                else:
                    st.sidebar.error(f"Erro: {res.text}")
            except Exception as e:
                st.sidebar.error(f"Erro de conexão: {e}")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Pergunte algo sobre seus documentos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    payload = {"message": prompt}
                    res = requests.post(f"{API_URL}/chat", json=payload, headers=HEADERS)
                    if res.status_code == 200:
                        ans = res.json().get("response", "Erro na resposta")
                        st.write(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                    else:
                        st.error(f"Erro API: {res.text}")
                except Exception as e:
                    st.error(f"Erro de conexão: {e}")

# --- 2. VISÃO ---
elif menu == "👁️ Visão Computacional":
    st.header("Análise de Imagens (GPT-4o)")
    img = st.file_uploader("Envie uma imagem técnica", type=["jpg", "png", "jpeg"])
    prompt_img = st.text_input("O que você quer saber sobre a imagem?", "Descreva esta imagem em detalhes técnicos.")
    
    if img and st.button("Analisar"):
        st.image(img, caption="Imagem Carregada", use_container_width=True)
        with st.spinner("Analisando pixels..."):
            files = {"file": img.getvalue()}
            params = {"prompt": prompt_img}
            try:
                res = requests.post(f"{API_URL}/vision", files={"file": img}, params=params, headers=HEADERS)
                if res.status_code == 200:
                    st.success("Análise Concluída:")
                    st.write(res.json().get("analysis"))
                else:
                    st.error(f"Erro: {res.text}")
            except Exception as e:
                st.error(f"Erro: {e}")

# --- 3. EXCEL ---
elif menu == "📊 Automação Excel":
    st.header("Preenchimento Automático (PDF -> Excel)")
    st.info("O sistema vai ler o PDF e preencher as colunas do Excel Template automaticamente.")
    
    col1, col2 = st.columns(2)
    pdf = col1.file_uploader("1. PDF Fonte (Dados)", type="pdf")
    xlsx = col2.file_uploader("2. Excel Template (Modelo Vazio)", type="xlsx")
    
    if st.button("Gerar Planilha") and pdf and xlsx:
        with st.spinner("A IA está trabalhando..."):
            files = {
                "pdf_file": pdf,
                "excel_template": xlsx
            }
            try:
                res = requests.post(f"{API_URL}/fill-excel", files=files, headers=HEADERS)
                if res.status_code == 200:
                    st.success("Pronto!")
                    st.download_button("📥 Baixar Planilha Preenchida", res.content, "Relatorio_Final.xlsx")
                else:
                    st.error(f"Erro no processamento: {res.text}")
            except Exception as e:
                st.error(f"Erro: {e}")