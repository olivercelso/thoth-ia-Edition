import base64
from langchain_openai import ChatOpenAI
# --- CORREÇÃO: Importando do lugar certo (langchain_core) ---
from langchain_core.messages import HumanMessage
from app.config import settings

def encode_image(image_bytes: bytes) -> str:
    """Converte bytes da imagem para string Base64."""
    return base64.b64encode(image_bytes).decode('utf-8')

async def analyze_image(image_bytes: bytes, prompt_text: str = "Descreva esta imagem em detalhes.") -> str:
    """Envia a imagem para o GPT-4o analisar."""
    try:
        # 1. Preparar a imagem em Base64
        base64_image = encode_image(image_bytes)

        # 2. Configurar o Modelo de Visão (GPT-4o)
        # Importante: O modelo 'vision' deve ser o gpt-4o definido no config
        llm = ChatOpenAI(
            model=settings.vision_llm_model, 
            openai_api_key=settings.openai_api_key,
            max_tokens=1000
        )

        # 3. Montar a Mensagem Multimodal (Texto + Imagem)
        # O LangChain moderno exige esta estrutura específica para imagens
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ]
        )

        # 4. Enviar para a IA
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return f"Erro na análise de visão: {str(e)}"