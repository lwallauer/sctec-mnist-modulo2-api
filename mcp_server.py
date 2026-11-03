import httpx
from mcp.server.fastmcp import FastMCP

# Inicializa o servidor MCP com o nome da nossa ferramenta
mcp = FastMCP("SCTec-MNIST-RAG")

@mcp.tool()
async def prever_digito_manuscrito(features: list[float]) -> str:
    """
    Analisa uma matriz de 784 pixels (imagem 28x28 achatada) de um dígito manuscrito usando Inteligência Artificial.
    A ferramenta consome uma API REST local e retorna o dígito identificado, a confiança estatística e o contexto adicional (RAG).
    
    Args:
        features: Lista exata de 784 valores numéricos (float entre 0.0 e 1.0).
    """
    if len(features) != 784:
        return "Erro MCP: A matriz de entrada deve conter exatamente 784 características (pixels)."
    
    try:
        # Comunica com a nossa FastAPI desenvolvida na Fase 5 do projeto
        async with httpx.AsyncClient() as client:
            resposta = await client.post(
                "http://127.0.0.1:8000/v1/predict",
                json={"features": features},
                timeout=10.0
            )
            
            if resposta.status_code == 200:
                dados = resposta.json()
                return (
                    f"🎯 Dígito Identificado: {dados['classe_predita']}\n"
                    f"📊 Confiança da IA: {dados['confianca'] * 100:.2f}%\n"
                    f"🧠 Contexto RAG: {dados['contexto_extra']}"
                )
            elif resposta.status_code == 422:
                return "Erro de Validação (422): O esquema de dados foi rejeitado pelo Pydantic da API."
            else:
                return f"Erro Interno da API ({resposta.status_code}): {resposta.text}"
                
    except Exception as e:
        return f"Falha de Conexão: A API FastAPI não está acessível. Detalhes: {str(e)}"

if __name__ == "__main__":
    print("🚀 Servidor MCP (SCTec-MNIST-RAG) inicializado via stdio.")
    mcp.run()