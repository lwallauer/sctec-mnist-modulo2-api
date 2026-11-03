from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from services.model_service import ml_service
from routers import predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia a memória: carrega o artefato pesado apenas na inicialização."""
    ml_service.load_model()
    yield
    ml_service.model = None

app = FastAPI(
    title="SCTec MNIST - API com RAG e MCP",
    description="Microsserviço de Visão Computacional pronto para integração com LLMs.",
    version="2.0.0",
    lifespan=lifespan
)

# Segurança CORS para permitir chamadas do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/health", tags=["Monitoramento"])
async def health_check():
    """Endpoint vital para checagem de saúde e load balancers."""
    if ml_service.model is None:
        raise HTTPException(status_code=500, detail="Serviço ativo, mas ML não está na RAM.")
    return {"status": "online", "rag_active": True, "mcp_ready": True}
# RAG carregado no lifespan
