from fastapi import APIRouter, HTTPException
from schemas.schemas import MnistRequest, MnistResponse
from services.model_service import ml_service

router = APIRouter(prefix="/v1", tags=["Inferência Inteligente"])

@router.post("/predict", response_model=MnistResponse, status_code=200)
async def predict_digit(request: MnistRequest):
    try:
        resultado = ml_service.predict_with_rag(request.features)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {str(e)}")