from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class MnistRequest(BaseModel):
    """Contrato de entrada: estritamente 784 pixels normalizados."""
    features: List[float] = Field(
        ...,
        min_length=784,
        max_length=784,
        description="Array 1D com 784 valores float (0.0 a 1.0) representando a imagem 28x28 achatada."
    )

    @field_validator('features')
    @classmethod
    def check_normalization(cls, v: List[float]) -> List[float]:
        if any(pixel < 0.0 or pixel > 1.0 for pixel in v):
            raise ValueError("Todos os pixels devem estar normalizados entre 0.0 e 1.0.")
        return v

class MnistResponse(BaseModel):
    """Contrato de saída: preparado para o enriquecimento RAG."""
    classe_predita: int = Field(..., ge=-1, le=9, description="Dígito previsto (0-9).")
    confianca: float = Field(..., ge=0.0, le=1.0, description="Probabilidade da predição.")
    contexto_extra: Optional[str] = Field(
        default=None, 
        description="Informação contextual adicionada via base de conhecimento local (RAG)."
    )