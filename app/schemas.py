"""Contratos de dados da API (Pydantic): o que a API aceita e o que devolve."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Um pixel: inteiro de 0 (preto) a 255 (branco)
Pixel = Annotated[int, Field(ge=0, le=255)]


class PredictRequest(BaseModel):
    # strict: recusa "12" (texto) ou 12.5 (decimal) em vez de convertê-los
    # forbid: recusa campos não previstos no contrato
    model_config = ConfigDict(strict=True, extra="forbid")

    pixels: list[Pixel] = Field(
        ...,
        min_length=784,
        max_length=784,
        description="Imagem 28x28 achatada em 784 pixels, cada um com intensidade inteira de 0 a 255.",
    )


class PredictResponse(BaseModel):
    digito: int = Field(..., ge=0, le=9, description="Dígito previsto pelo modelo.")
    confianca: float = Field(..., ge=0.0, le=1.0, description="Probabilidade atribuída ao dígito previsto.")
    model_version: str = Field(..., description="Versão do artefato que gerou a predição.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado operacional do serviço.")
    model_loaded: bool = Field(..., description="Indica se o modelo está carregado em memória RAM.")
    model_version: str | None = Field(None, description="Versão do modelo em memória, se houver.")