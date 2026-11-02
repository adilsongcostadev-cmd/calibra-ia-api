"""Camada de rotas: endpoints HTTP agrupados em um APIRouter."""
import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services import model_service

logger = logging.getLogger("uvicorn.error")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoramento"],
    summary="Verifica se o serviço está ativo e se o modelo está em memória",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.carregado,
        model_version=model_service.metadata.get("model_version") if model_service.carregado else None,
    )


@router.post(
    "/v1/predict",
    response_model=PredictResponse,
    tags=["Inferência"],
    summary="Classifica uma imagem 28x28 de dígito manuscrito",
    responses={
        400: {"description": "Imagem sem conteúdo (todos os pixels com o mesmo valor)"},
        422: {"description": "Payload fora do contrato (tipo, faixa ou tamanho inválidos)"},
        500: {"description": "Erro interno inesperado durante a inferência"},
        503: {"description": "Modelo indisponível em memória"},
    },
)
def predict(payload: PredictRequest) -> PredictResponse:
    # 503: o serviço está no ar, mas o modelo não foi carregado no lifespan
    if not model_service.carregado:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo indisponível. Verifique o /health.")

    # 400: tipo válido (o Pydantic aceitou), mas sem sentido para o negócio
    if min(payload.pixels) == max(payload.pixels):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Imagem uniforme, sem traço de dígito.")

    # 500: qualquer falha inesperada é registrada no log e devolvida de forma controlada
    try:
        digito, confianca = model_service.prever(payload.pixels)
    except Exception:
        logger.exception("Erro durante a inferência.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Erro interno ao processar a predição.")

    return PredictResponse(
        digito=digito,
        confianca=round(confianca, 4),
        model_version=model_service.metadata["model_version"],
    )