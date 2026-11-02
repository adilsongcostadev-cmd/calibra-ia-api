"""Ponto de entrada da aplicação: ciclo de vida (lifespan) e inclusão dos roteadores."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import router
from app.services import model_service

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: executa UMA única vez, antes de a API aceitar requisições ---
    try:
        model_service.carregar()
        logger.info("Modelo %s carregado em memória.", model_service.metadata.get("model_version"))
    except Exception:
        # A API sobe mesmo assim: o /health informará que o modelo não está disponível
        logger.exception("Falha ao carregar o modelo na inicialização.")
    yield
    # --- Shutdown: libera o modelo ao desligar o servidor ---
    model_service.modelo = None
    logger.info("Modelo liberado da memória.")


app = FastAPI(
    title="calibra-ia-api",
    description="API REST para classificação de dígitos manuscritos (MNIST) com um Pipeline MLP do scikit-learn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)