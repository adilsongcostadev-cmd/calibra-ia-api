"""Ponto de entrada da aplicação: cria o app e inclui os roteadores."""
from fastapi import FastAPI

from app.routers import router

app = FastAPI(
    title="calibra-ia-api",
    description="API REST para classificação de dígitos manuscritos (MNIST) com um Pipeline MLP do scikit-learn.",
    version="1.0.0",
)

app.include_router(router)