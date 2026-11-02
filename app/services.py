"""Camada de serviço: carrega o modelo e executa a inferência (sem dependência de HTTP)."""
import json
from pathlib import Path

import joblib
import numpy as np

# Caminho calculado a partir deste arquivo: funciona de qualquer pasta em que o servidor for iniciado
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "models" / "mlp_mnist_v1.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"


class ModelService:
    def __init__(self) -> None:
        self.modelo = None
        self.metadata: dict = {}

    def carregar(self) -> None:
        """Lê o artefato do disco. Deve ser chamado UMA única vez, no lifespan."""
        self.modelo = joblib.load(MODELO_PATH)
        self.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    @property
    def carregado(self) -> bool:
        return self.modelo is not None

    def prever(self, pixels: list[int]) -> tuple[int, float]:
        """Recebe 784 pixels brutos (0-255) e devolve (dígito, confiança)."""
        if not self.carregado:
            raise RuntimeError("Modelo não está carregado em memória.")
        X = np.asarray(pixels, dtype=np.float32).reshape(1, -1)
        proba = self.modelo.predict_proba(X)[0]
        indice = int(np.argmax(proba))
        return int(self.modelo.classes_[indice]), float(proba[indice])


# Instância única compartilhada pela aplicação
model_service = ModelService()