"""Gera payloads JSON de exemplo para testar o endpoint POST /v1/predict."""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_FILE = BASE_DIR / "data" / "mnist" / "mnist_784.npz"
SAIDA = BASE_DIR / "data" / "payloads"
SAIDA.mkdir(parents=True, exist_ok=True)

dados = np.load(CACHE_FILE)  # cache criado pelo notebook da Fase 1
X, y = dados["X"], dados["y"]

# Uma imagem real de um dígito 7, convertida para inteiros de 0 a 255
indice = int(np.where(y == 7)[0][-1])
pixels = X[indice].astype(int).tolist()

payloads = {
    "valido_digito_7.json": {"pixels": pixels},
    "erro_400_imagem_vazia.json": {"pixels": [0] * 784},
    "erro_422_fora_da_faixa.json": {"pixels": [300] + pixels[1:]},
    "erro_422_texto.json": {"pixels": ["abc"] + pixels[1:]},
}

for nome, conteudo in payloads.items():
    (SAIDA / nome).write_text(json.dumps(conteudo, separators=(",", ":")), encoding="utf-8")
    print(f"Gerado: data/payloads/{nome}")