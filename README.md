# calibra-ia-api

API REST para **classificação de dígitos manuscritos**, que disponibiliza como microsserviço um modelo de Machine Learning treinado sobre o dataset MNIST.

## 1. Problema de negócio

Sistemas de digitalização de documentos (formulários, cheques, protocolos) precisam converter dígitos escritos à mão em dados estruturados. Esta API recebe uma imagem 28×28 em escala de cinza e devolve o dígito reconhecido com a respectiva confiança, pronta para ser consumida por qualquer aplicação externa via HTTP.

## 2. Origem do modelo

**Opção A:** reaproveitamento do mini-projeto [calibra-ia](https://github.com/adilsongcostadev-cmd/calibra-ia), no qual foram comparados três modelos (Regressão Logística, Random Forest e MLP). O **MLP** obteve o melhor desempenho e foi retreinado neste projeto dentro de um `Pipeline` do scikit-learn, com divisão estratificada treino/validação/teste (70/15/15).

| Conjunto | Acurácia | F1 ponderado | ROC-AUC (OvR) |
|---|---|---|---|
| Validação | 0,9806 | 0,9806 | 0,9996 |
| **Teste** | **0,9813** | **0,9813** | **0,9996** |

O treinamento, as justificativas técnicas e o relatório completo de métricas estão em [`notebooks/fase1_modelo.ipynb`](notebooks/fase1_modelo.ipynb).

## 3. Tecnologias utilizadas

| Tecnologia | Versão | Papel |
|---|---|---|
| Python | 3.14.5 | Linguagem |
| scikit-learn | 1.9.1 | Pipeline (`MinMaxScaler` + `MLPClassifier`) |
| NumPy | 2.5.4 | Manipulação dos pixels |
| joblib | 1.6.0 | Serialização do artefato |
| FastAPI | 0.162.0 | Framework da API |
| Pydantic | 2.14.2 | Validação estrita dos contratos de dados |
| Uvicorn | ver `requirements.txt` | Servidor ASGI |

## 4. Arquitetura

```
calibra-ia-api/
├── app/
│   ├── main.py        # ponto de entrada: lifespan e inclusão dos roteadores
│   ├── routers.py     # endpoints HTTP (APIRouter) e tratamento de erros
│   ├── schemas.py     # contratos de entrada e saída (Pydantic)
│   └── services.py    # carga do modelo e inferência, sem dependência de HTTP
├── models/
│   ├── mlp_mnist_v1.joblib   # pipeline serializado (2,7 MB)
│   └── metadata.json         # versão, métricas e versões das bibliotecas
├── notebooks/
│   └── fase1_modelo.ipynb    # treino, avaliação e serialização
├── scripts/
│   └── gerar_payloads.py     # gera os JSONs de teste
├── data/
│   ├── payloads/             # exemplos para os cenários 200, 400 e 422
│   └── imagens_proprias/     # dígitos manuscritos do autor (projeto anterior)
└── requirements.txt
```

**Decisões principais:**
- **Camadas desacopladas:** as rotas não conhecem o modelo; apenas chamam `services.py`.
- **Lifespan:** o modelo é carregado **uma única vez** na inicialização (`@asynccontextmanager`) e mantido em RAM. Nenhuma requisição lê o disco.
- **Pré-processamento dentro do Pipeline:** a API recebe pixels brutos (0 a 255), e o escalonamento é aplicado pelo próprio artefato, sem risco de divergência entre treino e produção.
- **Validação estrita:** `strict=True` impede que texto seja convertido em número silenciosamente.

## 5. Instalação e execução

```bash
# 1. Clonar o repositório
git clone https://github.com/adilsongcostadev-cmd/calibra-ia-api.git
cd calibra-ia-api

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # Linux / macOS

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Iniciar o servidor
uvicorn app.main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000`. O modelo já está serializado em `models/`, então **não é necessário retreinar** para executar a API.

## 6. Documentação da API

Documentação interativa (Swagger): **http://127.0.0.1:8000/docs**

### `GET /health`

Verifica se o serviço está ativo e se o modelo está em memória.

```json
{ "status": "ok", "model_loaded": true, "model_version": "v1" }
```

### `POST /v1/predict`

Recebe uma imagem 28×28 achatada em **exatamente 784 inteiros de 0 a 255**.

**Requisição** (exemplo abreviado; o arquivo completo está em `data/payloads/valido_digito_7.json`):
```json
{ "pixels": [0, 0, 0, 0, 0, 84, 185, 159, 151, 60, 36, 0, "... 784 valores no total"] }
```

**Resposta 200:**
```json
{ "digito": 7, "confianca": 1.0, "model_version": "v1" }
```

**Teste pelo terminal com um payload pronto:**
```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d @data/payloads/valido_digito_7.json
```

### Códigos de resposta

| Código | Quando ocorre | Payload de exemplo |
|---|---|---|
| **200** | Predição realizada com sucesso | `valido_digito_7.json` |
| **400** | Imagem uniforme, sem traço de dígito | `erro_400_imagem_vazia.json` |
| **422** | Pixel fora de 0 a 255, texto em campo numérico ou quantidade diferente de 784 | `erro_422_fora_da_faixa.json`, `erro_422_texto.json` |
| **500** | Falha inesperada na inferência (registrada em log) | — |
| **503** | Modelo indisponível em memória | — |

## 7. Melhorias futuras

- Endpoint que aceite o upload direto de uma imagem (PNG/JPG), com o pré-processamento para 28×28 feito no servidor.
- Testes automatizados com `pytest` e `TestClient`.
- Empacotamento em contêiner Docker.
- Registro e versionamento do modelo com MLflow.

## 8. Evidências

🎥 **Vídeo de apresentação:** _link do Google Drive_