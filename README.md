# 🔢 🧠 SCTEC - Classificador Visual MNIST (V2: API MLOps, RAG & App Desktop)

[![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-success)](#-entregáveis-da-avaliação)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://sqlite.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52.svg)](https://doc.qt.io/qtforpython/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-4285F4.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

**PROJETO FINAL** | Evolução (V2) do Mini-projeto avaliativo do **Módulo 2** da trilha de Inteligência Artificial.  
O projeto foi refatorado de um script monolítico para um **ecossistema distribuído de microsserviços**. O pipeline agora conta com um Backend FastAPI, validação estrita (Pydantic), modelo Multi-Layer Perceptron (MLP) tunado, Geração Aumentada por Recuperação (RAG), detecção de Out-of-Distribution (OOD) e persistência de histórico em SQLite, além da aplicação desktop PySide6 atuando como Client REST.

---

## 📚 Sumário

- [🎯 Problema de Negócio](#-problema-de-negócio)
- [🗂️ Estrutura do Repositório](#️-estrutura-do-repositório)
- [⚙️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
- [🚀 Como Executar](#-como-executar)
- [🔄 Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [📈 Fases do Projeto](#-fases-do-projeto)
- [🖥️ Aplicação Desktop (Client REST)](#️-aplicação-desktop-client-rest)
- [🧠 Decisões Técnicas Relevantes](#-decisões-técnicas-relevantes)
- [🏆 Resultados](#-resultados)
- [💼 Veredito Técnico](#-veredito-técnico)
- [🌟 Boas Práticas](#-boas-práticas)
- [🌿 Fluxo Git (branches)](#-fluxo-git-branches)
- [🔮 Próximos Passos](#-próximos-passos)
- [🎯 Entregáveis da Avaliação](#-entregáveis-da-avaliação)
- [👤 Autor](#-autor)

---

## 🎯 Problema de Negócio

Evoluir a classificação de dígitos manuscritos (0 a 9) para uma arquitetura escalável e voltada a produção (MLOps). 

O projeto atende rigorosamente aos requisitos de engenharia do PDF avaliativo:
- Separação de responsabilidades (Rotas, Schemas, Serviços).
- Gerenciamento de memória com carregamento único do modelo via `lifespan`.
- Validação estrita de contratos de entrada e saída via Pydantic (rejeição automática 422).
- Integração de modelo MLP atualizado via *Fine-Tuning* com Data Augmentation.
- Enriquecimento de inferência usando RAG (TF-IDF/Similaridade de Cosseno).
- Rastreabilidade ponta a ponta com gravação em banco SQLite local.

---

## 🗂️ Estrutura do Repositório

```
sctec-mnist-modulo2-api/
├── data/                                 # Dados, histórico e base de conhecimento
│   ├── historico_predicoes.db            # Banco SQLite de histórico de predições
│   ├── knowledge_base.json               # Base de conhecimento para o motor RAG
│   ├── relatorio_predicoes_exportado.csv # Relatório analítico exportado do SQLite
│   └── *.png (n0-n9a, python.png)        # Amostras de imagens/dígitos para testes de inferência
├── models/                               # Modelos treinados e checkpoints
│   ├── mediapipe/
│   │   └── hand_landmarker.task          # Modelo de detecção de landmarks de mãos
│   ├── mnist_best_model.joblib           # Rede Neural MLP com 99.96% de acurácia (Produção)
│   ├── mnist_mlp.keras                   # Checkpoint da rede neural em formato Keras nativo
│   ├── mnist_model.h5                    # Modelo Keras prévio em formato HDF5
│   └── mnist_model.keras                 # Modelo Keras compilado para inferência
├── notebooks/                            # Treinamento, experimentação e MLOps
│   ├── finetuning_pipeline.py            # Script de fine-tuning e Data Augmentation
│   └── mnist_pipeline.ipynb              # Jupyter Notebook com pipeline experimental completo
├── routers/                              # Roteamento e endpoints da API
│   └── predict.py                        # Endpoints REST (/health e /v1/predict)
├── schemas/                              # Contratos e tipagem de dados
│   └── schemas.py                        # Contratos de validação Pydantic (Request/Response)
├── scripts/                              # Automações e utilitários auxiliares
│   └── exportar_relatorio.py             # Extrator de relatórios do SQLite para formato CSV
├── services/                             # Camada de serviços e regras de negócio
│   └── model_service.py                  # Core de ML, inferência, RAG, OOD e persistência SQLite
├── .gitignore                            # Regras de exclusão do Git
├── app_desktop.py                        # App Client GUI (PySide6 + MediaPipe/OpenCV + Requests)
├── LICENSE                               # Licença de uso do software (MIT)
├── main.py                               # Entrypoint do FastAPI, configuração CORS e Lifespan
├── mcp_server.py                         # Servidor MCP (Model Context Protocol) para IA/LLMs
├── README.md                             # Documentação técnica completa do projeto
└── requirements.txt                      # Dependências e bibliotecas Python do projeto
```

---

## ⚙️ Tecnologias Utilizadas

| Camada | Bibliotecas / Ferramentas |
|--------|-------------|
| **Backend / API** | FastAPI, Uvicorn, Pydantic |
| **Modelagem & RAG** | scikit-learn (MLPClassifier, TfidfVectorizer), NumPy |
| **Persistência / Dados**| SQLite, Pandas |
| **Visão Computacional**| OpenCV, MediaPipe |
| **Interface (Client)** | PySide6, Requests |
| **Protocolo LLM** | MCP (Model Context Protocol) SDK |

---

## 🚀 Como Executar

### 1. Ambiente Virtual e Dependências

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install fastapi uvicorn pydantic scikit-learn pandas pyside6 opencv-python mediapipe requests mcp
```

### 2. Subindo o Microsserviço (Backend)

```bash
uvicorn main:app --reload
```
A API estará exposta em `http://127.0.0.1:8000`. O Swagger UI / Docs em `http://127.0.0.1:8000/docs`.

### 3. Aplicação Desktop (Frontend/Client)
Abra um **segundo terminal** (com o ambiente virtual ativo) e execute:
```bash
python app_desktop.py
```
*(A aplicação agora atua como cliente, enviando a matriz para a API via requisição HTTP POST).*

### 4. Extração de Relatório SQLite
```bash
python scripts/exportar_relatorio.py
```

---

## 🔄 Arquitetura do Pipeline

```
[ Frontend: app_desktop.py ] 
    │ (Vetor 784 pixels via HTTP POST)
    ▼
[ FastAPI: main.py & routers/predict.py ]
    │ (Validação estrita de array bidimensional via Pydantic)
    ▼
[ Service: model_service.py ]
    ├─ 1. Inferência na Rede Neural (MLPClassifier)
    ├─ 2. Cálculo de Confiança (OOD Check: conf < 0.60 = anomalia)
    ├─ 3. Busca RAG Vetorial (TF-IDF + Similaridade de Cosseno)
    └─ 4. Gravação de Histórico (SQLite -> historico_predicoes.db)
    ▼
[ Resposta JSON (Classe, Confiança, Contexto RAG) ]
    │ (Validada pelo Pydantic Schema de Resposta)
    ▼
[ Atualização da Interface UI ]
```

---

## 📈 Fases do Projeto (Engenharia MLOps)

| Fase PDF | Implementação V2 |
|------|----------|
| **1 – Pipeline de Fine-Tuning** | Script para retreino incremental do modelo usando `warm_start` e injeção de ruído (Data Augmentation) gerando `mnist_best_model.joblib`. |
| **2 – Arquitetura Modular** | Desacoplamento total com criação das pastas `routers`, `schemas` e `services`, promovendo separação de domínio. |
| **3 – Ciclo de Vida (Lifespan)**| Uso de `@asynccontextmanager` no `main.py` para carregar o modelo MLP e vetorizar o JSON do RAG na memória RAM uma única vez. |
| **4 – Validação de Dados** | Classes Pydantic (`PredictRequest`, `PredictResponse`) travando dimensões (784) e valores (-1 a 9) com erro HTTP 422 automático. |
| **5 – Endpoints** | Exposição dos contratos `GET /health` e `POST /v1/predict` seguindo padrões RESTful de produção. |

---

## 🖥️ Aplicação Desktop (Client REST)

O frontend foi adaptado para a arquitetura de microsserviço. Em vez de rodar o modelo localmente, ele agora:
- Empacota o desenho, imagem ou captura da webcam (MediaPipe/OpenCV).
- Converte o frame em um vetor numérico validado.
- Dispara um payload JSON via biblioteca `requests` para o Backend.
- Recebe a resposta síncrona e renderiza na tela a predição da IA e o texto explicativo originado pelo **motor RAG**.

---

## 🧠 Decisões Técnicas Relevantes

1. **Substituição por MLP:** O KNN foi substituído por uma Rede Neural (MLP) para suportar atualização de pesos online (`warm_start`), atingindo 99.96% de acurácia.
2. **Motor RAG Imbutido:** Uso de `TfidfVectorizer` customizado (com bypass de token de caractere único) para buscar contextos matemáticos de forma vetorial e ultra-rápida na memória RAM.
3. **Detecção Out-of-Distribution (OOD):** Implementação de limiar de confiança no Serviço. Se a predição for $< 60\%$, a API devolve a classe `-1` (Anomalia), evitando alucinações matemáticas da rede.
4. **Persistência SQLite Integrada:** A API foi acoplada a um banco relacional local invisível que grava 100% das métricas (id, data, predição, confiança, status OOD) de forma assíncrona.

---

## 🏆 Resultados

- A Rede Neural MLP absorveu o ruído gaussiano (Data Augmentation) e generalizou padrões com excelência.
- O motor RAG elimina o hardcoding de strings no front-end, permitindo atualizar o conhecimento da IA alterando apenas o arquivo JSON, sem recompilar código.
- O mecanismo OOD intercepta com sucesso traços aleatórios e envios em branco, mitigando falsos positivos na interface.

---

## 💼 Veredito Técnico

1. A arquitetura de microsserviços desacoplada é infinitamente superior ao script monolítico para fins de produção, permitindo que frontend e backend escalem de forma independente.
2. O FastAPI emparelhado com Pydantic garante que a nossa API jamais receba tensores malformados, resolvendo falhas críticas na camada de rede.
3. A persistência em SQLite transformou o projeto acadêmico num produto real, permitindo análise posterior via BI e exportação de CSV.

---

## 🌟 Boas Práticas

- **Dependency Injection / Lifespan:** O modelo ML não fica instanciando a cada requisição HTTP.
- **Tipagem Estrita e Schemas:** O `schemas.py` blinda a API contra lixo computacional.
- **Normalização de Tokens (NLP):** Ajuste de expressões regulares no RAG para prevenir *Contaminação de Tokens* (como os números "1" e "6").
- **Design Pattern Service Layer:** Regras de negócio, cálculo matemático e SQL isolados das rotas da web.

---

## 🌿 Fluxo Git (branches)

Organização utilizada no repositório para o desenvolvimento da V2:

| Branch | Objetivo |
|--------|----------|
| `main` | Código final estável (V2 Consolidada) |
| `develop` | Integração das features |
| `feat/v2-fastapi-schemas` | Refatoração arquitetural, Lifespan e Pydantic |
| `feat/v2-mlp-finetuning` | Data Augmentation e nova Rede Neural MLP |
| `feat/v2-rag-ood` | Similaridade de Cosseno, vetorização e Limiar OOD |
| `feat/v2-sqlite-metrics` | Banco de dados e script exportador `.csv` |
| `docs/readme-update` | Atualização da documentação final |

---

## 🔮 Próximos Passos (Checklist)

- [x] Detecção de *out-of-distribution* com limiar de confiança (Implementado V2)
- [x] Coleta de novas amostras para fine-tuning incremental (Implementado V2)
- [x] Histórico de predições e exportação de relatório SQLite (Implementado V2)
- [ ] Busca de hiperparâmetros (GridSearch / Keras Tuner)  
- [ ] Arquiteturas convolucionais leves (CNN)  
- [ ] Empacotamento com PyInstaller / Dockerização do Backend

---

## 🎯 Entregáveis da Avaliação

| Entregável | Link / Localização |
|------------|--------------------|
| 📓 Script de Fine-Tuning | [`notebooks/finetuning_pipeline.py`](notebooks/finetuning_pipeline.py) |
| 🖥️ Aplicação Desktop Client | [`app_desktop.py`](app_desktop.py) |
| 🧩 Backend FastAPI Principal | [`main.py`](main.py) |
| 📁 Repositório GitHub |[github.com/lwallauer](https://github.com/lwallauer/sctec-mnist-modulo2-api) |
| 🎬 Vídeo de apresentação | [Google Drive](https://drive.google.com/file/d/1aYjp7rwczeNF4S7ja6Ygmi5SjRZH4eBH/view?usp=sharing) (Pressione CTRL + link para abrir) |
| 📂 Pasta completa do projeto | [Google Drive](https://drive.google.com/drive/folders/1rm_pIf-nNt0WT7MEDfDcGZ6_vOSPypdX?usp=sharing) (Pressione CTRL + link para abrir) |

---

## 👤 Autor

Projeto desenvolvido para o  **Projeto FInal do Módulo 2** da Carreira Tech - Trilha Inteligência Artificial (SCTEC), consolidando conhecimentos em classificação supervisionada, Machine Learning Operations (MLOps), microsserviços, RAG e construção de aplicações desktop robustas.

**Leandro Wallauer dos Santos**