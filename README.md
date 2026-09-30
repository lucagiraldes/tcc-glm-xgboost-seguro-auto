# Modelagem de prêmio puro em seguro de automóvel com GLM e XGBoost

Rotinas utilizadas no Trabalho de Conclusão de Curso do MBA em Data Science e Analytics (USP/Esalq).

- **Autor:** Luca D'Angelo Giraldes
- **Orientador:** Prof. Dr. Henrique Raymundo Gioia
- **Ano:** 2026

## Objetivo

Comparar o desempenho de Modelos Lineares Generalizados (distribuições Poisson e Gamma) com o algoritmo XGBoost na modelagem de frequência, severidade e prêmio puro em seguro de automóvel, considerando tanto a performance preditiva quanto a interpretabilidade dos modelos.

## Dados

Base pública **freMTPL2**, com apólices de seguro de automóvel do mercado francês, obtida via `sklearn.datasets.fetch_openml`:

| Base | Registros | Conteúdo |
|---|---|---|
| `freMTPL2freq` | 678.013 | exposição, características do veículo e do condutor, número de sinistros |
| `freMTPL2sev` | 26.639 | valores dos sinistros pagos |

Os dados são baixados automaticamente na primeira execução do notebook, portanto não são versionados neste repositório.

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `modelagem_glm_xgboost.ipynb` | Notebook com todas as rotinas: leitura dos dados, análise exploratória, GLM Poisson (frequência), GLM Gamma (severidade), modelos XGBoost, métricas de avaliação e importância de variáveis |
| `requirements.txt` | Versões das bibliotecas utilizadas |

## Etapas do notebook

1. Leitura das bases `freMTPL2freq` e `freMTPL2sev`.
2. Análise exploratória e verificação de valores ausentes.
3. Pré-processamento: dummies para os GLMs e Label Encoding (um único mapeamento para treino e teste) para o XGBoost.
4. Divisão treino/teste (80%/20%, `random_state = 42`) das apólices; os sinistros de cada apólice acompanham a divisão, de modo que nenhuma apólice de teste é usada no ajuste da severidade.
5. GLM Poisson para frequência, com `log(Exposure)` como offset, inclusive na predição.
6. GLM Gamma para severidade, restrito aos registros com sinistro.
7. XGBoost com objetivos `count:poisson` (alvo: frequência anualizada, ponderada pela exposição) e `reg:gamma` — 200 estimadores, profundidade 4, taxa de aprendizado 0,05.
8. Avaliação no conjunto de teste: MSE, MAE e deviance média (Poisson e Gamma) para frequência e severidade, razão predito/observado na frequência, e MSE e MAE do prêmio puro por apólice.
9. Importância de variáveis do XGBoost.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook modelagem_glm_xgboost.ipynb
```

## Ambiente utilizado

Python 3.13.5, com as versões listadas em `requirements.txt`.
