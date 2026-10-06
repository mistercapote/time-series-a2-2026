# Task 1 — Diagnóstico, Baselines e Modelagem SARIMA

Este repositório contém a entrega da **Task 1** (25% da avaliação A2) da disciplina de Séries Temporais. O projeto contempla análise exploratória e diagnóstica, implementação de quatro modelos *baseline* em horizonte de 28 dias e a geração automatizada das previsões e métricas de desempenho para três séries temporais diárias da loja Walmart (`store_total`, `FOODS` e `HOBBIES`).


### Modelos Baseline (Horizonte de 28 dias)

- Média (media): Projeção constante da média histórica de treino.

- Naive (naive): Repetição do último valor observado no treino ($y_T$).

- Naive Sazonal (naive_sazonal): Repetição do ciclo sazonal semanal ($m = 7$).

- Drift (drift): Variação linear calculada entre o primeiro e o último ponto de treino.

### Métricas de Avaliação

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MASE (Mean Absolute Scaled Error): Escalonado pelo MAE in-sample do Naive Sazonal semanal ($m = 7$) no treino.

---

## 1. Instalação e Ambiente
Recomenda-se o uso de um ambiente virtual (venv ou conda) com Python 3.10+.

A partir da raiz do repositório, instale todas as dependências necessárias:

```bash
python -m pip install -r requirements.txt
```


Reprodução Automatizada das Métricas (metricas.csv)
Para recalcular os erros e regerar o arquivo `metricas.csv`, execute:
```bash
python run.py
```
Ou o comando integrado:
```bash
python -m pip install -r requirements.txt && python run.py
```

## 2. Análise Interativa
O fluxo completo de modelagem, diagnósticos de autocorrelação (ACF/PACF) e inspeção visual das previsões pode ser executado diretamente no Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```


### 3. Verificação da Submissão (Checks)
Para rodar os testes parciais de conformidade fornecidos no pacote da disciplina, execute a partir da raiz:

```bash
python checks/check_task1.py --submission .
```
