# Task 1 — Diagnóstico, baselines e ARIMA/SARIMA

**Curso:** Séries temporais · FGV EMAp · 2026/2  
**Prazo:** 30/09/2026, entrega **online** (repo do grupo). Não há encontro presencial nessa data.  
**Peso:** **25% da A2**. A Task 1 **não entra na A1**. A prova A1 é presencial e individual; este trabalho é em grupo (~5–6 grupos na turma).  
**Compute:** cabe em laptop. Nada exige nuvem.

## O que vocês estão prevendo

Três séries **diárias** da mesma loja Walmart (recorte M5 `CA_1`), já agregadas — não é o desafio Kaggle inteiro:

| Coluna | Série |
|--------|--------|
| `store_total` | total da loja |
| `FOODS` | categoria alimentos |
| `HOBBIES` | categoria hobbies |

Arquivos em `dados/`:

- `treino.csv` — 2011-01-29 a 2016-03-27 (com y)
- `validacao.csv` — 2016-03-28 a 2016-04-24 (com y; **horizonte da Task 1**)
- `holdout_datas.csv` — 2016-04-25 a 2016-05-22 (**só datas**, sem y)

O y do holdout **não** faz parte desta entrega. Ele entra na A2 (parcela de 45%). Usar alvo futuro, embaralhar linhas ou treinar com a validação como se fosse treino zera o critério de split.

## O que entregar

No repo do grupo, na raiz:

1. Código executável (notebook e/ou scripts) **na raiz do repo** (`*.py` ou `*.ipynb`; o check não olha só `src/`) que, a partir de `dados/`, reproduz diagnóstico, previsões e métricas.
2. `previsoes_validacao.csv` — colunas: `date`, `series`, `modelo`, `yhat`  
   `series` ∈ {`store_total`, `FOODS`, `HOBBIES`}  
   `modelo` inclui pelo menos: `media`, `naive`, `naive_sazonal`, `drift`, `sarima` (ou `arima`).
3. `metricas.csv` — colunas: `series`, `modelo`, `mae`, `rmse`, `mase`  
   Métricas **só** no horizonte de validação (28 dias).
4. Figuras de diagnóstico (série no tempo + ACF + PACF) para as três séries, no treino.
5. `AI_USAGE.md` no formato abaixo.

Um comando documentado no `README.md` do repo deve reproduzir `metricas.csv` (por exemplo `python -m pip install -r requirements.txt && python run.py`).

## Conteúdo obrigatório

**Diagnóstico.** Para cada série: gráfico da série, ACF e PACF no **treino**. Comentário curto: tendência, sazonalidade semanal (m = 7), outliers (zeros de calendário existem).

**Quatro baselines da aula 20/08**, no mesmo split, para as três séries, horizonte de 28 dias:

- média  
- naive (último valor)  
- naive sazonal com período **7**  
- drift  

**ARIMA/SARIMA.** Identificar ordem com ACF/PACF (e diferenciação se precisar), ajustar **só no treino**, prever os 28 dias da validação. Covariáveis (promoção, SNAP, preço, feriado como regressor) **não** são exigência desta Task: SARIMA sem X fecha o critério. **Não há ponto extra** por usar regressor — o peso do ARIMA/SARIMA continua 2,0.

**Métricas.** MAE, RMSE e MASE na validação, para cada série e cada modelo. O MASE usa como escala o MAE in-sample do naive sazonal semanal no treino (período 7).

O modelo mais complexo tem de ser comparado com os quatro baselines. Se perder para a média ou para o naive sazonal, o texto precisa dizer isso — não esconder.

## Nota (10 pontos, crédito parcial)

A nota da Task 1 é a **soma** dos critérios. Falhar em um zera **só** aquele peso.

| Critério | Pontos | Como entra no check |
|----------|--------|---------------------|
| Split temporal sem vazamento | 2,5 | treino só até 2016-03-27; previsões da Task 1 só em 2016-03-28–2016-04-24; nenhuma linha do holdout com y |
| MAE, RMSE e MASE na validação conhecida | 2,0 | as três métricas, nas três séries, no horizonte de 28 dias |
| ARIMA/SARIMA identificado e ajustado | 2,0 | modelo `arima` ou `sarima` em `previsoes_validacao.csv` e `metricas.csv`, com ordem justificada no texto/notebook |
| Quatro baselines (média, naive, naive sazonal, drift) | 1,5 | os quatro nomes em `modelo`, no mesmo split |
| Repo executável | 1,0 | o comando do README reproduz `metricas.csv` |
| Diagnóstico entregue (série + ACF/PACF) | 0,5 | figuras (ou células) para as três séries |
| `AI_USAGE.md` no formato | 0,5 | arquivo existe, não vazio, com os cinco itens abaixo |
| **Total** | **10** | |

O script público `checks/check_task1.py` (nesta pasta) é o mesmo critério que o professor usa na parte machine-checkable. Rode antes do prazo. Não há janela informal de “conserta depois”.

## `AI_USAGE.md` (obrigatório)

IA generativa **pode** ser usada. Sem este arquivo a nota máxima desta Task não fecha (0,5 ponto e, na A2, a política de auditoria do semestre).

Conteúdo mínimo (~1–2 páginas):

1. Ferramentas usadas (ou a frase **não usamos IA**).
2. Onde ajudou (tarefas / partes do pipeline).
3. **2–5 prompts representativos** (não o histórico completo).
4. **1–3 erros da IA que o grupo corrigiu** (ex.: leakage, métrica errada, uso de alvo futuro).
5. Responsabilidade: o grupo responde pelo que entregou.

## Fora desta entrega

- y do holdout e score cego da A2  
- treino em nuvem  
- covariáveis obrigatórias no SARIMA  
- leaderboard Kaggle  

## Dúvidas de recorte

Os dados já vêm cortados. Não baixem o M5 completo nem misturem outras lojas. Qualquer dúvida de vazamento: se a informação não existiria em 2016-03-27, ela não entra no treino.
