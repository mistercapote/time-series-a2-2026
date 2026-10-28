# Task 2 — Features e três classes

**Curso:** Séries temporais · FGV EMAp · 2026/2  
**Prazo:** entrega **online** (repo do grupo), no fim de cada domingo (horário de Brasília). 

| Domingo | O que se acrescenta |
|---------|--------------------|
| 01/11/2026 | `linear` nas três séries, com MAE, RMSE e MASE (1,5) |
| 08/11/2026 | `boosting` ou `arvore` nas três séries, com MAE, RMSE e MASE (1,5) |
| 15/11/2026 | `rede` nas três séries, com MAE, RMSE e MASE (1,5) |

Cada entrega acrescenta essa classe. Não se refaz a Task 1, nem a classe cujo domingo já passou. Uma classe que só aparece depois do domingo dela não recupera os 1,5. O split, o naive sazonal, o SARIMA e o repo já vieram na Task 1: o check só confirma que continuam no arquivo. O 2,0 de MAE, RMSE e MASE e o 0,5 do `AI_USAGE.md` fecham em 15/11, lendo as linhas que cada domingo já acrescentou. Não é um segundo cálculo das classes anteriores. A seção nova do `AI_USAGE.md` cresce; a da Task 1 não se reescreve.

## O que vocês estão prevendo

As mesmas três séries diárias da Task 1, loja Walmart `CA_1`:

| Coluna | Série |
|--------|--------|
| `store_total` | total da loja |
| `FOODS` | categoria alimentos |
| `HOBBIES` | categoria hobbies |

Os arquivos estão em `dados/`, nesta pasta:

- `treino.csv` — 2011-01-29 a 2016-03-27 (com y)
- `validacao.csv` — 2016-03-28 a 2016-04-24 (com y; **horizonte da Task 2**)
- `holdout_datas.csv` — 2016-04-25 a 2016-05-22 (**só datas**, sem y)
- `covariaveis.csv` — as mesmas datas, 2011-01-29 a 2016-05-22

As três classes usam as colunas de `covariaveis.csv`. O texto de cada classe diz quais entraram.

O SNAP é o programa federal americano que deposita um crédito mensal para compra de comida. `snap_CA` vale 1 no dia em que as lojas da Califórnia aceitam esse pagamento, e 0 nos outros.

Os eventos trazem nome e tipo, até dois no mesmo dia: `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`. Na maioria dos dias ficam vazios.

`preco_FOODS` e `preco_HOBBIES` são a média da lista da semana Walmart, só daquela categoria, repetida nos sete dias, sem peso pela quantidade vendida. `preco_store_total` é a mesma média na loja inteira, inclusive HOUSEHOLD. As três séries de venda contam unidades, então não geram esses preços. Promoção não existe no M5.

Evento, SNAP e o preço da semana já se conhecem na origem. O preço que entra no ajuste usa média e desvio só do treino. Continua de fora qualquer covariável que não está nesse CSV.

A origem é o domingo **2016-03-27**. O ajuste usa só o treino. O y da validação serve para calcular MAE, RMSE e MASE. O y do holdout continua fora desta entrega.

Usar alvo futuro, embaralhar linhas, padronizar com a validação ou escolher hiperparâmetro pelo erro destes 28 dias zera o critério de split no espírito da Task 1: a validação publica; não escolhe.

## O que entregar

O formato é o da Task 1. O mesmo repositório do grupo serve. A Task 1 não se reabre: código, figuras, ordem do SARIMA, baselines e as linhas já notadas permanecem. Esta Task acrescenta linhas e código. O `AI_USAGE.md` pode ganhar uma seção desta Task; a seção da Task 1 fica como foi entregue.

No repo do grupo, na raiz:

1. Código executável (notebook e/ou scripts) **na raiz do repo** (`*.py` ou `*.ipynb`; o check não olha só `src/`) que, a partir de `dados/`, reproduz previsões e métricas.
2. `previsoes_validacao.csv` — colunas: `date`, `series`, `modelo`, `yhat`  
   `series` ∈ {`store_total`, `FOODS`, `HOBBIES`}  
   `modelo` inclui, nas três séries: `naive_sazonal`, `sarima` (ou `arima`), `linear`, `boosting` **ou** `arvore`, e `rede` (alias aceitos: `lstm`, `rnn`, `transformer`).
3. `metricas.csv` — colunas: `series`, `modelo`, `mae`, `rmse`, `mase`  
   Métricas **só** no horizonte de validação (28 dias).
4. `AI_USAGE.md` no formato abaixo, cobrindo esta Task (features, árvore e rede). Um arquivo que só fala do SARIMA não fecha o item.

Um comando documentado no `README.md` do repo deve reproduzir `metricas.csv` (por exemplo `python -m pip install -r requirements.txt && python run.py`).

Os quatro baselines da Task 1 continuam no CSV, como foram entregues. O naive sazonal e o SARIMA ficam nessas linhas; o check só vê se continuam na tabela. Média, naive de um passo e drift não entram de novo na nota.

## Conteúdo obrigatório

**Barras, na mesma origem.** O naive sazonal repete a última semana do treino (período 7), quatro vezes. O SARIMA é o da Task 1: mesma ordem, ajuste só até 2016-03-27. A ordem não se reescolhe pelo MAE destes 28 dias. O ajuste é o da Task 1; este domingo não pede outro.

**Direto ou recursivo.** O grupo escolhe. Na direta, para cada dia do horizonte as colunas usam só o que já se conhece em 2016-03-27 e o calendário do dia que se prevê (dia da semana, Natal, véspera). No recursivo, a previsão de um dia pode voltar como lag do dia seguinte. Nas duas, o y da validação não entra na coluna: o que volta, se voltar, é a previsão do próprio modelo. O texto diz qual das duas entrou em `metricas.csv`.

Cada série tem o próprio ajuste. `FOODS`, `HOBBIES` e `store_total` não vão para a mesma tabela: o sábado de uma não é o sábado da outra.

**Três classes**, hiperparâmetros fixados antes de ler o MAE da validação (ou escolhidos em origens que ainda terminam em 2016-03-27):

- `linear` — o grupo escolhe a regressão. O nome no CSV é `linear`.
- `boosting` ou `arvore` — o grupo escolhe a árvore. Um dos dois nomes basta.
- `rede` — o grupo escolhe a rede. O nome no CSV é `rede`, `lstm`, `rnn` ou `transformer`.

A média e o desvio que padronizam a série, e os do preço, saem só do treino.

**Métricas.** MAE, RMSE e MASE na validação, para cada série e cada modelo exigido. O MASE usa como escala o MAE in-sample do naive sazonal semanal no treino (período 7), a mesma escala da Task 1.

O texto compara, em cada série, o linear, a árvore e a rede com o naive sazonal e com o SARIMA. Se algum perder para o naive sazonal, o texto diz em qual série e em qual métrica. `HOBBIES` não herda o ajuste de `FOODS` sem uma frase sobre o que mudou.

## Nota (10 pontos, crédito parcial)

A nota da Task 2 é a **soma** dos critérios. Falhar em um zera **só** aquele peso. Esses 10 pontos são os 15% da A2.

| Critério | Pontos | Fecha | Como entra no check |
|----------|--------|-------|---------------------|
| Split temporal sem vazamento | 2,0 | já na Task 1 | previsões só em 2016-03-28–2016-04-24; nenhuma data do holdout; o check confirma o que a Task 1 já deixou |
| MAE, RMSE e MASE na validação conhecida | 2,0 | 15/11 | as três métricas, nas três séries, nas linhas que cada domingo já acrescentou; o recálculo a partir do `yhat` fecha em 5%. Não é um segundo cálculo das classes anteriores |
| `linear` nas três séries | 1,5 | 01/11 | nome `linear` em `previsoes_validacao.csv` e `metricas.csv`, com MAE, RMSE e MASE que fecham |
| `boosting` ou `arvore` nas três séries | 1,5 | 08/11 | um dos dois nomes, nas três séries, com MAE, RMSE e MASE que fecham |
| `rede` nas três séries | 1,5 | 15/11 | `rede`, `lstm`, `rnn` ou `transformer`, com MAE, RMSE e MASE que fecham |
| Naive sazonal e SARIMA da Task 1 | 0,5 | já na Task 1 | `naive_sazonal` e `sarima` (ou `arima`) continuam na mesma tabela; o check não pede outro ajuste |
| Repo executável | 0,5 | já na Task 1 | o comando do README reproduz `metricas.csv`; o check confirma o repo que a Task 1 já deixou |
| `AI_USAGE.md` no formato, cobrindo esta Task | 0,5 | 15/11 | a seção nova traz os cinco itens, com linear, árvore e rede; a seção da Task 1 não se reescreve |
| **Total** | **10** | | |

O script público `checks/check_task2.py` (nesta pasta) é o mesmo critério que o professor usa na parte machine-checkable. Antes de cada domingo, rode com `--ate linear`, `--ate arvore` ou `--ate rede`. O default é `rede`, o check de 15/11. Não há janela informal de “conserta depois”.

O check não lê o notebook. A estratégia (direta ou recursiva), a série ajustada sozinha, quais colunas de `covariaveis.csv` entraram, o hiperparâmetro fora da validação e a frase quando o modelo perde para o naive sazonal ficam no texto. A defesa oral pergunta isso.

## Arguição em sala

A partir de 27/10, em sala, sorteia-se uma pessoa de cada grupo. A pergunta é sobre o que o repositório já entregou naquele dia: a Task 1 e, depois, a classe cujo domingo já passou. Resposta errada ou falta entra na nota da A2. O peso desse item não está nesta Task e não entra nos 10 pontos.

## `AI_USAGE.md` (obrigatório)

IA generativa **pode** ser usada. Sem este arquivo, no formato e cobrindo a Task 2, a nota máxima desta Task não fecha (0,5 ponto).

Conteúdo mínimo (~1–2 páginas):

1. Ferramentas usadas (ou a frase **não usamos IA**).
2. Onde ajudou (tarefas / partes do pipeline) — features, árvore e rede entram neste item.
3. **2–5 prompts representativos** (não o histórico completo).
4. **1–3 erros da IA que o grupo corrigiu** (ex.: leakage, métrica errada, early stopping na validação, lag com alvo futuro).
5. Responsabilidade: o grupo responde pelo que entregou.

## Fora desta entrega

- y do holdout e o score cego da Task 3 (30% da A2)
- Prophet, STL, Holt–Winters, ensemble e modelos de fundação
- intervalo de previsão
- covariáveis que não estão no CSV
- treino em nuvem

## Dúvidas de recorte

Vendas, datas do holdout e covariáveis estão em `dados/`. Não baixem o M5 completo nem misturem outras lojas. Qualquer dúvida de vazamento: se a informação não existiria em 2016-03-27, ela não entra na coluna da origem.
