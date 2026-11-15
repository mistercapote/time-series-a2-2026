# Registro de uso de IA — Task 1

## Ferramentas utilizadas

- Codex, assistente de IA generativa da OpenAI - GPT-6.1 Sol.
- Gemini via Antigravity
- Gemini 3.1 Pro

## Ajudas fornecidas pelos modelos

- Gemini utilizado na implementação do Algoritmo de Durbin-Levinson, em teoria é para estar fazendo do mesmo jeito que a biblioteca.

- Confirmar análises visuais das séries e ajudar na redação e inserção dos comentários.

- Gemini utilizado para descobrir como utilizar o modelo SARIMA no python, com a função `SARIMAX` da biblioteca `statsmodels`.

- Codex utilizado para ajudar na comparação de MAE, RMSE e MASE entre o SARIMA e os quatro baselines, conferir as métricas com as previsões e auxiliar na redação da análise comparativa da validação de 28 dias.

- Codex utilizado para formatar a tabela de métricas, gerar gráficos comparativos e organizar os títulos do notebook em uma hierarquia Markdown.

- Codex utilizado para revisar a interpretação de AIC/BIC.

## Prompts Representativos

- *"Diante das séries obtidas, confirme se a minha análise de tendência faz sentido, e me ajude a interpretar o gráfico do ACF"*

- _"Como posso utilizar o modelo SARIMA no python?"_

- Solicitação de orientação para comparar as métricas do SARIMA com as dos baselines e escrever uma análise objetiva dos resultados.

- Solicitação de visualização das métricas e padronização da apresentação do notebook.

## Erros e correções

- Análise de AIC/BIC foi inicialmente interpretada de forma incorreta, sugerindo que um certo conjunto de parâmetros era o melhor, quando na verdade não era. Em seguida, a análise foi corrigida e o conjunto correto de parâmetros foi identificado.

## Responsabilidade

O grupo responde pelas análises e pelo conteúdo entregue. Este registro descreve o apoio recebido de ferramentas de IA generativa, mas não substitui a responsabilidade do grupo sobre o trabalho final.

---

# Registro de uso de IA — Task 2

## Ferramentas utilizadas

- Codex (OpenAI)
- Gemini via Antigravity
- Claude Code (Anthropic), modelos Claude Sonnet 5.5 e Claude Opus 5.5

## Ajudas fornecidas pelos modelos

- Auxílio na organização das covariáveis de calendário, eventos, `snap_CA` e preços semanais por categoria.
- Auxílio na separação das entradas por série e na padronização do preço e das covariáveis usando somente o treino.
- Auxílio na implementação das funções genéricas de previsão (`prever_recursivo`, `prever_direto` e `prever_modelo`), do modelo `linear` (comparando regressão linear padrão OLS, Ridge e Lasso com `StandardScaler` e grade de $\alpha$) e do `sarimax` (SARIMA com covariáveis).
- Auxílio na seleção de hiperparâmetros e estratégia em janela de validação interna do treino (encerrada em `2016-03-27`), na atualização do pipeline em `run.py` e do gráfico comparativo em `src/viz.py`.
- Auxílio na implementação dos modelos Árvore e Boosting.
- Auxílio na implementação do grid search do SARIMA.
- Rede neural: o Claude Code atuou como tutor, passo a passo. Ajudou na compreensão e implementação do early stopping, padronização dentro do `fit` e a API do scikit-learn (`fit`/`predict`, `clone`).
- Claude ajudou na elaboração de Testes anti-vazamento da rede (`verificar_rede` em `src/testes.py`): sugestão da técnica de "espião" sobre `MLPTorch.fit`, que registra o que o ajuste recebe enquanto o pipeline real roda.
- Claude ajudou na separação de tarefas, e na estruturação de issues e pull requests para organizar o processo de desenvolvimento do projeto.
- Gemini ajudou na tarefa extra de estudo de RNNs (`lstm`): apoio do assistente na implementação de `montar_janelas`, `LSTMTorch` (reaproveitando `_treinar` em `src/nets.py`), `prever_recursivo_seq` sem vazamento da validação (`src/models.py`), seleção de hiperparâmetros com janelas $L \in \{14, 28\}$  e verificações em testes


## Prompts Representativos

- *"Quero criar essas covariáveis: dia da semana, semana do ano, mês e ano, junto com as outras variáveis."*
- *"Para os nomes dos eventos e tipos, não dá para unificar o 1 com o 2 no one-hot encoding?"*
- *"Verifique as implementações do modelo linear com covariáveis, tanto a recursiva quanto a direta, e veja se elas estão corretas e sem vazamento de dados."*
- *"Testar o modelo linear padrão, com Lasso e com Ridge, usando o direto e o recursivo para cada série, com normalização dos dados e seleção de hiperparâmetros usando o conjunto de validação interno até 2016-03-27."*
- *"Como identifico a ordem do modelo SARIMA com covariáveis?"*
- *"Separe o trabalho em sub-tarefas, e redija as descrições das issues e pull requests"*
- *"Mas o que é esse early stopping? Não entendi bem, pensei que fosse o mecanismo de encerrar o treinamento quando o gradiente do backpropagation fosse muito baixo."*

## Erros e correções

- **Desalinhamento temporal na estratégia direta:** a primeira versão pareava as covariáveis do dia de origem com o alvo futuro, mas usava as covariáveis do dia previsto na previsão. A função foi corrigida para manter o alinhamento temporal entre treino e previsão.
- **Reajuste redundante na estratégia recursiva:** o estimador era ajustado a cada passo. A função foi corrigida para clonar e treinar um único modelo antes do loop e realimentar a previsão anterior (`y_anterior`).
- **Duplo cálculo no pipeline:** `run.py` recalculava as previsões ao gerar as métricas. `comparar_metricas` foi ajustada para reutilizar as previsões já calculadas.
- **Seleção de hiperparâmetros e estratégia sem vazamento:** em vez de fixar arbitrariamente um único modelo linear, a escolha entre OLS padrão, Ridge e Lasso (com padronização via `StandardScaler` ajustado no treino), bem como do parâmetro $\alpha$ e da estratégia (`direta` ou `recursiva`), passou a ser feita em uma janela temporal interna do treino (`2016-02-29` a `2016-03-27`), preservando a validação final apenas para o reporte das métricas.
- **Nome de branch inválido sugerido pela IA:**: a IA propôs branches curtas `rede/7-torch` ao lado da branch `rede`, o que o Git não aceita (`refs/heads/rede` já existe como arquivo). O grupo passou a usar `rede-<issue>-<tema>`.

## Responsabilidade

O grupo revisou o código, validou a ausência de vazamento temporal e responde pelas decisões metodológicas, implementações e análises entregues nesta Task 2.