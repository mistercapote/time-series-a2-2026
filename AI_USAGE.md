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

# Registro de uso de IA — Task 2

## Ferramentas utilizadas

- Codex (OpenAI)
- Gemini via Antigravity

## Ajudas fornecidas pelos modelos

- Auxílio na organização das covariáveis de calendário, eventos, `snap_CA` e preços semanais por categoria.
- Auxílio na separação das entradas por série e na padronização do preço de cada série usando somente o treino.
- Auxílio na implementação do modelo `linear` com regularização Ridge (`alpha = 10`) nas abordagens direta e recursiva.
- Auxílio na atualização do pipeline em `run.py` e do gráfico comparativo em `src/viz.py`.

## Prompts Representativos

- *"Quero criar essas covariáveis: dia da semana, semana do ano, mês e ano, junto com as outras variáveis."*
- *"Para os nomes dos eventos e tipos, não dá para unificar o 1 com o 2 no one-hot encoding?"*
- *"Verifique as implementações do modelo linear com covariáveis, tanto a recursiva quanto a direta, e veja se elas estão corretas e sem vazamento de dados."*

## Erros e correções

- **Desalinhamento temporal na estratégia direta:** a primeira versão pareava as covariáveis do dia de origem com o alvo futuro, mas usava as covariáveis do dia previsto na previsão. A função foi corrigida para manter o alinhamento temporal entre treino e previsão.
- **Reajuste redundante na estratégia recursiva:** o Ridge era ajustado a cada passo. A função foi corrigida para treinar um único modelo antes do loop e realimentar a previsão anterior.
- **Duplo cálculo no pipeline:** `run.py` recalculava as previsões ao gerar as métricas. `comparar_metricas` foi ajustada para reutilizar as previsões já calculadas.
- **Seleção da estratégia:** a escolha entre direta e recursiva passou a ser feita em uma janela interna do treino, preservando a validação final apenas para o reporte das métricas.

## Responsabilidade

O grupo revisou o código, validou a ausência de vazamento temporal e responde pelas decisões metodológicas, implementações e análises entregues nesta Task 2.