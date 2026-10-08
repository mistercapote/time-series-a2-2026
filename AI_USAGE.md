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
