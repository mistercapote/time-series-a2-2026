# Registro de uso de IA — Task 1

## Ferramentas utilizadas

- Codex, assistente de IA generativa da OpenAI - GPT-6.1 Sol.
- Gemini via Antigravity

## Ajudas fornecidas pelos modelos

- Gemini utilizado na implementação do Algoritmo de Durbin-Levinson, em teoria é para estar fazendo do mesmo jeito que a biblioteca.

- Modelo generativo utilizado para confirmar análises visuais das séries e ajudar na redação e inserção dos comentários.

- Codex utilizado para apoiar a comparação de MAE, RMSE e MASE entre o SARIMA e os quatro baselines, conferir as métricas com as previsões e redigir a análise comparativa da validação de 28 dias.

- Codex utilizado para revisar a interpretação de AIC/BIC e alinhar a configuração do SARIMA de `HOBBIES` entre o notebook e o script de reprodução.

- Codex utilizado para formatar a tabela de métricas, gerar gráficos comparativos e organizar os títulos do notebook em uma hierarquia Markdown.

## Prompts Representativos

- *"Diante das séries obtidas, confirme se a minha análise de tendência faz sentido, e me ajude a interpretar o gráfico do ACF"*

- Solicitação de orientação para comparar as métricas do SARIMA com as dos baselines e escrever uma análise objetiva dos resultados.

- Solicitação de correção das inconsistências na justificativa dos modelos e na configuração do script de reprodução.

- Solicitação de visualização das métricas e padronização da apresentação do notebook.


## Erros e correções

Uma tentativa de conferência utilizou uma dependência indisponível no ambiente; o próprio assistente ajustou o procedimento. Não foi documentado, nesta sessão, um erro da IA corrigido pelo grupo.

Na revisão, foram identificadas e corrigidas afirmações invertidas sobre AIC/BIC e uma divergência na ordem do SARIMA de `HOBBIES` entre notebook e script. A origem dessas inconsistências não foi estabelecida, portanto elas não são atribuídas à IA. As métricas foram conferidas com as previsões, e o cálculo pelo script corrigido foi reproduzido sem diferenças relevantes em relação aos resultados registrados.

## Responsabilidade

O grupo responde pelas análises e pelo conteúdo entregue. Este registro descreve o apoio observado nesta sessão e deve ser complementado caso tenham ocorrido outros usos ou correções.
