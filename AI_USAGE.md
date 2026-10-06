# Registro de uso de IA — Task 1

## 1. Ferramentas utilizadas

Codex, assistente de IA generativa da OpenAI, em sessão de apoio à análise e à documentação do trabalho. O identificador exato do modelo não foi registrado nesta sessão.

## 2. Onde a IA ajudou

Modelo generativo utilizado para confirmar análises visuais das séries e ajudar na redação dos comentários.

O apoio registrado abrangeu as séries `store_total`, `FOODS` e `HOBBIES`, usando exclusivamente os dados de treino para o diagnóstico. Isaías examinou os gráficos e descreveu suas observações; o assistente ajudou a organizar essas observações em comentários curtos para o notebook.

Foram discutidos o crescimento do nível das séries, os picos recorrentes da ACF nos lags 7, 14 e 21 e as quedas a zero em 25 de dezembro de 2011 a 2015. O assistente também calculou autocorrelações e consultou as datas dos zeros no arquivo de treino para apoiar a conferência. A redação distingue a evidência de sazonalidade semanal, com período 7, de um possível padrão anual: os zeros recorrentes no Natal, por si só, não comprovam um ciclo anual de vendas.

Este registro descreve o uso de IA observado nesta sessão. Não pretende atestar a ausência de outros usos pelo grupo nem atribuir à IA a implementação integral do pipeline.

## 3. Prompts representativos

1. “Dê uma olhada no trabalho de séries temporais em /home/isaias/Vault do Isaías/FGV 2026.2/Séries Temporais/time-series-a2-2026”
2. “O próximo passo que eu devo fazer: Registrar uma observação objetiva para cada série, cobrindo: tendência, evidência de sazonalidade semanal com **m = 7** e presença de outliers, especialmente zeros de calendário. Texto curto, direto e reutilizável no relatório/notebook. Me oriente e me indique o que deve ser feito”
3. “Pode inserir para mim” — solicitação de inserção dos comentários construídos na conversa no notebook.

## 4. Erros da IA e correções documentadas

Houve um erro técnico do assistente ao tentar conferir as autocorrelações: ele utilizou inicialmente `statsmodels` em um ambiente onde essa dependência não estava disponível. A tentativa falhou com `ModuleNotFoundError`; o assistente corrigiu o procedimento calculando a ACF diretamente com NumPy no ambiente que dispunha de pandas e NumPy.

Essa correção foi feita pelo próprio assistente durante a sessão. Não há, nesta conversa, registro de um erro da IA corrigido pelo grupo. Portanto, este item ainda precisa ser complementado pelo grupo com uma ocorrência real para atender literalmente ao pedido do enunciado de “1–3 erros da IA que o grupo corrigiu”; não foram inventadas ocorrências.

## 5. Responsabilidade

O grupo responde pelo conteúdo entregue, incluindo código, análises, previsões, métricas e comentários. O uso de IA não transfere essa responsabilidade. Cabe ao grupo revisar este registro, conferir as interpretações com os gráficos e dados de treino e complementar eventuais usos de IA e correções que não estejam documentados nesta sessão.
