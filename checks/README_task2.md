# Checks da Task 2

O mesmo critério machine-checkable da correção. Crédito parcial: cada critério zera só o próprio peso.

A partir da raiz do repositório do curso:

```bash
python materia/projeto-task2/checks/check_task2.py --submission /caminho/do/repo/do/grupo
```

`--dados` é opcional: o default é `dados/` nesta pasta da Task 2, ao lado de `checks/`.

`--ate` limita a classe que já venceu. O default é `rede`, o check de 15/11 (10 pontos). Antes de 01/11 use `--ate linear` (máximo 7: árvore e rede saem e aparecem como prazo futuro). Antes de 08/11 use `--ate arvore` (máximo 8,5: só a rede fica de fora). Cada corte cobra a classe nova; split, barras e repo só confirmam o que a Task 1 já deixou no arquivo. O recálculo de MAE, RMSE e MASE usa só os modelos já devidos nesse corte, então um linear correto não perde os 2,0 de métricas por falta da rede. O `AI_USAGE.md` precisa citar as classes do corte (`linear`; no corte da árvore, também a árvore; em `rede`, as três). O 2,0 de métricas e o 0,5 do `AI_USAGE.md` fecham no check de 15/11, nas linhas que cada domingo já acrescentou.

Pesos e o domingo em que cada critério fecha: ver [`../enunciado.md`](../enunciado.md).
