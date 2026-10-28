# Dados visíveis — Task 1

Loja M5 `CA_1`. Três séries agregadas: `store_total`, `FOODS`, `HOBBIES`.

| Arquivo | Conteúdo |
|---------|----------|
| `treino.csv` | vendas até 2016-03-27 |
| `validacao.csv` | vendas de 2016-03-28 a 2016-04-24 (horizonte da Task 1) |
| `holdout_datas.csv` | só as datas de 2016-04-25 a 2016-05-22 (sem y) |

O y do holdout não está nesta pasta.

## Dados acrescentados — Task 2

Mesma loja e mesmas três séries. `treino.csv`, `validacao.csv` e `holdout_datas.csv` são os mesmos da Task 1 (não foram refeitos).

| Arquivo | Conteúdo |
|---------|----------|
| `covariaveis.csv` | SNAP (`snap_CA`), eventos (`event_name_1/2`, `event_type_1/2`) e preço semanal (`preco_FOODS`, `preco_HOBBIES`, `preco_store_total`), para 2011-01-29 a 2016-05-22 |

Detalhes de cada coluna e das regras de vazamento: ver [`../enunciado-task2.md`](../enunciado-task2.md).
