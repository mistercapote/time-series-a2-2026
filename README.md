# Task 1 — Diagnóstico, baselines e ARIMA/SARIMA

Pacote da Task 1 (prazo 30/09, 25% da A2). Enunciado, critérios e formato de entrega: [`enunciado.md`](enunciado.md).

| Peça | Arquivo |
|------|---------|
| Enunciado (prazo 30/09, 25% da A2, tabela de 10 pontos) | [`enunciado.md`](enunciado.md) · [`enunciado.pdf`](enunciado.pdf) |
| Treino e validação conhecida | [`dados/`](dados/) |
| Datas do holdout, sem y | [`dados/holdout_datas.csv`](dados/holdout_datas.csv) |
| Script de checks (crédito parcial) | [`checks/check_task1.py`](checks/check_task1.py) |

O y do holdout não entra nesta entrega. Rodar o check a partir da raiz desta pasta (o default de `--dados` é a pasta `dados/` ao lado de `checks/`):

```bash
python checks/check_task1.py --submission /caminho/do/repo/do/grupo
```
