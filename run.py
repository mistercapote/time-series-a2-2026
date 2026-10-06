"""Script para reprodução automatizada das métricas de validação.

Gera o arquivo 'metricas.csv' na raiz do projeto.
"""

import src.data as data
import src.metrics as metrics
import src.models as models

CAMINHO_TRAIN = "dados/treino.csv"
CAMINHO_VAL = "dados/validacao.csv"
DATA_FIM_TRAIN = "2016-03-27"
DATA_INICIO_VAL = "2016-03-28"
DATA_FIM_VAL = "2016-04-24"

ORDER_SARIMA = (0,1,2)
SAZONAL_ORDER_SARIMA = (0,1,1,7)

MODELOS_DICT = {
    "media": lambda s, n: models.mean(s, n=n),
    "naive": lambda s, n: models.naive(s, n=n),
    "naive_sazonal": lambda s, n: models.naive_sazonal(s, m=7, n=n),
    "drift": lambda s, n: models.drift(s, n=n),
    "sarima": lambda s, n: models.sarima(s, ORDER_SARIMA, SAZONAL_ORDER_SARIMA, n=n)
}

def main():
    train_data = data.carregar_dados(caminho=str(CAMINHO_TRAIN), data_fim=DATA_FIM_TRAIN)
    val_data = data.carregar_dados(caminho=str(CAMINHO_VAL), data_inicio=DATA_INICIO_VAL,data_fim=DATA_FIM_VAL)
    labels = val_data.columns[1:]
    n_val = len(val_data)

    resultados = []
    for label in labels:
        y_train = train_data[label]
        y_val = val_data[label].values
        for nome_modelo, func_modelo in MODELOS_DICT.items():
            y_pred = func_modelo(y_train, n_val)
            val_mae = metrics.mae(y_val, y_pred)
            val_rmse = metrics.rmse(y_val, y_pred)
            val_mase = metrics.mase(y_train, y_val, y_pred, m=7)

            resultados.append({
                "series": label,
                "modelo": nome_modelo,
                "mae": val_mae,
                "rmse": val_rmse,
                "mase": val_mase,
            })

    data.criar_csv(resultados, "metricas.csv")

if __name__ == "__main__":
    main()