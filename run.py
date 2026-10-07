"""Script para reprodução automatizada das métricas de validação.

Gera o arquivo 'resultados/metricas.csv'.
"""

import src.data as data
import src.model_checking as model_checking
import src.models as models

CAMINHO_TRAIN = "dados/treino.csv"
CAMINHO_VAL = "dados/validacao.csv"
DATA_FIM_TRAIN = "2016-03-27"
DATA_INICIO_VAL = "2016-03-28"
DATA_FIM_VAL = "2016-04-24"

SARIMA_DICT = {
    "store_total": ((1, 1, 1), (0, 1, 1, 7)),
    "FOODS": ((1, 1, 1), (0, 1, 1, 7)),
    "HOBBIES": ((0, 1, 2), (0, 1, 1, 7)),
}

MODELOS_DICT = {
    "media": lambda s, n: models.mean(s, n=n),
    "naive": lambda s, n: models.naive(s, n=n),
    "naive_sazonal": lambda s, n: models.naive_sazonal(s, m=7, n=n),
    "drift": lambda s, n: models.drift(s, n=n),
    "sarima": lambda s, n: models.sarima(
        s,
        order=SARIMA_DICT[s.name][0],
        seasonal_order=SARIMA_DICT[s.name][1],
        n=n,
    ),
}

def main():
    train_data = data.carregar_dados(caminho=str(CAMINHO_TRAIN), data_fim=DATA_FIM_TRAIN)
    val_data = data.carregar_dados(caminho=str(CAMINHO_VAL), data_inicio=DATA_INICIO_VAL,data_fim=DATA_FIM_VAL)
    labels = val_data.columns[1:]

    resultados = model_checking.comparar_metricas(MODELOS_DICT, train_data, val_data, labels)

    data.criar_csv(resultados, "metricas.csv")

if __name__ == "__main__":
    main()