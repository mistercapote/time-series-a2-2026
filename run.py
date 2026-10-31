"""Script para reprodução automatizada das previsões e métricas de validação.

Gera 'previsoes_validacao.csv' e 'metricas.csv', em resultados/ e na raiz.
"""

import src.config as config
import src.data as data
import src.model_checking as model_checking
import src.models as models

CAMINHO_TRAIN = config.CAMINHO_TRAIN
CAMINHO_VAL = config.CAMINHO_VAL
DATA_FIM_TRAIN = config.DATA_FIM_TRAIN
DATA_INICIO_VAL = config.DATA_INICIO_VAL
DATA_FIM_VAL = config.DATA_FIM_VAL

SARIMA_DICT = config.SARIMA_DICT

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

    registros = model_checking.comparar_previsoes(MODELOS_DICT, train_data, val_data["date"], labels)

    data.criar_csv(registros, "previsoes_validacao.csv")

    resultados = model_checking.comparar_metricas(MODELOS_DICT, train_data, val_data, labels)

    data.criar_csv(resultados, "metricas.csv")

if __name__ == "__main__":
    main()