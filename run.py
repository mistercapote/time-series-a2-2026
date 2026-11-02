"""Script para reprodução automatizada das previsões e métricas de validação.

Gera 'previsoes_validacao.csv' e 'metricas.csv', em resultados/ e na raiz.
"""

import src.config as config
import src.data as data
import src.features as features
import src.model_checking as model_checking
import src.models as models

CAMINHO_TRAIN = config.CAMINHO_TRAIN
CAMINHO_VAL = config.CAMINHO_VAL
DATA_FIM_TRAIN = config.DATA_FIM_TRAIN
DATA_INICIO_VAL = config.DATA_INICIO_VAL
DATA_FIM_VAL = config.DATA_FIM_VAL

SARIMA_DICT = config.SARIMA_DICT
ESTRATEGIA_LINEAR = "recursiva"

def main():
    train_data = data.carregar_dados(caminho=str(CAMINHO_TRAIN), data_fim=DATA_FIM_TRAIN)
    val_data = data.carregar_dados(caminho=str(CAMINHO_VAL), data_inicio=DATA_INICIO_VAL,data_fim=DATA_FIM_VAL)
    labels = val_data.columns[1:]
    tabela_treino, tabela_validacao = features.montar_treino_e_validacao()
    modelos = models.construir_modelos(
        tabela_treino,
        tabela_validacao,
        SARIMA_DICT,
        estrategia_linear=ESTRATEGIA_LINEAR,
    )

    registros = model_checking.comparar_previsoes(modelos, train_data, val_data["date"], labels)

    data.criar_csv(registros, "previsoes_validacao.csv")

    resultados = model_checking.comparar_metricas(
        modelos, train_data, val_data, labels, registros=registros
    )

    data.criar_csv(resultados, "metricas.csv")

if __name__ == "__main__":
    main()