"""Constantes compartilhadas pelo notebook, pelo run.py e pelos módulos de src/.

Tudo que descreve o recorte do problema mora aqui, em um lugar só. Antes estes
valores estavam repetidos no run.py, no main.ipynb e no src/features.py: mudar
a origem da previsão em um deles e esquecer os outros faria o ajuste e a
previsão discordarem sem levantar erro.
"""

CAMINHO_TRAIN = "dados/treino.csv"
CAMINHO_VAL = "dados/validacao.csv"
CAMINHO_COVARIAVEIS = "dados/covariaveis.csv"

DATA_FIM_TRAIN = "2016-03-27"
DATA_INICIO_VAL = "2016-03-28"
DATA_FIM_VAL = "2016-04-24"

SERIES = ["store_total", "FOODS", "HOBBIES"]

SARIMA_DICT = {
    "store_total": ((1, 1, 1), (0, 1, 1, 7)),
    "FOODS": ((1, 1, 1), (0, 1, 1, 7)),
    "HOBBIES": ((0, 1, 2), (0, 1, 1, 7)),
}
