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
    "store_total": ((0, 1, 2), (0, 1, 1, 7)),
    "FOODS": ((1, 1, 1), (0, 1, 1, 7)),
    "HOBBIES": ((1, 1, 1), (0, 1, 1, 7)),
}

SARIMAX_DICT = {
    "store_total": ((0, 1, 2), (0, 1, 1, 7)),
    "FOODS": ((1, 1, 1), (0, 1, 1, 7)),
    "HOBBIES": ((1, 1, 1), (0, 1, 1, 7)),
}

# Configurações (tipo, alpha, estrategia) selecionadas na validação interna (até 2016-03-27)
LINEAR_DICT = {
    "store_total": ("lasso", 25.0, "recursiva"),
    "FOODS": ("lasso", 50.0, "recursiva"),
    "HOBBIES": ("ridge", 500.0, "recursiva"),
}

ARVORE_DICT = {
    "store_total": ("arvore", 10, 5, 100, "recursiva"),
    "FOODS": ("rf", 5, 5, 100, "recursiva"),
    "HOBBIES": ("arvore", 10, 5, 100, "recursiva"),
}

BOOSTING_DICT = {
    "store_total": ("gbr", 100, 0.1, 3, "recursiva"),
    "FOODS": ("hist", 50, 0.1, 3, "recursiva"),
    "HOBBIES": ("gbr", 100, 0.1, 3, "direta"),
}
