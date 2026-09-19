import numpy as np
import pandas as pd


def mean(serie: pd.Series, n: int = 7):
    """
    Calcula a previsão usando o método da média.

    Args:
        serie (pd.Series): Série temporal com os dados.
        n (int): Número de períodos futuros a serem previstos. Defaults to 7.
    """
    return np.full(shape=n, fill_value=serie.mean())


def naive_sazonal(serie: pd.Series, m: int = 7, n: int = 7):
    """
    Calcula a previsão usando o método Naive Sazonal.

    Args:
        serie (pd.Series): Série temporal com os dados.
        m (int): Período da sazonalidade. Defaults to 7.
        n (int): Número de períodos futuros a serem previstos. Defaults to 7.
    """
    T = len(serie)
    Y_pred = []
    for h in np.arange(1, n + 1):
        k = (h - 1) // m
        i = (T + h) - m * (k + 1) - 1
        Y_pred.append(serie.iloc[i])
    return np.array(Y_pred)


def drift(serie: pd.Series, n: int = 7):
    """
    Calcula a previsão usando o método Naive Sazonal.

    Args:
        serie (pd.Series): Série temporal com os dados.
        n (int): Número de períodos futuros a serem previstos. Defaults to 7.
    """
    T = len(serie)
    y_T = serie.iloc[-1]
    y_1 = serie.iloc[0]
    C = (y_T - y_1) / (T-1)
    h =  np.arange(1, n + 1)
    return y_T + h * C