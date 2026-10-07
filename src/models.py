import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def mean(serie: pd.Series, n: int = 28) -> np.ndarray:
    """
    Calcula a previsão usando o método da média.

    Args:
        serie (pd.Series): Série temporal com os dados.
        n (int): Número de períodos futuros a serem previstos. Defaults to 28.

    Returns:
        np.ndarray: Previsões para os períodos futuros.
    """
    return np.full(shape=n, fill_value=serie.mean())


def naive(serie: pd.Series, n: int = 28) -> np.ndarray:
    """
    Calcula a previsão usando o método naive.

    Args:
        serie (pd.Series): Série temporal com os dados.
        n (int, optional): Número de períodos futuros a serem previstos. Defaults to 28.

    Returns:
        np.ndarray: Previsões para os períodos futuros.
    """
    last = serie.iloc[-1]
    return last * np.ones(n)


def naive_sazonal(serie: pd.Series, m: int = 7, n: int = 28) -> np.ndarray:
    """
    Calcula a previsão usando o método Naive Sazonal.

    Args:
        serie (pd.Series): Série temporal com os dados.
        m (int): Período da sazonalidade. Defaults to 7.
        n (int): Número de períodos futuros a serem previstos. Defaults to 28.
        
    Returns:
        np.ndarray: Previsões para os períodos futuros.
    """
    T = len(serie)
    Y_pred = []
    for h in np.arange(1, n + 1):
        k = (h - 1) // m
        i = (T + h) - m * (k + 1) - 1
        Y_pred.append(serie.iloc[i])
    return np.array(Y_pred)


def drift(serie: pd.Series, n: int = 28) -> np.ndarray:
    """
    Calcula a previsão usando o método Drift.

    Args:
        serie (pd.Series): Série temporal com os dados.
        n (int): Número de períodos futuros a serem previstos. Defaults to 28.
        
    Returns:
        np.ndarray: Previsões para os períodos futuros.
    """
    T = len(serie)
    y_T = serie.iloc[-1]
    y_1 = serie.iloc[0]
    C = (y_T - y_1) / (T-1)
    h =  np.arange(1, n + 1)
    return y_T + h * C

def sarima(serie: pd.Series, order: tuple[int], seasonal_order: tuple[int], n: int = 28, resid: bool = False) -> np.ndarray:
    """
    Calcula a previsão usando o método SARIMA(p,d,q)(P,D,Q)m

    Args:
        serie (pd.Series): Série temporal com os dados.
        order (tuple[int]): Parâmetros (p,d,q).
        seasonal_order (tuple[int]): Parâmetros (P,D,Q)m
        n (int, optional): Número de períodos futuros a serem previstos. Defaults to 28.
        resid (bool, optional): Retorna os resíduos. Defaults to False.

    Returns:
        np.ndarray: Previsões para os períodos futuros.
    """
    model = SARIMAX(
                    serie,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                    )

    result = model.fit(disp=False)
    prediction = result.get_forecast(steps=n)

    if resid:
        return result.resid

    return prediction.predicted_mean