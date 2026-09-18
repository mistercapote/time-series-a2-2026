import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data import filtrar_intervalo_data


def grafico_temporal(
    df: pd.DataFrame,
    coluna: str,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    y_max: int | float = None,
):
    """
    Plota o gráfico de uma coluna do dataframe ao longo do tempo.
    É possível definir o intervalo de tempo e altura máxima do gráfico para controlar a visualização.

    Args:
        df (pd.DataFrame): Dataframe
        coluna (str): Coluna
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        y_max (int | float, optional): Altura máxima do gráfico. Defaults to None.
    """
    # FILTRO

    df_filtrado = filtrar_intervalo_data(df, data_inicio, data_fim)

    # PLOT

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df_filtrado["date"], df_filtrado[coluna])

    if y_max is not None:
        ax.set_ylim(0, y_max)

    if titulo is not None:
        ax.set_title(titulo)

    plt.show()


def grafico_ACF(
    df: pd.DataFrame,
    coluna: str,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    nlags: int = 30,
):
    """'
    Plota o gráfico de autocorrelação (ACF) de uma coluna do dataframe.

    Args:
        df (pd.DataFrame): Dataframe
        coluna (str): Coluna
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        nlags (int, optional): Número de lags a serem calculados. Defaults to 30.

    Returns:
        None
    """
    # FILTRO
    df_filtrado = filtrar_intervalo_data(df, data_inicio, data_fim)

    # CÁLCULO DA ACF
    y = np.array(df_filtrado[coluna].dropna())
    T = len(y)

    nlags = min(nlags, T - 1)

    y_diff = y - np.mean(y)

    Hs = np.arange(nlags + 1)
    gamma_H = []

    # Loop para calcular a autocovariância para cada lag
    for h in Hs:
        if h == 0:
            cov_h = np.sum(y_diff * y_diff) / T
        else:
            cov_h = np.sum(y_diff[:-h] * y_diff[h:]) / T
        gamma_H.append(cov_h)

    gamma_H = np.array(gamma_H)
    # Calcular a autocorrelação
    rho = gamma_H / gamma_H[0]

    # Plotar o gráfico da ACF
    intervalo_confianca = 1.96 / np.sqrt(T)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stem(Hs, rho, basefmt="k-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(
        Hs, -intervalo_confianca, intervalo_confianca, color="blue", alpha=0.2
    )

    ax.set_title(titulo)
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel(r"$\hat{\rho}(h)$")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def grafico_PACF(
    df: pd.DataFrame,
    coluna: str,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    nlags: int = 30,
):
    """
    Plota o gráfico de autocorrelação parcial (PACF) de uma coluna do dataframe.

    Usa o algoritmo de Durbin-Levinson para calcular a PACF.

    Args:
        df (pd.DataFrame): Dataframe
        coluna (str): Coluna
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        nlags (int, optional): Número de lags a serem calculados. Defaults to 30.

    Returns:
        None
    """
    # FILTRO
    df_filtrado = filtrar_intervalo_data(df, data_inicio, data_fim)

    # CÁLCULO DA PACF
    y = np.array(df_filtrado[coluna].dropna())
    T = len(y)

    nlags = min(nlags, T - 1)
    y_diff = y - np.mean(y)

    Hs = np.arange(nlags + 1)
    gamma_H = []

    # Loop para calcular a autocovariância para cada lag
    for h in Hs:
        if h == 0:
            cov_h = np.sum(y_diff * y_diff) / T
        else:
            cov_h = np.sum(y_diff[:-h] * y_diff[h:]) / T
        gamma_H.append(cov_h)

    gamma_H = np.array(gamma_H)

    # Calcular a autocorrelação
    rho = gamma_H / gamma_H[0]

    # Algoritmo de Durbin-Levinson para obter o PACF
    pacf = np.zeros(nlags + 1)
    phi = np.zeros((nlags + 1, nlags + 1))

    pacf[0] = 1.0
    if nlags >= 1:
        pacf[1] = rho[1]
        phi[1, 1] = rho[1]

    for k in range(2, nlags + 1):
        # Cálculo da PACF para o lag k
        num = rho[k] - np.sum(phi[k - 1, 1:k] * rho[k - 1 : 0 : -1])
        den = 1.0 - np.sum(phi[k - 1, 1:k] * rho[1:k])
        phi[k, k] = num / den
        pacf[k] = phi[k, k]

        # Atualização dos coeficientes intermediários
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    # Ou pode ser usado o da biblioteca statsmodels
    # from statsmodels.tsa.stattools import pacf as sm_pacf
    # pacf = sm_pacf(y, nlags=nlags, method='ywm')

    # Plotar o gráfico da PACF
    intervalo_confianca = 1.96 / np.sqrt(T)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stem(Hs, pacf, basefmt="k-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(
        Hs,
        -intervalo_confianca,
        intervalo_confianca,
        color="blue",
        alpha=0.2,
    )

    ax.set_title(titulo if titulo else f"Autocorrelação Parcial (PACF) - {coluna}")
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel(r"$\hat{\alpha}(h)$")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()