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

    plt.plot(df_filtrado["date"], df_filtrado[coluna])

    if y_max is not None:
        plt.ylim(0, y_max)

    if titulo is not None:
        plt.title(titulo)

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

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(Hs, rho, basefmt="k-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(
        Hs, -intervalo_confianca, intervalo_confianca, color="blue", alpha=0.2
    )

    ax.set_title(titulo)
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel("rho(h)")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()
