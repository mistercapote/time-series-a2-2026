import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

from src.data import filtrar_intervalo_data


def grafico_temporal(
    serie: pd.Series,
    datas: pd.Series,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    y_max: int | float = None,
):
    """
    Plota o gráfico de uma coluna do dataframe ao longo do tempo.
    É possível definir o intervalo de tempo e altura máxima do gráfico para controlar a visualização.

    Args:
        serie (pd.Series): Série temporal a ser plotada.
        datas (pd.Series): Séries de datas correspondentes.
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        y_max (int | float, optional): Altura máxima do gráfico. Defaults to None.
    """
    # FILTRO

    datas_filtradas, serie_filtrada = filtrar_intervalo_data(serie, datas, data_inicio, data_fim)

    # PLOT

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(datas_filtradas, serie_filtrada)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    if titulo is not None:
        ax.set_title(titulo)

    plt.show()
    
def grafico_temporal_com_previsao(
    series_completo: pd.Series,
    datas_completo: pd.Series,
    previsoes: pd.Series,
    datas_previsao: pd.Series,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    y_max: int | float = None,
):
    """
    Plota o gráfico de uma coluna do dataframe ao longo do tempo, incluindo a previsão.
    É possível definir o intervalo de tempo e altura máxima do gráfico para controlar a visualização.

    Args:
        series_completo (pd.Series): Série temporal do completo a ser plotada.
        datas_completo (pd.Series): Séries de datas correspondentes ao completo.
        series_validacao (pd.Series): Série temporal da validação a ser plotada.
        datas_validacao (pd.Series): Séries de datas correspondentes à validação.
        previsao (pd.Series): Série temporal da previsão.
        datas_previsao (pd.Series): Séries de datas correspondentes à previsão.
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        y_max (int | float, optional): Altura máxima do gráfico. Defaults to None.
    """
    # FILTRO

    datas_filtradas, serie_filtrada = filtrar_intervalo_data(series_completo, datas_completo, data_inicio, data_fim)

    # PLOT

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(datas_filtradas, serie_filtrada, label=r"Y_t", color="blue")
    for label, previsao in previsoes.items():
        ax.plot(datas_previsao, previsao, label=f"{label}")
    
    # linha vertical para separar treino e validação
    if len(datas_previsao) > 0:
        ax.axvline(x=datas_previsao.iloc[0], color="gray", alpha=0.5, linestyle="--", label="Início da Validação")

    if y_max is not None:
        ax.set_ylim(0, y_max)

    if titulo is not None:
        ax.set_title(titulo)

    ax.legend(ncols=5,loc="upper center")
    plt.show()


def grafico_ACF(
    serie: pd.Series,
    datas: pd.Series,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    nlags: int = 30,
):
    """'
    Plota o gráfico de autocorrelação (ACF) de uma coluna do dataframe.

    Args:
        serie (pd.Series): Série temporal
        datas (pd.Series): Datas correspondentes à série
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        nlags (int, optional): Número de lags a serem calculados. Defaults to 30.

    Returns:
        None
    """
    # FILTRO
    datas_filtradas, serie_filtrada = filtrar_intervalo_data(serie, datas, data_inicio, data_fim)

    # CÁLCULO DA ACF
    y = np.array(serie_filtrada.dropna())
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
    serie: pd.Series,
    datas: pd.Series,
    titulo: str = None,
    data_inicio: str = None,
    data_fim: str = None,
    nlags: int = 30,
):
    """
    Plota o gráfico de autocorrelação parcial (PACF) de uma coluna do dataframe.

    Usa o algoritmo de Durbin-Levinson para calcular a PACF.

    Args:
        serie (pd.Series): Série temporal
        datas (pd.Series): Datas correspondentes à série
        titulo (str, optional): Título do gráfico. Defaults to None.
        data_inicio (str, optional): Data de início da visualização. Defaults to None.
        data_fim (str, optional): Data de fim da visualização. Defaults to None.
        nlags (int, optional): Número de lags a serem calculados. Defaults to 30.

    Returns:
        None
    """
    # FILTRO
    datas_filtradas, serie_filtrada = filtrar_intervalo_data(serie, datas, data_inicio, data_fim)

    # CÁLCULO DA PACF
    y = np.array(serie_filtrada.dropna())
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

    ax.set_title(titulo)
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel(r"$\hat{\alpha}(h)$")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()

def grafico_comparativo_metricas_por_modelo(resultados: list[dict]):
    """
    Gera um gráfico que compara as métricas de cada modelo em cada série.

    Args:
        resultados (list[dict]): Resultados com as métricas.
    """
    metricas_grafico = pd.DataFrame(resultados)
    ordem_modelos = ["media", "naive", "naive_sazonal", "drift", "sarima"]
    nomes_modelos = ["Média", "Naive", "Naive sazonal", "Drift", "SARIMA"]
    series_grafico = ["store_total", "FOODS", "HOBBIES"]
    cores = ["#94a3b8", "#94a3b8", "#94a3b8", "#94a3b8", "#16827c"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for linha, serie in enumerate(series_grafico):
        dados_serie = metricas_grafico.loc[
            metricas_grafico["series"] == serie
        ].set_index("modelo").reindex(ordem_modelos)
        for coluna, metrica in enumerate(["mae", "rmse", "mase"]):
            ax = axes[linha, coluna]
            valores = dados_serie[metrica].to_numpy()
            barras = ax.barh(nomes_modelos, valores, color=cores)
            ax.invert_yaxis()
            ax.set_xlim(0, valores.max() * 1.25)
            ax.bar_label(barras, fmt="%.3f" if metrica == "mase" else "%.2f", padding=4)
            ax.set_title(f"{serie} — {metrica.upper()}", loc="left", fontweight="bold")
            ax.set_xlabel("Erro escalonado" if metrica == "mase" else "Vendas/dia")
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Erros de previsão na validação de 28 dias\n"
        "Menor é melhor • SARIMA em verde • escalas próprias em cada painel",
        fontsize=15,
    )
    Path("resultados").mkdir(exist_ok=True)
    fig.savefig("resultados/comparacao_metricas.png", dpi=160, bbox_inches="tight")
    plt.show()