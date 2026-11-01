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
    ordem_preferida = [
        "media",
        "naive",
        "naive_sazonal",
        "drift",
        "linear",
        "sarima",
    ]
    modelos_presentes = set(metricas_grafico["modelo"].dropna())
    ordem_modelos = [
        modelo for modelo in ordem_preferida if modelo in modelos_presentes
    ]
    ordem_modelos.extend(
        modelo for modelo in metricas_grafico["modelo"].dropna().unique()
        if modelo not in ordem_modelos
    )
    nomes_por_modelo = {
        "media": "Média",
        "naive": "Naive",
        "naive_sazonal": "Naive sazonal",
        "drift": "Drift",
        "linear": "Linear",
        "sarima": "SARIMA",
    }
    nomes_modelos = [
        nomes_por_modelo.get(modelo, str(modelo)) for modelo in ordem_modelos
    ]
    series_grafico = ["store_total", "FOODS", "HOBBIES"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(top=0.79, hspace=0.5, wspace=0.35)
    for linha, serie in enumerate(series_grafico):
        dados_serie = metricas_grafico.loc[
            metricas_grafico["series"] == serie
        ].set_index("modelo").reindex(ordem_modelos)
        for coluna, metrica in enumerate(["mae", "rmse", "mase"]):
            ax = axes[linha, coluna]
            valores = dados_serie[metrica].to_numpy()
            ordem_melhores = np.argsort(valores)
            cores = np.full(len(valores), "#94a3b8", dtype=object)
            if len(ordem_melhores) >= 1:
                cores[ordem_melhores[0]] = "#16827c"
            if len(ordem_melhores) >= 2:
                cores[ordem_melhores[1]] = "#d97706"

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
        "Erros de previsão na validação de 28 dias",
        fontsize=15,
        y=0.96,
    )
    POSICAO_SUBTITULO = 0.875
    legenda_subtitulo = [
        ("Menor é melhor • ", "#475569", "normal"),
        ("Verde", "#16827c", "bold"),
        (" = melhor • ", "#475569", "normal"),
        ("laranja", "#d97706", "bold"),
        (" = segundo melhor • ", "#475569", "normal"),
        ("cinza", "#94a3b8", "bold"),
        (" = demais • escalas próprias em cada painel", "#475569", "normal"),
    ]
    textos_subtitulo = [
        fig.text(
            0,
            POSICAO_SUBTITULO,
            texto,
            color=cor,
            fontweight=peso,
            fontsize=11,
        )
        for texto, cor, peso in legenda_subtitulo
    ]
    fig.canvas.draw()
    largura_total = sum(
        texto.get_window_extent(fig.canvas.get_renderer()).width
        for texto in textos_subtitulo
    )
    largura_figura = fig.bbox.width
    x_atual = (largura_figura - largura_total) / (2 * largura_figura)
    for texto in textos_subtitulo:
        texto.set_position((x_atual, POSICAO_SUBTITULO))
        x_atual += (
            texto.get_window_extent(fig.canvas.get_renderer()).width / largura_figura
        )
    Path("resultados").mkdir(exist_ok=True)
    fig.savefig("resultados/comparacao_metricas.png", dpi=160, bbox_inches="tight")
    plt.show()