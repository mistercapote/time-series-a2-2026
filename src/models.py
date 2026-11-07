import numpy as np
import pandas as pd
import src.features as features
from sklearn.base import clone
from sklearn.linear_model import Ridge
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


def sarima(
    serie: pd.Series,
    order: tuple[int],
    seasonal_order: tuple[int],
    n: int = 28,
    resid: bool = False,
    maxiter: int = 200,
) -> np.ndarray:
    """
    Calcula a previsão usando o método SARIMA(p,d,q)(P,D,Q)m

    Args:
        serie (pd.Series): Série temporal com os dados.
        order (tuple[int]): Parâmetros (p,d,q).
        seasonal_order (tuple[int]): Parâmetros (P,D,Q)m
        n (int, optional): Número de períodos futuros a serem previstos. Defaults to 28.
        resid (bool, optional): Retorna os resíduos. Defaults to False.
        maxiter (int, optional): Número máximo de iterações da otimização. Defaults to 200.

    Returns:
        np.ndarray: Previsões para os períodos futuros.
    """
    model = SARIMAX(
                    serie,
                    order=order,
                    seasonal_order=seasonal_order
                    )

    result = model.fit(disp=False, maxiter=maxiter)
    prediction = result.get_forecast(steps=n)

    if resid:
        return result.resid

    return prediction.predicted_mean


def _obter_dados_serie(
    serie: str | pd.Series,
    n: int = 28,
    tabela_treino: pd.DataFrame | None = None,
    tabela_validacao: pd.DataFrame | None = None,
    excluir: list[str] | None = None,
    padronizar_preco: bool = True,
    padronizar_y: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float, float]:
    """Prepara matrizes X e y a partir de src.features e padroniza usando apenas o treino."""
    serie_nome = serie if isinstance(serie, str) else str(serie.name or "store_total")

    if tabela_treino is None or tabela_validacao is None:
        tab_tr, tab_val = features.montar_treino_e_validacao()
        tabela_treino = tabela_treino if tabela_treino is not None else tab_tr
        tabela_validacao = tabela_validacao if tabela_validacao is not None else tab_val

    X_treino, y_treino = features.separar_X_y(
        serie_nome, periodo="treino", excluir=excluir, tabela=tabela_treino
    )
    X_pred, _ = features.separar_X_y(
        serie_nome, periodo="validacao", excluir=excluir, tabela=tabela_validacao
    )
    X_pred = X_pred.iloc[:n].copy()

    preco_col = f"preco_{serie_nome}"
    if padronizar_preco and preco_col in X_treino.columns:
        p_mean = float(X_treino[preco_col].mean())
        p_std = float(X_treino[preco_col].std())
        if p_std > 0:
            X_treino = X_treino.copy()
            X_treino[preco_col] = (X_treino[preco_col] - p_mean) / p_std
            X_pred[preco_col] = (X_pred[preco_col] - p_mean) / p_std

    y_mean, y_std = float(y_treino.mean()), float(y_treino.std())
    if padronizar_y and y_std > 0:
        y_treino = (y_treino - y_mean) / y_std

    return X_treino, y_treino, X_pred, y_mean, y_std


def prever_recursivo(
    serie: str | pd.Series,
    modelo: object,
    n: int = 28,
    tabela_treino: pd.DataFrame | None = None,
    tabela_validacao: pd.DataFrame | None = None,
    padronizar_preco: bool = True,
    padronizar_y: bool = False,
    excluir: list[str] | None = None,
) -> np.ndarray:
    """Previsão recursiva genérica: ajusta um único modelo e realimenta y_anterior."""
    X_tr, y_tr, X_pr, y_mean, y_std = _obter_dados_serie(
        serie, n, tabela_treino, tabela_validacao, excluir, padronizar_preco, padronizar_y
    )

    X_tr_rec = X_tr.iloc[1:].copy()
    X_tr_rec["y_anterior"] = y_tr.iloc[:-1].values
    y_tr_rec = y_tr.iloc[1:].values

    estimador = clone(modelo).fit(X_tr_rec, y_tr_rec)

    preds = []
    ultimo_y = float(y_tr.iloc[-1])
    for step in range(len(X_pr)):
        row = X_pr.iloc[step : step + 1].copy()
        row["y_anterior"] = ultimo_y
        y_hat = float(np.asarray(estimador.predict(row)).reshape(-1)[0])
        preds.append(y_hat)
        ultimo_y = y_hat

    resultado = np.asarray(preds)
    return resultado * y_std + y_mean if (padronizar_y and y_std > 0) else resultado


def prever_direto(
    serie: str | pd.Series,
    modelo: object,
    n: int = 28,
    tabela_treino: pd.DataFrame | None = None,
    tabela_validacao: pd.DataFrame | None = None,
    padronizar_preco: bool = True,
    padronizar_y: bool = False,
    excluir: list[str] | None = None,
) -> np.ndarray:
    """Previsão direta genérica: ajusta um modelo independente para cada horizonte h."""
    X_tr, y_tr, X_pr, y_mean, y_std = _obter_dados_serie(
        serie, n, tabela_treino, tabela_validacao, excluir, padronizar_preco, padronizar_y
    )

    ultimo_y_origem = float(y_tr.iloc[-1])
    previsoes = []
    for horizonte in range(1, len(X_pr) + 1):
        n_amostras = len(X_tr) - horizonte
        X_horizonte = X_tr.iloc[horizonte : horizonte + n_amostras].copy()
        X_horizonte["y_origem"] = y_tr.iloc[:n_amostras].values
        y_horizonte = y_tr.iloc[horizonte : horizonte + n_amostras].values

        X_pred_h = X_pr.iloc[horizonte - 1 : horizonte].copy()
        X_pred_h["y_origem"] = ultimo_y_origem

        estimador = clone(modelo).fit(X_horizonte, y_horizonte)
        previsoes.append(float(np.asarray(estimador.predict(X_pred_h)).reshape(-1)[0]))

    resultado = np.asarray(previsoes)
    return resultado * y_std + y_mean if (padronizar_y and y_std > 0) else resultado


def prever_modelo(
    serie: str | pd.Series,
    n: int,
    tabela_treino: pd.DataFrame,
    tabela_validacao: pd.DataFrame,
    modelo: object,
    estrategia: str = "direta",
    **kwargs,
) -> np.ndarray:
    """Executa a previsão de qualquer modelo com covariáveis na estratégia escolhida."""
    if estrategia == "recursiva":
        return prever_recursivo(
            serie, modelo, n=n, tabela_treino=tabela_treino, tabela_validacao=tabela_validacao, **kwargs
        )
    if estrategia == "direta":
        return prever_direto(
            serie, modelo, n=n, tabela_treino=tabela_treino, tabela_validacao=tabela_validacao, **kwargs
        )
    raise ValueError(f"Estratégia inválida: '{estrategia}'. Use 'direta' ou 'recursiva'.")


def construir_modelos(
    tabela_treino: pd.DataFrame,
    tabela_validacao: pd.DataFrame,
    sarima_dict: dict,
    modelos_covariaveis: dict[str, tuple[object, str]] | None = None,
) -> dict:
    """Monta o dicionário de modelos usado na reprodução das previsões."""
    if modelos_covariaveis is None:
        modelos_covariaveis = {"linear": (Ridge(alpha=10.0), "recursiva")}

    dicionario = {
        "media": lambda s, n: mean(s, n=n),
        "naive": lambda s, n: naive(s, n=n),
        "naive_sazonal": lambda s, n: naive_sazonal(s, m=7, n=n),
        "drift": lambda s, n: drift(s, n=n),
        "sarima": lambda s, n: sarima(
            s,
            order=sarima_dict[s.name][0],
            seasonal_order=sarima_dict[s.name][1],
            n=n,
        ),
    }

    for nome, (estimador, estrategia) in modelos_covariaveis.items():
        dicionario[nome] = (
            lambda s, n, est=estimador, estr=estrategia: prever_modelo(
                s, n, tabela_treino, tabela_validacao, modelo=est, estrategia=estr
            )
        )

    return dicionario
