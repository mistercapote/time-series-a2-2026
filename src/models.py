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


def base_previsao_linear(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray | list,
    X_pred: pd.DataFrame | np.ndarray,
    alpha: float = 10.0,
) -> np.ndarray:
    """Função base que treina a regressão linear e constrói a previsão para os dias escolhidos.

    Args:
        X_train (pd.DataFrame | np.ndarray): Matriz de features de treino.
        y_train (pd.Series | np.ndarray | list): Alvo de treino (Y).
        X_pred (pd.DataFrame | np.ndarray): Matriz de features dos dias a prever.
        alpha (float): Fator de regularização L2 Ridge. Defaults to 10.0.

    Returns:
        np.ndarray: Previsões lineares correspondentes a X_pred.
    """
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=alpha).fit(X_train, y_train)
    return reg.predict(X_pred)


def _obter_dados_serie(
    serie: str | pd.Series,
    n: int = 28,
    X_treino: pd.DataFrame | None = None,
    y_treino: pd.Series | None = None,
    X_pred: pd.DataFrame | None = None,
    tabela_treino: pd.DataFrame | None = None,
    tabela_validacao: pd.DataFrame | None = None,
    excluir: list[str] | None = None,
    padronizar_preco: bool = True,
) -> tuple[str, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara matrizes X e y a partir de src.features para uma dada série."""
    import src.features as features

    if isinstance(serie, str):
        serie_nome = serie
    elif hasattr(serie, "name") and serie.name:
        serie_nome = str(serie.name)
    else:
        serie_nome = "store_total"

    if X_treino is None or y_treino is None:
        if tabela_treino is None:
            tabela_treino, _ = features.montar_treino_e_validacao()
        X_treino, y_treino = features.separar_X_y(
            serie_nome, periodo="treino", excluir=excluir, tabela=tabela_treino
        )

    if X_pred is None:
        if tabela_validacao is None:
            _, tabela_validacao = features.montar_treino_e_validacao()
        X_pred, _ = features.separar_X_y(
            serie_nome, periodo="validacao", excluir=excluir, tabela=tabela_validacao
        )

    if len(X_pred) > n:
        X_pred = X_pred.iloc[:n].copy()

    # Padronização do preço da série estritamente com a média e desvio do treino
    preco_col = f"preco_{serie_nome}"
    if padronizar_preco and preco_col in X_treino.columns:
        p_mean = float(X_treino[preco_col].mean())
        p_std = float(X_treino[preco_col].std())
        if p_std > 0:
            X_treino = X_treino.copy()
            X_pred = X_pred.copy()
            X_treino[preco_col] = (X_treino[preco_col] - p_mean) / p_std
            X_pred[preco_col] = (X_pred[preco_col] - p_mean) / p_std

    return serie_nome, X_treino, y_treino, X_pred


def prever_linear_recursivo(
    serie: str | pd.Series,
    n: int = 28,
    X_treino: pd.DataFrame | None = None,
    y_treino: pd.Series | None = None,
    X_pred: pd.DataFrame | None = None,
    alpha: float = 10.0,
    padronizar_preco: bool = True,
    excluir: list[str] | None = None,
    **kwargs,
) -> np.ndarray:
    """Previsão linear recursiva.

    Definição: treina o modelo para, dado os dados de um dia, prever qual o Y do próximo,
    incluindo o Y do anterior. Pega o último dado conhecido e passa pro modelo.

    Args:
        serie (str | pd.Series): Nome da série ou série com histórico.
        n (int): Quantidade de períodos a prever. Defaults to 28.
        X_treino (pd.DataFrame | None): Features de treino (opcional, obtido via src.features).
        y_treino (pd.Series | None): Alvo de treino (opcional, obtido via src.features).
        X_pred (pd.DataFrame | None): Features dos dias a prever.
        alpha (float): Regularização L2 Ridge. Defaults to 10.0.
        padronizar_preco (bool): Se True, padroniza preço com média e desvio do treino.
        excluir (list[str] | None): Colunas adicionais a descartar.

    Returns:
        np.ndarray: Previsões geradas recursivamente.
    """
    from sklearn.linear_model import Ridge

    serie_nome, X_tr, y_tr, X_pr = _obter_dados_serie(
        serie=serie,
        n=n,
        X_treino=X_treino,
        y_treino=y_treino,
        X_pred=X_pred,
        excluir=excluir,
        padronizar_preco=padronizar_preco,
    )

    # Prepara treino incluindo o Y do dia anterior como feature:
    # Para prever y_tr[t], utiliza X_tr[t] e y_tr[t-1]
    X_tr_rec = X_tr.iloc[1:].copy()
    X_tr_rec["y_anterior"] = y_tr.iloc[:-1].values
    y_tr_rec = y_tr.iloc[1:].values

    reg = Ridge(alpha=alpha).fit(X_tr_rec, y_tr_rec)

    # Loop recursivo: pega o último dado conhecido e passa pro modelo
    preds = []
    ultimo_y = float(y_tr.iloc[-1])

    for step in range(len(X_pr)):
        row = X_pr.iloc[step : step + 1].copy()
        row["y_anterior"] = ultimo_y
        y_hat = float(reg.predict(row)[0])
        preds.append(y_hat)
        ultimo_y = y_hat

    return np.array(preds)


def prever_linear_direto(
    serie: str | pd.Series,
    n: int = 28,
    X_treino: pd.DataFrame | None = None,
    y_treino: pd.Series | None = None,
    X_pred: pd.DataFrame | None = None,
    alpha: float = 10.0,
    padronizar_preco: bool = True,
    excluir: list[str] | None = None,
    **kwargs,
) -> np.ndarray:
    """Previsão linear direta com retreino contínuo.

    Definição: treina o modelo para, dado todos seus dados, prever qual o próximo,
    daí pega esses todos novos dados para prever o Yt+1, daí repete várias vezes.

    Args:
        serie (str | pd.Series): Nome da série ou série com histórico.
        n (int): Quantidade de períodos a prever. Defaults to 28.
        X_treino (pd.DataFrame | None): Features de treino (opcional, obtido via src.features).
        y_treino (pd.Series | None): Alvo de treino (opcional, obtido via src.features).
        X_pred (pd.DataFrame | None): Features dos dias a prever.
        alpha (float): Regularização L2 Ridge. Defaults to 10.0.
        padronizar_preco (bool): Se True, padroniza preço com média e desvio do treino.
        excluir (list[str] | None): Colunas adicionais a descartar.

    Returns:
        np.ndarray: Previsões geradas passo a passo com retreino.
    """
    serie_nome, X_tr, y_tr, X_pr = _obter_dados_serie(
        serie=serie,
        n=n,
        X_treino=X_treino,
        y_treino=y_treino,
        X_pred=X_pred,
        excluir=excluir,
        padronizar_preco=padronizar_preco,
    )

    curr_X = X_tr.copy()
    curr_y = list(y_tr.values)
    preds = []

    for step in range(len(X_pr)):
        next_row = X_pr.iloc[step : step + 1]
        y_hat = float(base_previsao_linear(curr_X, curr_y, next_row, alpha=alpha)[0])
        preds.append(y_hat)
        curr_X = pd.concat([curr_X, next_row], ignore_index=True)
        curr_y.append(y_hat)

    return np.array(preds)


def linear(
    serie: str | pd.Series,
    n: int = 28,
    estrategia: str = "direta",
    **kwargs,
) -> np.ndarray:
    """Modelo linear exigido pela Task 2 para as três séries.

    Args:
        serie (str | pd.Series): Nome da série ('store_total', 'FOODS', 'HOBBIES') ou série histórica.
        n (int): Horizonte de previsão (padrão 28 dias).
        estrategia (str): 'direta' ou 'recursiva'. Padrão 'direta'.
        **kwargs: Parâmetros adicionais repassados para a função escolhida.

    Returns:
        np.ndarray: Previsões geradas pelo modelo linear.
    """
    if estrategia == "recursiva":
        return prever_linear_recursivo(serie, n=n, **kwargs)
    elif estrategia == "direta":
        return prever_linear_direto(serie, n=n, **kwargs)
    else:
        raise ValueError(f"Estratégia inválida: '{estrategia}'. Use 'direta' ou 'recursiva'.")

