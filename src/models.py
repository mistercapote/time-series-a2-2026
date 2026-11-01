import numpy as np
import pandas as pd
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
) -> np.ndarray:
    """Previsão linear recursiva com um único modelo Ridge.

    Treina um único modelo que recebe as covariáveis do dia previsto e o alvo
    do dia imediatamente anterior (`y_anterior`). Na previsão, o primeiro passo
    usa o último valor observado no treino e os passos seguintes realimentam a
    própria previsão do passo anterior.

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

    _, X_tr, y_tr, X_pr = _obter_dados_serie(
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

    # Ajusta o modelo uma única vez antes do loop recursivo
    modelo = Ridge(alpha=alpha).fit(X_tr_rec, y_tr_rec)

    preds = []
    ultimo_y = float(y_tr.iloc[-1])

    for step in range(len(X_pr)):
        row = X_pr.iloc[step : step + 1].copy()
        row["y_anterior"] = ultimo_y
        y_hat = float(modelo.predict(row)[0])
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
) -> np.ndarray:
    """Previsão linear direta com um modelo independente por horizonte.

    Para cada horizonte ``h``, ajusta um modelo independente usando as
    covariáveis do próprio dia previsto ``X_{t+h}`` e o último valor conhecido
    na origem ``y_t`` para prever ``y_{t+h}``. Na previsão do horizonte ``h``,
    utiliza ``X_{T+h}`` e ``y_T`` (conhecido em 2016-03-27).

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
        np.ndarray: Previsões para todo o horizonte.
    """
    _, X_tr, y_tr, X_pr = _obter_dados_serie(
        serie=serie,
        n=n,
        X_treino=X_treino,
        y_treino=y_treino,
        X_pred=X_pred,
        excluir=excluir,
        padronizar_preco=padronizar_preco,
    )

    ultimo_y_origem = float(y_tr.iloc[-1])
    previsoes = []
    for horizonte in range(1, len(X_pr) + 1):
        n_amostras = len(X_tr) - horizonte
        if n_amostras < 1:
            raise ValueError(
                "O treino precisa conter mais observações que o horizonte."
            )

        X_horizonte = X_tr.iloc[horizonte : horizonte + n_amostras].copy()
        X_horizonte["y_origem"] = y_tr.iloc[:n_amostras].values
        y_horizonte = y_tr.iloc[horizonte : horizonte + n_amostras].values

        X_pred_h = X_pr.iloc[horizonte - 1 : horizonte].copy()
        X_pred_h["y_origem"] = ultimo_y_origem

        previsoes.append(
            float(
                base_previsao_linear(
                    X_horizonte,
                    y_horizonte,
                    X_pred_h,
                    alpha=alpha,
                )[0]
            )
        )

    return np.asarray(previsoes)


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


def prever_linear_com_covariaveis(
    serie: pd.Series,
    n: int,
    tabela_treino: pd.DataFrame,
    tabela_validacao: pd.DataFrame,
    estrategia: str = "direta",
) -> np.ndarray:
    """Gera previsões lineares usando as covariáveis de uma série."""
    import src.features as features

    nome_serie = str(serie.name)
    X_treino, y_treino = features.separar_X_y(
        nome_serie, periodo="treino", tabela=tabela_treino
    )
    X_pred, _ = features.separar_X_y(
        nome_serie, periodo="validacao", tabela=tabela_validacao
    )

    return linear(
        nome_serie,
        n=n,
        X_treino=X_treino,
        y_treino=y_treino,
        X_pred=X_pred,
        estrategia=estrategia,
    )


def construir_modelos(
    tabela_treino: pd.DataFrame,
    tabela_validacao: pd.DataFrame,
    sarima_dict: dict,
    estrategia_linear: str = "direta",
) -> dict:
    """Monta o dicionário de modelos usado na reprodução das previsões."""
    return {
        "media": lambda s, n: mean(s, n=n),
        "naive": lambda s, n: naive(s, n=n),
        "naive_sazonal": lambda s, n: naive_sazonal(s, m=7, n=n),
        "drift": lambda s, n: drift(s, n=n),
        "linear": lambda s, n: prever_linear_com_covariaveis(
            s,
            n,
            tabela_treino,
            tabela_validacao,
            estrategia=estrategia_linear,
        ),
        "sarima": lambda s, n: sarima(
            s,
            order=sarima_dict[s.name][0],
            seasonal_order=sarima_dict[s.name][1],
            n=n,
        ),
    }
