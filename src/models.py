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


def extrair_covariaveis_base(
    df_datas: pd.DataFrame,
    covariaveis: pd.DataFrame,
    serie_nome: str,
    p_mean: float,
    p_std: float,
    incluir_snap: bool = True,
) -> pd.DataFrame:
    """Extrai features exógenas e de calendário para uma série temporal.

    Garante ausência de vazamento: o preço é padronizado estritamente
    utilizando a média e desvio amostral calculados no conjunto de treino.

    Args:
        df_datas (pd.DataFrame): DataFrame contendo coluna 'date'.
        covariaveis (pd.DataFrame): Dados de covariaveis.csv.
        serie_nome (str): Nome da série ('store_total', 'FOODS', 'HOBBIES').
        p_mean (float): Média histórica do preço no treino.
        p_std (float): Desvio-padrão histórico do preço no treino.
        incluir_snap (bool): Se True, inclui indicador SNAP CA. Defaults to True.

    Returns:
        pd.DataFrame: Matriz de covariáveis alinhada com as datas fornecidas.
    """
    sub = pd.merge(df_datas[["date"]], covariaveis, on="date", how="left")
    price_col = f"preco_{serie_nome}"
    
    # Padronização com média e desvio fixados no treino
    if p_std > 0:
        sub["preco_std"] = (sub[price_col] - p_mean) / p_std
    else:
        sub["preco_std"] = 0.0

    sub["dayofweek"] = sub["date"].dt.dayofweek
    sub["is_event"] = sub["event_name_1"].notna().astype(float)
    
    # Tendência linear normalizada
    sub["trend"] = (sub["date"] - pd.Timestamp("2011-01-29")).dt.days / 1885.0

    # Dummies para os dias da semana (6 dummies, segunda-feira como referência)
    dow = pd.get_dummies(sub["dayofweek"], prefix="dow", drop_first=True, dtype=float)
    for i in range(1, 7):
        col = f"dow_{i}"
        if col not in dow.columns:
            dow[col] = 0.0
    dow = dow[[f"dow_{i}" for i in range(1, 7)]]

    feats = pd.concat([sub[["trend", "preco_std", "is_event"]], dow], axis=1)

    if incluir_snap:
        feats["snap_CA"] = sub["snap_CA"].astype(float)
    else:
        feats["snap_CA"] = 0.0

    return feats


def base_previsao_linear(
    y_train: list | np.ndarray | pd.Series,
    datas_treino: pd.Series | pd.DatetimeIndex | list,
    datas_prever: pd.Series | pd.DatetimeIndex | list,
    covariaveis: pd.DataFrame,
    serie_nome: str,
    lags: tuple[int, ...] | list[int] | None = None,
    incluir_snap: bool | None = None,
    alpha: float = 10.0,
) -> np.ndarray:
    """Função base que ajusta a regressão linear e constrói previsões para os dias escolhidos.

    Args:
        y_train (list | np.ndarray | pd.Series): Valores históricos conhecidos até a origem.
        datas_treino (pd.Series | pd.DatetimeIndex | list): Datas correspondentes a y_train.
        datas_prever (pd.Series | pd.DatetimeIndex | list): Datas futuras a serem previstas.
        covariaveis (pd.DataFrame): Dados de covariaveis.csv.
        serie_nome (str): Nome da série ('store_total', 'FOODS', 'HOBBIES').
        lags (tuple[int, ...] | list[int] | None): Lags autorregressivos a incluir.
        incluir_snap (bool | None): Se True, inclui snap_CA. Se None, True exceto para HOBBIES.
        alpha (float): Regularização L2 Ridge. Defaults to 10.0.

    Returns:
        np.ndarray: Previsões lineares para as datas em datas_prever.
    """
    from sklearn.linear_model import Ridge

    if incluir_snap is None:
        incluir_snap = (serie_nome != "HOBBIES")

    datas_treino = pd.to_datetime(pd.Series(datas_treino)).reset_index(drop=True)
    datas_prever = pd.to_datetime(pd.Series(datas_prever)).reset_index(drop=True)
    df_datas_all = pd.concat(
        [pd.DataFrame({"date": datas_treino}), pd.DataFrame({"date": datas_prever})],
        ignore_index=True,
    )

    # Padronização de preços estritamente com estatísticas do treino original (<= 2016-03-27)
    price_col = f"preco_{serie_nome}"
    mask_treino_orig = covariaveis["date"] <= pd.Timestamp("2016-03-27")
    p_train = covariaveis.loc[mask_treino_orig, price_col]
    p_mean = float(p_train.mean())
    p_std = float(p_train.std())

    exog = extrair_covariaveis_base(
        df_datas_all,
        covariaveis,
        serie_nome,
        p_mean,
        p_std,
        incluir_snap=incluir_snap,
    )

    n_tr = len(datas_treino)
    n_pr = len(datas_prever)
    all_y = list(y_train)

    if lags is None or len(lags) == 0:
        X_train = exog.iloc[:n_tr].values
        y_train_fit = np.asarray(all_y)
        X_pred = exog.iloc[n_tr : n_tr + n_pr].values
    else:
        max_lag = max(lags)
        X_train_rows = []
        y_train_rows = []
        for i in range(max_lag, n_tr):
            row = list(exog.iloc[i].values)
            for lag in lags:
                row.append(all_y[i - lag])
            X_train_rows.append(row)
            y_train_rows.append(all_y[i])
        X_train = np.array(X_train_rows)
        y_train_fit = np.array(y_train_rows)

        X_pred_rows = []
        for j in range(n_pr):
            idx = n_tr + j
            row = list(exog.iloc[idx].values)
            for lag in lags:
                row.append(all_y[idx - lag])
            X_pred_rows.append(row)
        X_pred = np.array(X_pred_rows)

    reg = Ridge(alpha=alpha).fit(X_train, y_train_fit)
    return reg.predict(X_pred)


def prever_linear_direto(
    serie: pd.Series,
    n: int = 28,
    covariaveis: pd.DataFrame | None = None,
    datas_treino: pd.Series | pd.DatetimeIndex | list | None = None,
    datas_prever: pd.Series | pd.DatetimeIndex | list | None = None,
    alpha: float = 10.0,
    incluir_snap: bool | None = None,
) -> np.ndarray:
    """Previsão linear direta em horizonte de n dias.

    Utiliza as variáveis determinísticas de calendário, eventos e covariáveis exógenas
    já conhecidas na data de origem para prever diretamente todo o horizonte em um único passo.

    Args:
        serie (pd.Series): Série temporal com histórico de treino.
        n (int): Horizonte de previsão (padrão 28 dias).
        covariaveis (pd.DataFrame | None): Dados de covariáveis.
        datas_treino (pd.Series | None): Datas do treino.
        datas_prever (pd.Series | None): Datas a serem previstas.
        alpha (float): Regularização L2 Ridge.
        incluir_snap (bool | None): Indicador de SNAP CA.

    Returns:
        np.ndarray: Previsões diretas para o horizonte.
    """
    if covariaveis is None:
        covariaveis = pd.read_csv("dados/covariaveis.csv", parse_dates=["date"])
    else:
        covariaveis = covariaveis.copy()
        covariaveis["date"] = pd.to_datetime(covariaveis["date"])

    serie_nome = serie.name if hasattr(serie, "name") and serie.name else "store_total"
    if datas_treino is None:
        if isinstance(serie.index, pd.DatetimeIndex):
            datas_treino = serie.index
        else:
            datas_treino = pd.date_range("2011-01-29", periods=len(serie), freq="D")

    if datas_prever is None:
        datas_prever = pd.date_range(datas_treino[-1] + pd.Timedelta(days=1), periods=n, freq="D")

    return base_previsao_linear(
        y_train=serie.values,
        datas_treino=datas_treino,
        datas_prever=datas_prever,
        covariaveis=covariaveis,
        serie_nome=serie_nome,
        lags=None,
        incluir_snap=incluir_snap,
        alpha=alpha,
    )


def prever_linear_recursivo(
    serie: pd.Series,
    n: int = 28,
    covariaveis: pd.DataFrame | None = None,
    datas_treino: pd.Series | pd.DatetimeIndex | list | None = None,
    datas_prever: pd.Series | pd.DatetimeIndex | list | None = None,
    lags: tuple[int, ...] | list[int] = (1, 2, 7, 14, 21, 28),
    alpha: float = 10.0,
    incluir_snap: bool | None = None,
) -> np.ndarray:
    """Previsão linear recursiva (autorregressiva com retreino / roll-forward passo a passo).

    Treina com a quantidade de tempo disponível, prevê mais um dia (t+1),
    adiciona esse dia previsto ao conjunto de treino e repete recursivamente.

    Args:
        serie (pd.Series): Série temporal com histórico de treino.
        n (int): Horizonte de previsão (padrão 28 dias).
        covariaveis (pd.DataFrame | None): Dados de covariáveis.
        datas_treino (pd.Series | None): Datas do treino.
        datas_prever (pd.Series | None): Datas a serem previstas.
        lags (tuple[int, ...] | list[int]): Lags autorregressivos.
        alpha (float): Regularização L2 Ridge.
        incluir_snap (bool | None): Indicador de SNAP CA.

    Returns:
        np.ndarray: Previsões recursivas para o horizonte.
    """
    if covariaveis is None:
        covariaveis = pd.read_csv("dados/covariaveis.csv", parse_dates=["date"])
    else:
        covariaveis = covariaveis.copy()
        covariaveis["date"] = pd.to_datetime(covariaveis["date"])

    serie_nome = serie.name if hasattr(serie, "name") and serie.name else "store_total"
    if datas_treino is None:
        if isinstance(serie.index, pd.DatetimeIndex):
            datas_treino = serie.index
        else:
            datas_treino = pd.date_range("2011-01-29", periods=len(serie), freq="D")

    if datas_prever is None:
        datas_prever = pd.date_range(datas_treino[-1] + pd.Timedelta(days=1), periods=n, freq="D")

    curr_y = list(serie.values)
    curr_datas = list(datas_treino)
    preds = []

    for step in range(n):
        proxima_data = datas_prever[step : step + 1]
        y_hat_step = base_previsao_linear(
            y_train=curr_y,
            datas_treino=curr_datas,
            datas_prever=proxima_data,
            covariaveis=covariaveis,
            serie_nome=serie_nome,
            lags=lags,
            incluir_snap=incluir_snap,
            alpha=alpha,
        )
        val_pred = float(y_hat_step[0])
        preds.append(val_pred)
        curr_y.append(val_pred)
        curr_datas.append(proxima_data[0])

    return np.array(preds)


def linear(
    serie: pd.Series,
    n: int = 28,
    estrategia: str = "recursiva",
    covariaveis: pd.DataFrame | None = None,
    **kwargs,
) -> np.ndarray:
    """Modelo linear exigido pela Task 2 para as três séries.

    Args:
        serie (pd.Series): Série histórica de treino.
        n (int): Horizonte de previsão (padrão 28 dias).
        estrategia (str): Estratégia de modelagem: 'recursiva' ou 'direta'. Padrão 'recursiva'.
        covariaveis (pd.DataFrame | None): Covariáveis exógenas.
        **kwargs: Argumentos adicionais repassados para a função de previsão.

    Returns:
        np.ndarray: Previsões geradas pelo modelo linear.
    """
    if estrategia == "recursiva":
        return prever_linear_recursivo(serie, n=n, covariaveis=covariaveis, **kwargs)
    elif estrategia == "direta":
        return prever_linear_direto(serie, n=n, covariaveis=covariaveis, **kwargs)
    else:
        raise ValueError(f"Estratégia inválida: '{estrategia}'. Escolha 'recursiva' ou 'direta'.")
