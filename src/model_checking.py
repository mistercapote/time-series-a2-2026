import warnings
import pandas as pd
import src.metrics as metrics
import src.models as models
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

def comparar_sarima(
    modelos: list[tuple[tuple, tuple]],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame | None = None,
    labels: list[str] | None = None,
    maxiter: int = 200,
) -> pd.DataFrame:
    """
    Compara AIC e BIC no conjunto de treino para ordens candidatas do SARIMA.

    Args:
        modelos (list[tuple[tuple, tuple]]): Lista com os parâmetros de cada modelo.
        train_data (pd.DataFrame): Dados de treino.
        val_data (pd.DataFrame | None, optional): Mantido apenas por compatibilidade.
        labels (list[str] | None): Lista com os nomes das colunas.
        maxiter (int, optional): Número máximo de iterações da otimização. Defaults to 200.

    Returns:
        pd.DataFrame: Tabela com AIC e BIC calculados no treino.
    """
    resultados_grid = []
    for label in labels:
        y_tr = train_data[label]

        for order, s_order in modelos:
            mod = SARIMAX(y_tr, order=order, seasonal_order=s_order)
            fit = mod.fit(disp=False, maxiter=maxiter)

            resultados_grid.append({
                "Série": label,
                "Ordem SARIMA": f"{order} x {s_order}",
                "AIC": round(fit.aic, 1),
                "BIC": round(fit.bic, 1),
            })

    return pd.DataFrame(resultados_grid)


def comparar_sarimax(
    modelos: list[tuple[tuple, tuple]],
    tabela_treino: pd.DataFrame,
    labels: list[str],
    maxiter: int = 500,
) -> pd.DataFrame:
    """Compara AIC e BIC entre ordens candidatas do SARIMAX com covariáveis em cada série."""
    resultados_grid = []
    for label in labels:
        X_tr, y_tr, _, _, _ = models._obter_dados_serie(
            label, tabela_treino=tabela_treino, tabela_validacao=tabela_treino.iloc[-28:]
        )
        scaler = StandardScaler()
        X_tr_norm = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
        )
        for order, s_order in modelos:
            mod = SARIMAX(
                y_tr,
                exog=X_tr_norm,
                order=order,
                seasonal_order=s_order,
                mle_regression=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EstimationWarning)
                fit = mod.fit(disp=False, maxiter=maxiter, method="powell")
            resultados_grid.append({
                "Série": label,
                "Ordem SARIMAX": f"{order} x {s_order}",
                "AIC": round(fit.aic, 1),
                "BIC": round(fit.bic, 1),
            })

    return pd.DataFrame(resultados_grid)

def comparar_previsoes(modelos: dict, train_data: pd.DataFrame, datas_val: pd.Series, labels: list[str]) -> list[dict]:
    """
    Calcula a previsão de cada modelo e coloca tudo em uma tabela.

    Args:
        modelos (dict): Dicionário com os modelos.
        train_data (pd.DataFrame): Dados de treino.
        datas_val (pd.Series): Datas de cada valor previsto.
        labels (list[str]): Lista com os nomes das colunas.

    Returns:
        list[dict]: Tabela com as previsões.
    """
    registros = []
    n_val = len(datas_val)

    for label in labels:
        y_train = train_data[label]
        for modelo_nome, func_modelo in modelos.items():
            y_pred = func_modelo(y_train, n_val)
            for data_obs, pred in zip(datas_val, y_pred):
                registros.append({
                    "date": data_obs,
                    "series": label,
                    "modelo": modelo_nome,
                    "yhat": float(pred),
                })

    return registros

def comparar_metricas(
    modelos: dict,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    labels: list[str],
    registros: list[dict] | None = None,
) -> list[dict]:
    """
    Compara métricas entre modelos.

    Args:
        modelos (dict): Dicionário com os modelos.
        train_data (pd.DataFrame): Dados de treino.
        val_data (pd.DataFrame): Dados de validação.
        labels (list[str]): Lista com os nomes das colunas.
        registros (list[dict] | None, optional): Previsões já geradas por
            `comparar_previsoes`, evitando reajustar os modelos. Defaults to None.

    Returns:
        list[dict]: Tabela com as métricas.
    """
    resultados = []
    n_val = len(val_data)
    df_registros = pd.DataFrame(registros) if registros is not None else None

    for label in labels:
        y_train = train_data[label]
        y_val = val_data[label].values

        for nome_modelo, func_modelo in modelos.items():
            if df_registros is not None:
                filtro = (df_registros["series"] == label) & (
                    df_registros["modelo"] == nome_modelo
                )
                y_pred = df_registros.loc[filtro, "yhat"].to_numpy()
            else:
                # Gera a previsão out-of-sample na janela de validação
                y_pred = func_modelo(y_train, n_val)

            # Cálculo das métricas
            val_mae = metrics.mae(y_val, y_pred)
            val_rmse = metrics.rmse(y_val, y_pred)
            val_mase = metrics.mase(y_train, y_val, y_pred, m=7)

            resultados.append({
                "series": label,
                "modelo": nome_modelo,
                "mae": val_mae,
                "rmse": val_rmse,
                "mase": val_mase,
            })

    return resultados


def comparar_estrategias(
    candidatos: dict[str, object] | object,
    tabela_treino: pd.DataFrame,
    labels: list[str],
    n_val_interna: int = 28,
    incluir_referencias: bool = False,
    sarima_dict: dict | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    """Compara as estratégias 'direta' e 'recursiva' para qualquer modelo na validação interna (até 2016-03-27)."""
    if not isinstance(candidatos, dict):
        candidatos = {candidatos.__class__.__name__.lower(): candidatos}

    tab_tr_int = tabela_treino.iloc[:-n_val_interna].reset_index(drop=True)
    tab_val_int = tabela_treino.iloc[-n_val_interna:].reset_index(drop=True)

    linhas = []
    for label in labels:
        y_tr_int = tab_tr_int[label]
        y_val_int = tab_val_int[label].values

        if incluir_referencias:
            refs = {
                "media": models.mean(y_tr_int, n=n_val_interna),
                "naive": models.naive(y_tr_int, n=n_val_interna),
                "naive_sazonal": models.naive_sazonal(y_tr_int, m=7, n=n_val_interna),
                "drift": models.drift(y_tr_int, n=n_val_interna),
            }
            if sarima_dict is not None and label in sarima_dict:
                refs["sarima"] = models.sarima(
                    y_tr_int,
                    order=sarima_dict[label][0],
                    seasonal_order=sarima_dict[label][1],
                    n=n_val_interna,
                )
            for nome_ref, y_pred_ref in refs.items():
                linhas.append({
                    "series": label,
                    "candidato": nome_ref,
                    "estrategia": "referencia",
                    "mae": metrics.mae(y_val_int, y_pred_ref),
                    "rmse": metrics.rmse(y_val_int, y_pred_ref),
                    "mase": metrics.mase(y_tr_int, y_val_int, y_pred_ref, m=7),
                })

        for estrategia in ("direta", "recursiva"):
            for nome_cand, estimador in candidatos.items():
                y_pred = models.prever_modelo(
                    label,
                    n=n_val_interna,
                    tabela_treino=tab_tr_int,
                    tabela_validacao=tab_val_int,
                    modelo=estimador,
                    estrategia=estrategia,
                    **kwargs,
                )
                linhas.append({
                    "series": label,
                    "candidato": nome_cand,
                    "estrategia": estrategia,
                    "mae": metrics.mae(y_val_int, y_pred),
                    "rmse": metrics.rmse(y_val_int, y_pred),
                    "mase": metrics.mase(y_tr_int, y_val_int, y_pred, m=7),
                })

    df_busca = pd.DataFrame(linhas)
    df_modelos = df_busca[df_busca["estrategia"].isin(("direta", "recursiva"))]
    melhores = df_modelos.sort_values("mase").groupby("series", as_index=False).first()
    melhores_dict = {
        row["series"]: (row["candidato"], row["estrategia"])
        for _, row in melhores.iterrows()
    }
    return df_busca, melhores_dict


def selecionar_linear(
    tabela_treino: pd.DataFrame,
    labels: list[str],
    alphas: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0),
    n_val_interna: int = 28,
) -> tuple[pd.DataFrame, dict[str, tuple[str, float | None, str]]]:
    """Seleciona tipo ('padrao', 'ridge', 'lasso'), alpha e estratégia usando comparar_estrategias."""
    candidatos = {"padrao": models.criar_modelo_linear("padrao", None)}
    meta = {"padrao": ("padrao", None)}
    for tipo in ("ridge", "lasso"):
        for a in alphas:
            chave = f"{tipo}_{float(a)}"
            candidatos[chave] = models.criar_modelo_linear(tipo, float(a))
            meta[chave] = (tipo, float(a))

    df_busca, melhores_dict = comparar_estrategias(
        candidatos, tabela_treino, labels, n_val_interna=n_val_interna
    )
    df_busca["tipo"] = df_busca["candidato"].map(lambda c: meta[c][0])
    df_busca["alpha"] = df_busca["candidato"].map(lambda c: meta[c][1])
    df_busca = df_busca[["series", "tipo", "alpha", "estrategia", "mae", "rmse", "mase"]]

    linear_dict = {
        serie: (meta[cand][0], meta[cand][1], estr)
        for serie, (cand, estr) in melhores_dict.items()
    }
    return df_busca, linear_dict



def selecionar_arvore(
    tabela_treino: pd.DataFrame,
    labels: list[str],
    max_depths: tuple[int | None, ...] = (3, 5, 10, None),
    min_samples_leafs: tuple[int, ...] = (1, 5, 10),
    n_estimators_tuple: tuple[int, ...] = (50, 100),
    n_val_interna: int = 28,
) -> tuple[pd.DataFrame, dict[str, tuple[str, int | None, int, int, str]]]:
    """Seleciona tipo ('arvore', 'rf'), hiperparâmetros e estratégia usando comparar_estrategias."""
    candidatos = {}
    meta = {}

    for d in max_depths:
        for leaf in min_samples_leafs:
            # Árvore de decisão simples (n_estimators fixo em 100 apenas para manter a tupla padrão)
            chave_dt = f"arvore_d{d}_l{leaf}"
            candidatos[chave_dt] = models.criar_modelo_arvore(
                tipo="arvore", max_depth=d, min_samples_leaf=leaf, n_estimators=100
            )
            meta[chave_dt] = ("arvore", d, int(leaf), 100)

            # Random Forest
            for n_est in n_estimators_tuple:
                chave_rf = f"rf_d{d}_l{leaf}_n{n_est}"
                candidatos[chave_rf] = models.criar_modelo_arvore(
                    tipo="rf", max_depth=d, min_samples_leaf=leaf, n_estimators=int(n_est)
                )
                meta[chave_rf] = ("rf", d, int(leaf), int(n_est))

    df_busca, melhores_dict = comparar_estrategias(
        candidatos, tabela_treino, labels, n_val_interna=n_val_interna
    )
    df_busca["tipo"] = df_busca["candidato"].map(lambda c: meta[c][0])
    df_busca["max_depth"] = df_busca["candidato"].map(lambda c: meta[c][1])
    df_busca["min_samples_leaf"] = df_busca["candidato"].map(lambda c: meta[c][2])
    df_busca["n_estimators"] = df_busca["candidato"].map(lambda c: meta[c][3])
    df_busca = df_busca[
        ["series", "tipo", "max_depth", "min_samples_leaf", "n_estimators", "estrategia", "mae", "rmse", "mase"]
    ]

    arvore_dict = {
        serie: (meta[cand][0], meta[cand][1], meta[cand][2], meta[cand][3], estr)
        for serie, (cand, estr) in melhores_dict.items()
    }
    return df_busca, arvore_dict


def selecionar_boosting(
    tabela_treino: pd.DataFrame,
    labels: list[str],
    tipos: tuple[str, ...] = ("gbr", "hist", "ada"),
    n_estimators_tuple: tuple[int, ...] = (50, 100),
    learning_rates: tuple[float, ...] = (0.05, 0.1),
    max_depths: tuple[int | None, ...] = (3, 5),
    n_val_interna: int = 28,
) -> tuple[pd.DataFrame, dict[str, tuple[str, int, float, int | None, str]]]:
    """Seleciona tipo ('gbr', 'hist', 'ada'), hiperparâmetros e estratégia usando comparar_estrategias."""
    candidatos = {}
    meta = {}

    for tipo in tipos:
        for n_est in n_estimators_tuple:
            for lr in learning_rates:
                if tipo == "ada":
                    chave = f"{tipo}_n{n_est}_lr{lr}"
                    candidatos[chave] = models.criar_modelo_boosting(
                        tipo=tipo, n_estimators=int(n_est), learning_rate=float(lr), max_depth=None
                    )
                    meta[chave] = (tipo, int(n_est), float(lr), None)
                else:
                    for d in max_depths:
                        chave = f"{tipo}_n{n_est}_lr{lr}_d{d}"
                        candidatos[chave] = models.criar_modelo_boosting(
                            tipo=tipo, n_estimators=int(n_est), learning_rate=float(lr), max_depth=d
                        )
                        meta[chave] = (tipo, int(n_est), float(lr), d)

    df_busca, melhores_dict = comparar_estrategias(
        candidatos, tabela_treino, labels, n_val_interna=n_val_interna
    )
    df_busca["tipo"] = df_busca["candidato"].map(lambda c: meta[c][0])
    df_busca["n_estimators"] = df_busca["candidato"].map(lambda c: meta[c][1])
    df_busca["learning_rate"] = df_busca["candidato"].map(lambda c: meta[c][2])
    df_busca["max_depth"] = df_busca["candidato"].map(lambda c: meta[c][3])
    df_busca = df_busca[
        ["series", "tipo", "n_estimators", "learning_rate", "max_depth", "estrategia", "mae", "rmse", "mase"]
    ]

    boosting_dict = {
        serie: (meta[cand][0], meta[cand][1], meta[cand][2], meta[cand][3], estr)
        for serie, (cand, estr) in melhores_dict.items()
    }
    return df_busca, boosting_dict