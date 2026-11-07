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


def selecionar_linear(
    tabela_treino: pd.DataFrame,
    labels: list[str],
    alphas: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0),
    n_val_interna: int = 28,
) -> tuple[pd.DataFrame, dict[str, tuple[str, float | None, str]]]:
    """Seleciona tipo ('padrao', 'ridge', 'lasso'), alpha e estratégia na validação interna (até 2016-03-27)."""
    tab_tr_int = tabela_treino.iloc[:-n_val_interna].reset_index(drop=True)
    tab_val_int = tabela_treino.iloc[-n_val_interna:].reset_index(drop=True)

    candidatos = [("padrao", None)]
    for tipo in ("ridge", "lasso"):
        for a in alphas:
            candidatos.append((tipo, float(a)))

    linhas = []
    for label in labels:
        y_tr_int = tab_tr_int[label]
        y_val_int = tab_val_int[label].values
        for estrategia in ("direta", "recursiva"):
            for tipo, alpha in candidatos:
                y_pred = models.linear(
                    label,
                    n=n_val_interna,
                    tabela_treino=tab_tr_int,
                    tabela_validacao=tab_val_int,
                    tipo=tipo,
                    alpha=alpha,
                    estrategia=estrategia,
                )
                linhas.append({
                    "series": label,
                    "tipo": tipo,
                    "alpha": alpha,
                    "estrategia": estrategia,
                    "mae": metrics.mae(y_val_int, y_pred),
                    "rmse": metrics.rmse(y_val_int, y_pred),
                    "mase": metrics.mase(y_tr_int, y_val_int, y_pred, m=7),
                })

    df_busca = pd.DataFrame(linhas)
    melhores = df_busca.sort_values("mase").groupby("series", as_index=False).first()
    linear_dict = {
        row["series"]: (
            row["tipo"],
            None if pd.isna(row["alpha"]) else float(row["alpha"]),
            row["estrategia"],
        )
        for _, row in melhores.iterrows()
    }
    return df_busca, linear_dict