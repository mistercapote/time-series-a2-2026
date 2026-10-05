import numpy as np
import pandas as pd


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calcula o Erro Médio Absoluto (Mean Absolute Error - MAE).

    Args:
        y_true (np.ndarray | pd.Series): Valores reais observados.
        y_pred (np.ndarray | pd.Series): Valores previstos pelo modelo.

    Returns:
        float: Valor do MAE.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calcula a Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE).

    Args:
        y_true (np.ndarray | pd.Series): Valores reais observados.
        y_pred (np.ndarray | pd.Series): Valores previstos pelo modelo.

    Returns:
        float: Valor do RMSE.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mase(
    y_train: np.ndarray | pd.Series,
    y_true_val: np.ndarray | pd.Series,
    y_pred_val: np.ndarray | pd.Series,
    m: int = 7,
) -> float:
    """Calcula o Erro Médio Absoluto Escalonado (Mean Absolute Scaled Error - MASE)

    utilizando o benchmark naive sazonal in-sample como fator de escala.

    Args:
        y_train (np.ndarray | pd.Series): Série histórica de treino (in-sample).
        y_true_val (np.ndarray | pd.Series): Valores reais da validação
          (out-of-sample).
        y_pred_val (np.ndarray | pd.Series): Previsões na validação.
        m (int, optional): Período da sazonalidade (padrão semanal: 7). Defaults
          to 7.

    Returns:
        float: Valor do MASE.
    """
    y_tr = np.asarray(y_train)
    # Erro médio absoluto do naive sazonal in-sample: |y_t - y_{t-m}|
    mae_in_sample_seasonal = np.mean(np.abs(y_tr[m:] - y_tr[:-m]))

    mae_val = mae(y_true_val, y_pred_val)
    return float(mae_val / mae_in_sample_seasonal)


def mae_in_sample_sazonal(serie_train: pd.Series | np.ndarray, m: int = 7) -> float:
    """Calcula o MAE in-sample do modelo Naive Sazonal.

    Para t >= m, a previsão in-sample é y_pred_t = y_{t-m}.

    Args:
        serie_train (pd.Series | np.ndarray): Série temporal de treino.
        m (int): Período da sazonalidade. Defaults to 7.

    Returns:
        float: Valor do MAE in-sample do naive sazonal.
    """
    y = np.asarray(serie_train)
    erros = np.abs(y[m:] - y[:-m])
    return float(np.mean(erros))