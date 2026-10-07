import numpy as np
import pandas as pd
import src.metrics as metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX

def comparar_sarima(modelos: list[tuple[tuple, tuple]], train_data: pd.DataFrame, val_data: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
  """
  Compara AIC, BIC e métricas de erro para modelos diferentes do SARIMA.

  Args:
      modelos (list[tuple[tuple, tuple]]): Lista com os parâmetros de cada modelo.
      train_data (pd.DataFrame): Dados de treino.
      val_data (pd.DataFrame): Dados de validação.
      labels (list[str]): Lista com os nomes das colunas.

  Returns:
      pd.DataFrame: Tabela com as métricas calculadas.
  """
  resultados_grid = []
  for label in labels:
      y_tr = train_data[label]
      y_v = val_data[label].values
      n_v = len(y_v)
      
      for order, s_order in modelos:
          mod = SARIMAX(y_tr, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False)
          fit = mod.fit(disp=False)
          pred = fit.get_forecast(steps=n_v).predicted_mean.values
          
          mae_v = metrics.mae(y_v, pred)
          rmse_v = metrics.rmse(y_v, pred)
          mase_v = metrics.mase(y_tr, y_v, pred)
          
          resultados_grid.append({
              "Série": label,
              "Ordem SARIMA": f"{order} x {s_order}",
              "AIC": round(fit.aic, 1),
              "BIC": round(fit.bic, 1),
              "MAE Validação": round(mae_v, 2),
              "RMSE Validação": round(rmse_v, 2),
              "MASE Validação": round(mase_v, 4),
          })

  return pd.DataFrame(resultados_grid)