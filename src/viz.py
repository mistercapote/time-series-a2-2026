import matplotlib.pyplot as plt
import pandas as pd

def grafico_temporal(df: pd.DataFrame, coluna: str, titulo: str=None, data_inicio: str=None, data_fim: str=None, y_max: int | float=None):
  """
  Plota o gráfico de uma coluna do dataframe ao longo do tempo. 
  É possível definir o intervalo de tempo e altura máxima do gráfico para controlar a visualização.

  Args:
      df (pd.DataFrame): Dataframe
      coluna (str): Coluna
      titulo (str, optional): Título do gráfico. Defaults to None.
      data_inicio (str, optional): Data de início da visualização. Defaults to None.
      data_fim (str, optional): Data de fim da visualização. Defaults to None.
      y_max (int | float, optional): Altura máxima do gráfico. Defaults to None.
  """
  # FILTRO

  df_filtrado = df.copy()

  if data_inicio is not None:
    df_filtrado = df_filtrado[df_filtrado['date'] >= data_inicio]

  if data_fim is not None:
      df_filtrado = df_filtrado[df_filtrado['date'] <= data_fim]

  # PLOT

  plt.plot(df_filtrado['date'],df_filtrado[coluna])

  if y_max is not None:
     plt.ylim(0,y_max)

  if titulo is not None:
    plt.title(titulo)

  plt.show()