import pandas as pd


def verificar_tipo_data(df: pd.DataFrame) -> bool:
    """
    Verifica se todos os elementos da coluna data são do tipo datetime

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        bool: True se todos forem do tipo datetime, False se algum não for do tipo datetime.
    """
    return pd.api.types.is_datetime64_any_dtype(df["date"])


def verificar_ordenacao_por_data(df: pd.DataFrame) -> bool:
    """
    Verifica se os dados estão ordenados por data.

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        bool: True se estiverem ordenados, False caso contrário.
    """
    return df["date"].is_monotonic_increasing


def verificar_unicidade_por_data(df: pd.DataFrame) -> bool:
    """
    Verifica se cada data é única.

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        bool: True se as datas forem únicas. False caso contrário.
    """
    return df["date"].is_unique


def verificar_intervalo_data(
    df: pd.DataFrame, data_inicio: str = None, data_fim: str = None
) -> bool:
    """
    Verifica se todas as datas estão no intervalo determinado.
    Se data_inicio ou data_fim não forem fornecidos, o limite correspondente não é aplicado.

    Args:
        df (pd.DataFrame): Dataframe
        data_inicio (str, optional): Data de início. Padrão é None.
        data_fim (str, optional): Data de fim. Padrão é None.

    Returns:
        bool: True se todas as datas estiverem no intervalo. False caso contrário.
    """
    maior_que_inicio = True
    menor_que_fim = True

    if data_inicio is not None:
        maior_que_inicio = (df["date"] >= data_inicio).all()

    if data_fim is not None:
        menor_que_fim = (df["date"] <= data_fim).all()

    return maior_que_inicio & menor_que_fim


def verificar_vazamento(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Verifica se os dois dataframes não compartilham nenhuma data.

    Args:
        df1 (pd.DataFrame): Primeiro dataframe
        df2 (pd.DataFrame): Segundo dataframe

    Returns:
        bool: True se os períodos são disjuntos, ou seja, se NÃO há vazamento.
          False se alguma data aparece nos dois.
    """
    return set(df1["date"]).isdisjoint(df2["date"])


def verificar_consistencia(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    data_inicio1: str = None,
    data_fim1: str = None,
    data_inicio2: str = None,
    data_fim2: str = None,
):
    """
    Verifica se os dados são consistentes, ou seja, se são do tipo correto, ordenados por data,
    com datas únicas, no intervalo determinado e sem vazamentos.
    Se falhar em alguma verificação, avisa qual falhou.
    Se passar em todas as verificações, envia uma mensagem de sucesso.

    Args:
        df1 (pd.DataFrame): Primeiro dataframe
        df2 (pd.DataFrame): Segundo dataframe
        data_inicio1 (str, optional): Data de início do primeiro dataframe. Padrão é None.
        data_fim1 (str, optional): Data de fim do primeiro dataframe. Padrão é None.
        data_inicio2 (str, optional): Data de início do segundo dataframe. Padrão é None.
        data_fim2 (str, optional): Data de fim do segundo dataframe. Padrão é None.
    """
    tudo_ok = True
    if not (verificar_tipo_data(df1) and verificar_tipo_data(df2)):
        print("FALHA: Algum elemento não é do tipo datetime")
        tudo_ok = False

    if not (verificar_ordenacao_por_data(df1) and verificar_ordenacao_por_data(df2)):
        print("FALHA: Os dados não estão ordenados por data")
        tudo_ok = False

    if not (verificar_unicidade_por_data(df1) and verificar_unicidade_por_data(df2)):
        print("FALHA: Existe algum dia com dados repetidos")
        tudo_ok = False

    if not (
        verificar_intervalo_data(df1, data_inicio=data_inicio1, data_fim=data_fim1)
        and verificar_intervalo_data(df2, data_inicio=data_inicio2, data_fim=data_fim2)
    ):
        print("FALHA: Os dados não estão no intervalo correto")
        tudo_ok = False

    if not verificar_vazamento(df1, df2):
        print("FALHA: Há vazamento nos dados")
        tudo_ok = False

    if tudo_ok:
        print(
            "SUCESSO: Os dados estão consistentes (Do tipo correto, ordenados, com datas únicas, no intervalo correto e sem vazamentos)"
        )

def verificar_padronizacao_rede(X, y) -> bool:
    """
    Verifica se a média e o desvio guardados pela MLPTorch vêm só dos dados passados ao fit.

    Args:
        X: Matriz de covariáveis passada ao fit
        y: Alvo passado ao fit

    Returns:
        bool: True se mu/sd do objeto batem com os de X e y. False caso contrário.
    """
    
    import numpy as np
    import src.nets as nets
    
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    m = nets.MLPTorch(epocas=1).fit(X, y)
    
    sd_x = X.std(axis=0)
    sd_x[sd_x == 0] = 1.0
    
    return (
        np.allclose(m.mu_x_, X.mean(axis=0))
        and np.allclose(m.sd_x_, sd_x)
        and np.isclose(m.mu_y_, y.mean())
        and np.isclose(m.sd_y_, y.std())
    )
    
def verificar_fatia_es_rede(X, y, n_es: int = 28) -> bool:
    """
    Verifica se a fatia de early stopping da MLPTorch é a cauda dos dados do fit.

    Args:
        X: Matriz de covariáveis passada ao fit
        y: Alvo passado ao fit
        n_es (int, optional): Tamanho da fatia. Padrão é 28.

    Returns:
        bool: True se a fatia são as últimas n_es linhas e o treino é o resto. False caso contrário.
    """
    
    import numpy as np
    import src.nets as nets
    
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()

    capturado = {}
    original = nets._treinar
    
    def espiao(rede, 
               X_tr, y_tr, 
               X_es, y_es, 
               *args, **kwargs):
        
        capturado.update(X_tr=X_tr, X_es=X_es, y_es=y_es)
        
        return original(rede, X_tr, y_tr, X_es, y_es, *args, **kwargs)

    nets._treinar = espiao
    
    try:
        m = nets.MLPTorch(epocas=1, n_es=n_es).fit(X, y)
    finally:
        nets._treinar = original  # restaura mesmo se o fit quebrar

    cauda_x = (X[-n_es:] - m.mu_x_) / m.sd_x_
    cauda_y = (y[-n_es:] - m.mu_y_) / m.sd_y_
    return (
        len(capturado["X_es"]) == n_es
        and len(capturado["X_tr"]) + n_es == len(X)
        and np.allclose(capturado["X_es"], cauda_x)
        and np.allclose(capturado["y_es"], cauda_y)
    )