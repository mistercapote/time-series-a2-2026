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
    Verifica se existem datas de um dataframe em outro (vazamento de dados).

    Args:
        df1 (pd.DataFrame): Primeiro dataframe
        df2 (pd.DataFrame): Segundo dataframe

    Returns:
        bool: True se tiver vazamento de dados. False caso contrário.
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
