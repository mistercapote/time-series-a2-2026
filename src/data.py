import pandas as pd


def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega os dados no caminho dado.

    Args:
        caminho (str): Caminho dos dados

    Returns:
        pd.DataFrame: Dados
    """
    return pd.read_csv(caminho)


def mudar_tipo_coluna_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Altera a coluna data para o tipo datetime (melhor para operações de data).

    Args:
        df (pd.DataFrame): Dataframe original

    Returns:
        pd.DataFrame: Dataframe modificado
    """
    df_modificado = df.copy()
    df_modificado["date"] = pd.to_datetime(df_modificado["date"])

    return df_modificado


def ordenar_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena as linhas de uma tabela pela coluna data (em ordem crescente).

    Args:
        df (pd.DataFrame): Dataframe original

    Returns:
        pd.DataFrame: Dataframe ordenado
    """
    df_ordenado = df.copy()
    df_ordenado = df_ordenado.sort_values("date")

    return df_ordenado


def filtrar_intervalo_data(
    serie: pd.Series, datas: pd.Series, data_inicio: str = None, data_fim: str = None
) -> pd.DataFrame:
    """
    Filtra as entradas de uma tabela para estarem apenas entre duas datas selecionadas.
    Se data_inicio ou data_fim não forem fornecidos, o limite correspondente não é aplicado.

    Args:
        df (pd.DataFrame): Dataframe original
        data_inicio (str): Data de início. Padrão é None
        data_fim (str): Data de fim. Padrão é None

    Returns:
        pd.DataFrame: Dataframe no intervalo selecionado
    """
    df_verificado = pd.DataFrame({"date": datas, "value": serie}).copy()

    # Se data_inicio foi preenchida, filtra do início em diante
    if data_inicio is not None:
        df_verificado = df_verificado[df_verificado["date"] >= data_inicio]

    # Se data_fim foi preenchida, filtra até a data de fim
    if data_fim is not None:
        df_verificado = df_verificado[df_verificado["date"] <= data_fim]

    return df_verificado["date"], df_verificado["value"]


def carregar_dados_e_tratar_data(
    caminho: str, data_inicio: str = None, data_fim: str = None
) -> pd.DataFrame:
    """
    Faz o pipeline completo de carregamento dos dados, converter data para datetime,
    ordenar pela data e colocar as linhas dentro de um intervalo especifico.

    Args:
        caminho (str): Caminho dos dados
        data_inicio (str): Data de início. Padrão é None.
        data_fim (str): Data de fim. Padrão é None.

    Returns:
        pd.Dataframe: Dados carregados e tratados
    """
    df = carregar_dados(caminho)
    df = mudar_tipo_coluna_data(df)
    df = ordenar_data(df)

    if data_inicio is not None:
        df = df[df["date"] >= pd.Timestamp(data_inicio)]

    if data_fim is not None:
        df = df[df["date"] <= pd.Timestamp(data_fim)]

    return df


def construir_full_data(
    train_data:pd.DataFrame, validation_data:pd.DataFrame
)-> pd.DataFrame:
    """
    Concatena o conjunto de treino e o conjunto de validação num único DataFrame
    
    Args:
            train_data (str): DataFrame de treino.
            validation_data (str): DataFrame de validação.
    
    Returns:
        pd.Dataframe: Dados concatenados
    """
    return pd.concat([train_data, validation_data], axis=0)

