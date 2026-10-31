import pandas as pd

import src.data as data

CAMINHO_COVARIAVEIS = "dados/covariaveis.csv"
CAMINHO_TREINO = "dados/treino.csv"
CAMINHO_VALIDACAO = "dados/validacao.csv"

FIM_TREINO = "2016-03-27"
INICIO_VALIDACAO = "2016-03-28"
FIM_VALIDACAO = "2016-04-24"

SERIES = ["store_total", "FOODS", "HOBBIES"]
PRECOS = [f"preco_{s}" for s in SERIES]
TIPOS_EVENTO = ["Cultural", "National", "Religious", "Sporting"]

# Grupos de colunas, para descartar um bloco inteiro de uma vez.
GRUPO_DIA_SEMANA = [f"dia_da_semana_{i}" for i in range(7)]
GRUPO_NUMERICO = ["semana_do_ano", "mes", "ano"]
GRUPO_EVENT_TYPE = [f"event_type_{t}" for t in TIPOS_EVENTO]


def carregar_covariaveis(caminho: str = CAMINHO_COVARIAVEIS) -> pd.DataFrame:
    """
    Carrega o arquivo de covariáveis com a coluna de data já convertida.

    Args:
        caminho (str): Caminho do arquivo de covariáveis.

    Returns:
        pd.DataFrame: Covariáveis ordenadas por data.
    """
    df = pd.read_csv(caminho, parse_dates=["date"])

    return df.sort_values("date").reset_index(drop=True)


def nomes_de_evento(covariaveis: pd.DataFrame) -> list[str]:
    """
    Lista os nomes de evento que aparecem no arquivo, em ordem alfabética.

    Os nomes saem dos dados em vez de ficarem fixos no código, para a tabela
    não quebrar se o arquivo de covariáveis mudar.

    Args:
        covariaveis (pd.DataFrame): Covariáveis carregadas.

    Returns:
        list[str]: Nomes de evento encontrados.
    """
    nomes = pd.concat([covariaveis["event_name_1"], covariaveis["event_name_2"]])

    return sorted(nomes.dropna().unique())


def adicionar_calendario(X: pd.DataFrame, datas: pd.Series) -> pd.DataFrame:
    """
    Acrescenta as colunas que saem apenas da data.

    A semana do ano, o mês e o ano entram na forma numérica. As indicadoras de
    dia da semana ficam em adicionar_dia_da_semana().

    Args:
        X (pd.DataFrame): Tabela de features em construção.
        datas (pd.Series): Coluna de datas.

    Returns:
        pd.DataFrame: Tabela com as colunas de calendário.
    """
    X = X.copy()

    X["semana_do_ano"] = datas.dt.isocalendar()["week"].astype(int)
    X["mes"] = datas.dt.month
    X["ano"] = datas.dt.year

    return X


def adicionar_dia_da_semana(X: pd.DataFrame, datas: pd.Series) -> pd.DataFrame:
    """
    Acrescenta sete indicadoras 0/1 de dia da semana, de segunda (0) a domingo (6).

    Atenção: as sete somam 1 em toda linha, então são colineares com o
    intercepto. O ajuste deve descartar uma delas ou rodar sem intercepto;
    ver GRUPO_DIA_SEMANA.

    Args:
        X (pd.DataFrame): Tabela de features em construção.
        datas (pd.Series): Coluna de datas.

    Returns:
        pd.DataFrame: Tabela com as indicadoras de dia da semana.
    """
    X = X.copy()
    dia_da_semana = datas.dt.dayofweek

    for i in range(7):
        X[f"dia_da_semana_{i}"] = (dia_da_semana == i).astype(int)

    return X


def adicionar_eventos(X: pd.DataFrame, covariaveis: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta uma indicadora por tipo de evento e uma por nome de evento.

    Um dia pode ter até dois eventos, então os dois campos do arquivo entram
    na mesma indicadora.

    Args:
        X (pd.DataFrame): Tabela de features em construção.
        covariaveis (pd.DataFrame): Covariáveis carregadas.

    Returns:
        pd.DataFrame: Tabela com as indicadoras de evento.
    """
    X = X.copy()

    for tipo in TIPOS_EVENTO:
        primeiro = covariaveis["event_type_1"] == tipo
        segundo = covariaveis["event_type_2"] == tipo
        X[f"event_type_{tipo}"] = (primeiro | segundo).astype(int)

    for nome in nomes_de_evento(covariaveis):
        primeiro = covariaveis["event_name_1"] == nome
        segundo = covariaveis["event_name_2"] == nome
        X[f"event_name_{nome}"] = (primeiro | segundo).astype(int)

    return X


def grupo_event_name(X: pd.DataFrame) -> list[str]:
    """
    Lista as indicadoras de nome de evento presentes na tabela.

    Os nomes dependem do arquivo, então saem da própria tabela.

    Args:
        X (pd.DataFrame): Tabela de features.

    Returns:
        list[str]: Nomes das colunas indicadoras de nome de evento.
    """
    return [c for c in X.columns if c.startswith("event_name_")]


def construir_features(caminho: str = CAMINHO_COVARIAVEIS) -> pd.DataFrame:
    """
    Monta a tabela de features para todas as datas do arquivo.

    Nenhuma coluna usa o valor vendido: tudo sai da data, do SNAP, dos eventos
    e do preço da semana, que já se conhecem na origem da previsão.

    Args:
        caminho (str): Caminho do arquivo de covariáveis.

    Returns:
        pd.DataFrame: Tabela com a coluna 'date' e as features.
    """
    covariaveis = carregar_covariaveis(caminho)

    X = pd.DataFrame({"date": covariaveis["date"]})
    X["snap_CA"] = covariaveis["snap_CA"]

    for coluna in PRECOS:
        X[coluna] = covariaveis[coluna]

    X = adicionar_calendario(X, covariaveis["date"])
    X = adicionar_eventos(X, covariaveis)
    X = adicionar_dia_da_semana(X, covariaveis["date"])

    return X


def cortar_periodo(
    X: pd.DataFrame, data_inicio: str = None, data_fim: str = None
) -> pd.DataFrame:
    """
    Recorta a tabela de features num intervalo de datas.

    Args:
        X (pd.DataFrame): Tabela de features.
        data_inicio (str): Data de início. Padrão é None.
        data_fim (str): Data de fim. Padrão é None.

    Returns:
        pd.DataFrame: Tabela no intervalo selecionado.
    """
    recorte = X.copy()

    if data_inicio is not None:
        recorte = recorte[recorte["date"] >= pd.Timestamp(data_inicio)]

    if data_fim is not None:
        recorte = recorte[recorte["date"] <= pd.Timestamp(data_fim)]

    return recorte.reset_index(drop=True)


def colunas_do_modelo(
    X: pd.DataFrame, serie: str, excluir: list[str] = None
) -> list[str]:
    """
    Lista as colunas que entram no ajuste de uma série.

    Sempre ficam de fora a data, as colunas de venda e o preço das outras duas
    categorias. O argumento 'excluir' descarta mais colunas, e aceita os nomes
    de grupo deste módulo: GRUPO_DIA_SEMANA, GRUPO_NUMERICO, GRUPO_EVENT_TYPE
    e grupo_event_name(X).

    Args:
        X (pd.DataFrame): Tabela de features.
        serie (str): Nome da série ('store_total', 'FOODS' ou 'HOBBIES').
        excluir (list[str]): Colunas a descartar além das obrigatórias.
          Padrão é None.

    Returns:
        list[str]: Nomes das colunas a usar no ajuste.
    """
    preco_da_serie = f"preco_{serie}"
    descartar = {"date"} | set(SERIES) | {p for p in PRECOS if p != preco_da_serie}

    if excluir is not None:
        descartar |= set(excluir)

    return [c for c in X.columns if c not in descartar]


def montar_tabela(
    caminho_y: str,
    caminho_covariaveis: str = CAMINHO_COVARIAVEIS,
    data_inicio: str = None,
    data_fim: str = None,
) -> pd.DataFrame:
    """
    Junta as vendas de um arquivo com as features das mesmas datas.

    O resultado tem uma linha por data, com as três séries e todas as colunas
    de entrada lado a lado. Se alguma data do arquivo de vendas não tiver
    covariável correspondente, levanta erro em vez de devolver linha vazia:
    desalinhamento silencioso casaria a venda de um dia com o calendário de
    outro.

    Args:
        caminho_y (str): Caminho do arquivo de vendas (treino ou validação).
        caminho_covariaveis (str): Caminho do arquivo de covariáveis.
        data_inicio (str): Data de início. Padrão é None.
        data_fim (str): Data de fim. Padrão é None.

    Returns:
        pd.DataFrame: Tabela completa, com as vendas e as features.
    """
    vendas = data.carregar_dados(caminho_y, data_inicio=data_inicio, data_fim=data_fim)
    X = construir_features(caminho_covariaveis)

    tabela = vendas.merge(X, on="date", how="left", validate="one_to_one")

    if tabela.isna().any(axis=None):
        datas_sem_covariavel = tabela.loc[tabela.isna().any(axis=1), "date"]
        raise ValueError(
            f"{len(datas_sem_covariavel)} data(s) do arquivo {caminho_y} ficaram sem "
            f"covariável, a começar por {datas_sem_covariavel.iloc[0].date()}."
        )

    return tabela.reset_index(drop=True)


def montar_treino_e_validacao(
    caminho_treino: str = CAMINHO_TREINO,
    caminho_validacao: str = CAMINHO_VALIDACAO,
    caminho_covariaveis: str = CAMINHO_COVARIAVEIS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monta as tabelas completas de treino e de validação.

    O recorte de datas é o da Task 1: o treino vai até 2016-03-27 e a validação
    cobre os 28 dias seguintes.

    Args:
        caminho_treino (str): Caminho do arquivo de treino.
        caminho_validacao (str): Caminho do arquivo de validação.
        caminho_covariaveis (str): Caminho do arquivo de covariáveis.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tabela de treino e de validação.
    """
    treino = montar_tabela(caminho_treino, caminho_covariaveis, data_fim=FIM_TREINO)
    validacao = montar_tabela(
        caminho_validacao,
        caminho_covariaveis,
        data_inicio=INICIO_VALIDACAO,
        data_fim=FIM_VALIDACAO,
    )

    return treino, validacao


def serie_completa(
    serie: str,
    periodo: str = "treino",
    excluir: list[str] = None,
    tabela: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Devolve a tabela de uma única série: a data, a venda dela e as entradas.

    Fica de fora tudo que pertence às outras séries, inclusive o preço delas.

    Args:
        serie (str): Nome da série ('store_total', 'FOODS' ou 'HOBBIES').
        periodo (str): 'treino' ou 'validacao'. Padrão é 'treino'.
        excluir (list[str]): Colunas de entrada a descartar, aceitando os
          grupos do módulo. Padrão é None.
        tabela (pd.DataFrame): Tabela já montada, para evitar reler os
          arquivos. Padrão é None.

    Returns:
        pd.DataFrame: Tabela com 'date', a coluna da série e as entradas dela.
    """
    if serie not in SERIES:
        raise ValueError(f"série desconhecida: {serie!r}. Use uma de {SERIES}.")

    if periodo not in ("treino", "validacao"):
        raise ValueError(
            f"período desconhecido: {periodo!r}. Use 'treino' ou 'validacao'."
        )

    if tabela is None:
        treino, validacao = montar_treino_e_validacao()
        tabela = treino if periodo == "treino" else validacao

    colunas = colunas_do_modelo(tabela, serie, excluir=excluir)

    return tabela[["date", serie] + colunas]


def separar_X_y(
    serie: str,
    periodo: str = "treino",
    excluir: list[str] = None,
    tabela: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Devolve as entradas e o alvo de uma série, prontos para o ajuste.

    A série devolvida mantém o próprio nome, que é como o run.py descobre de
    qual série se trata (serie.name).

    Args:
        serie (str): Nome da série ('store_total', 'FOODS' ou 'HOBBIES').
        periodo (str): 'treino' ou 'validacao'. Padrão é 'treino'.
        excluir (list[str]): Colunas de entrada a descartar. Padrão é None.
        tabela (pd.DataFrame): Tabela já montada. Padrão é None.

    Returns:
        tuple[pd.DataFrame, pd.Series]: As entradas e o alvo.
    """
    completa = serie_completa(serie, periodo, excluir=excluir, tabela=tabela)

    X = completa.drop(columns=["date", serie])
    y = completa[serie]

    return X, y


if __name__ == "__main__":
    treino, validacao = montar_treino_e_validacao()

    print(f"tabela larga -> treino {treino.shape}, validação {validacao.shape}")
    print()

    foods = serie_completa("FOODS", "treino", tabela=treino)
    print(f'serie_completa("FOODS", "treino") -> {foods.shape}')
    print(foods.columns.tolist())
    print()

    X, y = separar_X_y("FOODS", "treino", tabela=treino)
    print(f'separar_X_y("FOODS") -> X {X.shape}, y {y.shape} chamado "{y.name}"')
    print()

    print("grupos que podem ser descartados:")
    print(f"  GRUPO_DIA_SEMANA       {len(GRUPO_DIA_SEMANA)} colunas")
    print(f"  GRUPO_NUMERICO         {GRUPO_NUMERICO}")
    print(f"  GRUPO_EVENT_TYPE       {GRUPO_EVENT_TYPE}")
    print(f"  grupo_event_name(X)    {len(grupo_event_name(treino))} colunas")
    print(f"  PRECOS                 {PRECOS}")
