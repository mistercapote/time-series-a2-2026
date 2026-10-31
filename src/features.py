import pandas as pd

import src.data as data

CAMINHO_COVARIAVEIS = "dados/covariaveis.csv"
CAMINHO_TREINO = "dados/treino.csv"
CAMINHO_VALIDACAO = "dados/validacao.csv"

FIM_TREINO = "2016-03-27"
INICIO_VALIDACAO = "2016-03-28"
FIM_VALIDACAO = "2016-04-24"

SERIES = ["store_total", "FOODS", "HOBBIES"]
DIAS_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]
TIPOS_EVENTO = ["Cultural", "National", "Religious", "Sporting"]
PRECOS = ["preco_store_total", "preco_FOODS", "preco_HOBBIES"]

# Grupos de colunas, para descartar um bloco inteiro de uma vez.
GRUPO_DIA_SEMANA = DIAS_SEMANA[1:]
GRUPO_NUMERICO = ["dia_do_mes", "mes", "ano"]
GRUPO_MES = [f"mes_{m}" for m in range(2, 13)]
GRUPO_EVENTO = [f"evento_{t.lower()}" for t in TIPOS_EVENTO] + ["natal", "vespera_natal"]


def carregar_covariaveis(caminho: str = "dados/covariaveis.csv") -> pd.DataFrame:
    """
    Carrega o arquivo de covariáveis com a coluna de data já convertida.

    Args:
        caminho (str): Caminho do arquivo de covariáveis.

    Returns:
        pd.DataFrame: Covariáveis ordenadas por data.
    """
    df = pd.read_csv(caminho, parse_dates=["date"])

    return df.sort_values("date").reset_index(drop=True)


def adicionar_calendario(X: pd.DataFrame, datas: pd.Series) -> pd.DataFrame:
    """
    Acrescenta as colunas que saem apenas da data.

    O dia da semana, o mês e o ano entram como indicadoras 0/1, cada bloco
    deixando a primeira categoria de fora para servir de base ao intercepto:
    segunda-feira, janeiro e o primeiro ano do arquivo. O mês e o ano entram
    também na forma numérica, para o caso de o grupo preferir usá-los como
    tendência; as duas formas são redundantes entre si, então o ajuste deve
    levar uma ou a outra, nunca as duas.

    Args:
        X (pd.DataFrame): Tabela de features em construção.
        datas (pd.Series): Coluna de datas.

    Returns:
        pd.DataFrame: Tabela com as colunas de calendário.
    """
    X = X.copy()
    dia_da_semana = datas.dt.dayofweek

    for i, nome in enumerate(DIAS_SEMANA):
        if i == 0:
            continue
        X[nome] = (dia_da_semana == i).astype(int)

    X["dia_do_mes"] = datas.dt.day
    X["mes"] = datas.dt.month
    X["ano"] = datas.dt.year

    for mes in range(2, 13):
        X[f"mes_{mes}"] = (datas.dt.month == mes).astype(int)

    for ano in sorted(datas.dt.year.unique())[1:]:
        X[f"ano_{ano}"] = (datas.dt.year == ano).astype(int)

    X["natal"] = ((datas.dt.month == 12) & (datas.dt.day == 25)).astype(int)
    X["vespera_natal"] = ((datas.dt.month == 12) & (datas.dt.day == 24)).astype(int)

    return X


def adicionar_eventos(X: pd.DataFrame, covariaveis: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta uma indicadora por tipo de evento.

    Um dia pode ter até dois eventos, então os dois campos de tipo entram na
    mesma indicadora.

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
        X[f"evento_{tipo.lower()}"] = (primeiro | segundo).astype(int)

    return X


def construir_features(caminho: str = "dados/covariaveis.csv") -> pd.DataFrame:
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
    X["snap"] = covariaveis["snap_CA"]
    X = adicionar_calendario(X, covariaveis["date"])
    X = adicionar_eventos(X, covariaveis)

    for coluna in PRECOS:
        X[coluna] = covariaveis[coluna]

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
    de grupo deste módulo: GRUPO_NUMERICO, GRUPO_MES, GRUPO_EVENTO,
    GRUPO_DIA_SEMANA e grupo_ano(X).

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


def grupo_ano(X: pd.DataFrame) -> list[str]:
    """
    Lista as indicadoras de ano presentes na tabela.

    Os nomes dependem dos anos do arquivo, então saem da própria tabela.

    Args:
        X (pd.DataFrame): Tabela de features.

    Returns:
        list[str]: Nomes das colunas indicadoras de ano.
    """
    return [c for c in X.columns if c.startswith("ano_")]


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
        tuple[pd.DataFrame, pd.DataFrame]: Tabela de treino e tabela de validação.
    """
    treino = montar_tabela(caminho_treino, caminho_covariaveis, data_fim=FIM_TREINO)
    validacao = montar_tabela(
        caminho_validacao,
        caminho_covariaveis,
        data_inicio=INICIO_VALIDACAO,
        data_fim=FIM_VALIDACAO,
    )

    return treino, validacao


if __name__ == "__main__":
    treino, validacao = montar_treino_e_validacao()

    print(f"treino:    {treino.shape[0]} linhas x {treino.shape[1]} colunas")
    print(f"validação: {validacao.shape[0]} linhas x {validacao.shape[1]} colunas")
    print()
    todas = colunas_do_modelo(treino, "FOODS")
    print(f"colunas de entrada por série: {len(todas)}")
    print()
    print("grupos que podem ser descartados:")
    print(f"  GRUPO_DIA_SEMANA  {GRUPO_DIA_SEMANA}")
    print(f"  GRUPO_NUMERICO    {GRUPO_NUMERICO}")
    print(f"  GRUPO_MES         {len(GRUPO_MES)} colunas, de mes_2 a mes_12")
    print(f"  grupo_ano(X)      {grupo_ano(treino)}")
    print(f"  GRUPO_EVENTO      {GRUPO_EVENTO}")
    print()
    print("exemplo de descarte — só dia da semana, snap e indicadoras de ano:")
    enxuto = colunas_do_modelo(
        treino, "FOODS", excluir=GRUPO_NUMERICO + GRUPO_MES + GRUPO_EVENTO + PRECOS
    )
    print(f"  {enxuto}")
