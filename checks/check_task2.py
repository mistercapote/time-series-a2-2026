"""Checks públicos da Task 2 — crédito parcial por critério (total 10)."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

SERIES = ("store_total", "FOODS", "HOBBIES")
TREE = ("boosting", "arvore")
NET = ("rede", "lstm", "rnn", "transformer")
ARIMA_NAMES = ("arima", "sarima")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")

PESOS = {
    "split": 2.0,
    "metricas": 2.0,
    "linear": 1.5,
    "arvore": 1.5,
    "rede": 1.5,
    "barras": 0.5,
    "repo": 0.5,
    "ai_usage": 0.5,
}

ATES = ("linear", "arvore", "rede")
PRAZO_FUTURO = {
    "arvore": "prazo 08/11",
    "rede": "prazo 15/11",
}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _norm_model(name: str) -> str:
    return _fold(name).strip().replace(" ", "_")


def _prepare(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if "modelo" in out.columns:
        out["modelo"] = out["modelo"].map(_norm_model)
    if "series" in out.columns:
        out["series"] = out["series"].astype(str)
    return out


def check_split(prev: pd.DataFrame | None, holdout_dates: set[pd.Timestamp]) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    need = {"date", "series", "modelo", "yhat"}
    if not need.issubset(prev.columns):
        return 0.0, f"colunas obrigatórias ausentes: {sorted(need - set(prev.columns))}"
    dates = pd.to_datetime(prev["date"])
    if (dates < VAL_START).any() or (dates > VAL_END).any():
        return 0.0, "há previsão fora de 2016-03-28–2016-04-24"
    if any(d in holdout_dates for d in dates):
        return 0.0, "há previsão no horizonte do holdout"
    n_val = ((dates >= VAL_START) & (dates <= VAL_END)).sum()
    if n_val == 0:
        return 0.0, "nenhuma previsão no horizonte de validação"
    return PESOS["split"], "previsões só na validação conhecida; holdout intocado"


def _modelos(df: pd.DataFrame | None, series: str) -> set[str]:
    if df is None or "modelo" not in df.columns or "series" not in df.columns:
        return set()
    sub = df.loc[df["series"] == series, "modelo"]
    return set(sub.astype(str))


def _classes_devidas(ate: str) -> set[str]:
    if ate not in ATES:
        raise ValueError(f"ate inválido: {ate}")
    corte = ATES.index(ate)
    return set(ATES[: corte + 1])


def _em_aberto(ate: str) -> set[str]:
    return set(ATES[ATES.index(ate) + 1 :])


def _exigidos(nomes: set[str], ate: str = "rede") -> tuple[set[str], list[str]]:
    faltas: list[str] = []
    chosen: set[str] = set()
    devidas = _classes_devidas(ate)
    if "naive_sazonal" not in nomes:
        faltas.append("naive_sazonal")
    else:
        chosen.add("naive_sazonal")
    if "linear" in devidas:
        if "linear" not in nomes:
            faltas.append("linear")
        else:
            chosen.add("linear")
    if "arvore" in devidas:
        trees = nomes & set(TREE)
        if not trees:
            faltas.append("boosting ou arvore")
        else:
            chosen |= trees
    if "rede" in devidas:
        nets = nomes & set(NET)
        if not nets:
            faltas.append("rede")
        else:
            chosen |= nets
    arimas = nomes & set(ARIMA_NAMES)
    if not arimas:
        faltas.append("sarima ou arima")
    else:
        chosen |= arimas
    return chosen, faltas


def check_linear(prev: pd.DataFrame | None, metricas: pd.DataFrame | None) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    if metricas is None:
        return 0.0, "falta metricas.csv"
    for s in SERIES:
        if "linear" not in _modelos(prev, s) or "linear" not in _modelos(metricas, s):
            return 0.0, f"falta linear em {s}"
    return PESOS["linear"], "linear nas três séries"


def check_arvore(prev: pd.DataFrame | None, metricas: pd.DataFrame | None) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    if metricas is None:
        return 0.0, "falta metricas.csv"
    for s in SERIES:
        prev_ok = _modelos(prev, s) & set(TREE)
        met_ok = _modelos(metricas, s) & set(TREE)
        if not prev_ok or not met_ok:
            return 0.0, f"falta boosting ou arvore em {s}"
    return PESOS["arvore"], "boosting ou arvore nas três séries"


def check_rede(prev: pd.DataFrame | None, metricas: pd.DataFrame | None) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    if metricas is None:
        return 0.0, "falta metricas.csv"
    for s in SERIES:
        if not (_modelos(prev, s) & set(NET)) or not (_modelos(metricas, s) & set(NET)):
            return 0.0, f"falta rede em {s}"
    return PESOS["rede"], "rede nas três séries"


def check_barras(prev: pd.DataFrame | None, metricas: pd.DataFrame | None) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    if metricas is None:
        return 0.0, "falta metricas.csv"
    for s in SERIES:
        for frame, onde in ((prev, "previsões"), (metricas, "métricas")):
            nomes = _modelos(frame, s)
            if "naive_sazonal" not in nomes or not (nomes & set(ARIMA_NAMES)):
                return 0.0, f"falta naive sazonal ou sarima nas {onde} de {s}"
    return PESOS["barras"], "naive sazonal e SARIMA nas três séries"


def _mase_scale(train: pd.DataFrame, col: str, period: int = 7) -> float:
    y = train[col].to_numpy()
    diffs = abs(y[period:] - y[:-period])
    return float(diffs.mean())


def check_metricas(
    metricas: pd.DataFrame | None,
    prev: pd.DataFrame | None,
    val: pd.DataFrame,
    train: pd.DataFrame,
    ate: str = "rede",
) -> tuple[float, str]:
    if metricas is None:
        return 0.0, "falta metricas.csv"
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    need = {"series", "modelo", "mae", "rmse", "mase"}
    if not need.issubset(metricas.columns):
        return 0.0, f"colunas ausentes em metricas.csv: {sorted(need - set(metricas.columns))}"
    prev_m = prev.copy()
    prev_m["date"] = pd.to_datetime(prev_m["date"])
    val = val.copy()
    val["date"] = pd.to_datetime(val["date"])
    max_rel = 0.0
    n_checked = 0
    for s in SERIES:
        presentes = _modelos(prev, s) & _modelos(metricas, s)
        _, faltas = _exigidos(presentes, ate)
        if faltas:
            return 0.0, f"métricas incompletas em {s}: {', '.join(faltas)}"
        chosen, _ = _exigidos(presentes, ate)
        scale = _mase_scale(train, s)
        y = val.set_index("date")[s]
        for modelo in sorted(chosen):
            g = prev_m[(prev_m["series"] == s) & (prev_m["modelo"] == modelo)]
            if g.empty:
                return 0.0, f"previsão ausente ({s}/{modelo})"
            if g["date"].duplicated().any():
                return 0.0, f"datas duplicadas ({s}/{modelo})"
            yhat = g.set_index("date")["yhat"].reindex(y.index)
            if yhat.isna().any():
                return 0.0, f"previsão incompleta no horizonte ({s}/{modelo})"
            err = pd.to_numeric(yhat, errors="coerce").to_numpy() - y.to_numpy()
            if pd.isna(err).any():
                return 0.0, f"yhat não numérico ({s}/{modelo})"
            mae = float(abs(err).mean())
            rmse = float((err**2).mean() ** 0.5)
            mase = mae / scale if scale else float("nan")
            row = metricas[(metricas["series"] == s) & (metricas["modelo"] == modelo)]
            if row.empty:
                return 0.0, f"métrica ausente ({s}/{modelo})"
            for nome, calc in ("mae", mae), ("rmse", rmse), ("mase", mase):
                reported = pd.to_numeric(row.iloc[0][nome], errors="coerce")
                if pd.isna(reported):
                    return 0.0, f"{nome} inválido ({s}/{modelo})"
                denom = max(abs(calc), 1e-6)
                max_rel = max(max_rel, abs(float(reported) - calc) / denom)
                n_checked += 1
    if n_checked == 0:
        return 0.0, "não foi possível cruzar métricas com previsões"
    if max_rel > 0.05:
        return 0.0, f"métricas reportadas divergem do recálculo (erro relativo até {max_rel:.0%})"
    return PESOS["metricas"], "MAE, RMSE e MASE conferidos na validação conhecida"


def check_repo(root: Path) -> tuple[float, str]:
    readme = root / "README.md"
    if not readme.is_file() or readme.stat().st_size < 20:
        return 0.0, "falta README.md descrevendo como executar"
    deps = any((root / n).is_file() for n in ("requirements.txt", "pyproject.toml", "environment.yml"))
    code = list(root.glob("*.py")) + list(root.glob("*.ipynb"))
    code = [p for p in code if "check_task2" not in p.name]
    if not deps:
        return 0.0, "falta requirements.txt (ou pyproject/environment)"
    if not code:
        return 0.0, "falta script ou notebook na raiz do repo"
    return PESOS["repo"], "README + dependências + código para reproduzir"


def check_ai_usage(root: Path, ate: str = "rede") -> tuple[float, str]:
    path = root / "AI_USAGE.md"
    if not path.is_file():
        return 0.0, "falta AI_USAGE.md"
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) < 80:
        return 0.0, "AI_USAGE.md vazio ou curto demais"
    low = _fold(text)
    tem_ferramenta = ("ferramenta" in low) or ("nao usamos ia" in low)
    tem_prompt = "prompt" in low
    tem_erro = "erro" in low
    tem_resp = "responsab" in low
    if not (tem_ferramenta and tem_prompt and tem_erro and tem_resp):
        return 0.0, "AI_USAGE.md sem os itens mínimos (ferramentas, prompts, erros, responsabilidade)"
    devidas = _classes_devidas(ate)
    faltas: list[str] = []
    if "linear" in devidas and "linear" not in low:
        faltas.append("linear")
    if "arvore" in devidas and not any(w in low for w in ("boosting", "arvore", "floresta")):
        faltas.append("árvore")
    if "rede" in devidas and not any(w in low for w in NET):
        faltas.append("rede")
    if faltas:
        return 0.0, "AI_USAGE.md não cobre " + ", ".join(faltas)
    if ate == "rede":
        return PESOS["ai_usage"], "AI_USAGE.md presente e cobre a Task 2"
    return PESOS["ai_usage"], f"AI_USAGE.md cobre o que já venceu ({ate})"


def evaluate(submission: Path, dados: Path, ate: str = "rede") -> dict:
    prev = _prepare(_read_csv(submission / "previsoes_validacao.csv"))
    metricas = _prepare(_read_csv(submission / "metricas.csv"))
    val = pd.read_csv(dados / "validacao.csv", parse_dates=["date"])
    train = pd.read_csv(dados / "treino.csv", parse_dates=["date"])
    holdout = pd.read_csv(dados / "holdout_datas.csv", parse_dates=["date"])
    holdout_dates = set(holdout["date"])
    em_aberto = _em_aberto(ate)

    resultados = {
        "split": check_split(prev, holdout_dates),
        "metricas": check_metricas(metricas, prev, val, train, ate),
        "linear": check_linear(prev, metricas),
        "arvore": check_arvore(prev, metricas),
        "rede": check_rede(prev, metricas),
        "barras": check_barras(prev, metricas),
        "repo": check_repo(submission),
        "ai_usage": check_ai_usage(submission, ate),
    }
    criterios: dict[str, dict] = {}
    nota = 0.0
    maximo = 0.0
    for k, (pontos, detalhe) in resultados.items():
        if k in em_aberto:
            criterios[k] = {
                "pontos": None,
                "max": PESOS[k],
                "detalhe": PRAZO_FUTURO[k],
                "conta": False,
            }
            continue
        criterios[k] = {
            "pontos": round(pontos, 2),
            "max": PESOS[k],
            "detalhe": detalhe,
            "conta": True,
        }
        nota += pontos
        maximo += PESOS[k]
    return {
        "nota": round(nota, 2),
        "maximo": maximo,
        "ate": ate,
        "criterios": criterios,
    }


def main() -> None:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Checks da Task 2 (crédito parcial).")
    parser.add_argument("--submission", type=Path, required=True, help="raiz do repo do grupo")
    parser.add_argument(
        "--dados",
        type=Path,
        default=here.parents[1] / "dados",
        help="pasta pública com treino, validacao, holdout_datas e covariaveis",
    )
    parser.add_argument(
        "--ate",
        choices=ATES,
        default="rede",
        help="até qual classe cobrar: linear (01/11), arvore (08/11) ou rede (15/11, default)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.dados.is_dir():
        raise SystemExit(f"pasta de dados não encontrada: {args.dados}")
    rel = evaluate(args.submission.resolve(), args.dados.resolve(), args.ate)
    if args.json:
        json.dump(rel, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    titulo = "Task 2" if args.ate == "rede" else f"Task 2 até {args.ate}"
    maximo = rel["maximo"]
    maximo_txt = f"{maximo:.0f}" if abs(maximo - round(maximo)) < 1e-9 else f"{maximo:.1f}"
    print(f"{titulo}  {rel['nota']:.1f} / {maximo_txt}")
    for nome, c in rel["criterios"].items():
        if c["pontos"] is None:
            print(f"  {nome:12} —        {c['detalhe']}")
        else:
            print(f"  {nome:12} {c['pontos']:.1f}/{c['max']:.1f}  {c['detalhe']}")


if __name__ == "__main__":
    main()
