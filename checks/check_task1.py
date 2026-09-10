"""Checks públicos da Task 1 — crédito parcial por critério (total 10)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SERIES = ("store_total", "FOODS", "HOBBIES")
BASELINES = ("media", "naive", "naive_sazonal", "drift")
ARIMA_NAMES = ("arima", "sarima")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")
HOLDOUT_START = pd.Timestamp("2016-04-25")

PESOS = {
    "split": 2.5,
    "metricas": 2.0,
    "sarima": 2.0,
    "baselines": 1.5,
    "repo": 1.0,
    "diagnostico": 0.5,
    "ai_usage": 0.5,
}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _norm_model(name: str) -> str:
    return str(name).strip().lower().replace("í", "i").replace(" ", "_")


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


def _combos(df: pd.DataFrame, models: tuple[str, ...]) -> set[tuple[str, str]]:
    m = df.copy()
    m["modelo"] = m["modelo"].map(_norm_model)
    m["series"] = m["series"].astype(str)
    have = set(zip(m["series"], m["modelo"], strict=True))
    want = {(s, mod) for s in SERIES for mod in models}
    return want - have


def check_baselines(prev: pd.DataFrame | None) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    missing = _combos(prev, BASELINES)
    if missing:
        return 0.0, "faltam baselines: " + ", ".join(f"{s}/{m}" for s, m in sorted(missing)[:8])
    return PESOS["baselines"], "média, naive, naive sazonal e drift nas três séries"


def check_sarima(prev: pd.DataFrame | None, metricas: pd.DataFrame | None) -> tuple[float, str]:
    if prev is None:
        return 0.0, "falta previsoes_validacao.csv"
    prev_m = prev.copy()
    prev_m["modelo"] = prev_m["modelo"].map(_norm_model)
    ok_prev = True
    for s in SERIES:
        sub = prev_m.loc[prev_m["series"].astype(str) == s, "modelo"]
        if not set(sub) & set(ARIMA_NAMES):
            ok_prev = False
    if not ok_prev:
        return 0.0, "falta previsão arima/sarima em alguma série"
    if metricas is None:
        return 0.0, "falta metricas.csv para o ARIMA/SARIMA"
    met = metricas.copy()
    met["modelo"] = met["modelo"].map(_norm_model)
    for s in SERIES:
        sub = met.loc[met["series"].astype(str) == s, "modelo"]
        if not set(sub) & set(ARIMA_NAMES):
            return 0.0, "falta linha de métrica arima/sarima em alguma série"
    return PESOS["sarima"], "ARIMA/SARIMA presente nas três séries"


def _mase_scale(train: pd.DataFrame, col: str, period: int = 7) -> float:
    y = train[col].to_numpy()
    diffs = abs(y[period:] - y[:-period])
    return float(diffs.mean())


def check_metricas(
    metricas: pd.DataFrame | None,
    prev: pd.DataFrame | None,
    val: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[float, str]:
    if metricas is None:
        return 0.0, "falta metricas.csv"
    need = {"series", "modelo", "mae", "rmse", "mase"}
    if not need.issubset(metricas.columns):
        return 0.0, f"colunas ausentes em metricas.csv: {sorted(need - set(metricas.columns))}"
    met = metricas.copy()
    met["modelo"] = met["modelo"].map(_norm_model)
    for s in SERIES:
        modelos = set(met.loc[met["series"].astype(str) == s, "modelo"])
        if any(b not in modelos for b in BASELINES):
            return 0.0, f"métricas incompletas dos baselines em {s}"
        if not modelos & set(ARIMA_NAMES):
            return 0.0, f"métricas sem arima/sarima em {s}"
        for col in ("mae", "rmse", "mase"):
            vals = pd.to_numeric(met.loc[met["series"].astype(str) == s, col], errors="coerce")
            if vals.isna().any() or (vals < 0).any():
                return 0.0, f"há {col} inválido em {s}"
    if prev is None:
        return PESOS["metricas"] * 0.5, "métricas reportadas, mas sem previsões para conferir"

    prev_m = prev.copy()
    prev_m["date"] = pd.to_datetime(prev_m["date"])
    prev_m["modelo"] = prev_m["modelo"].map(_norm_model)
    val = val.copy()
    val["date"] = pd.to_datetime(val["date"])
    max_rel = 0.0
    n_checked = 0
    for s in SERIES:
        scale = _mase_scale(train, s)
        y = val.set_index("date")[s]
        for modelo in list(BASELINES) + [m for m in ARIMA_NAMES if m in set(prev_m["modelo"])]:
            g = prev_m[(prev_m["series"].astype(str) == s) & (prev_m["modelo"] == modelo)]
            if g.empty:
                continue
            g = g.set_index("date")["yhat"].reindex(y.index)
            if g.isna().any():
                return 0.0, f"previsão incompleta no horizonte ({s}/{modelo})"
            err = g.to_numpy() - y.to_numpy()
            mae = float(abs(err).mean())
            rmse = float((err**2).mean() ** 0.5)
            mase = mae / scale if scale else float("nan")
            row = met[(met["series"].astype(str) == s) & (met["modelo"] == modelo)]
            if row.empty:
                continue
            for nome, calc in ("mae", mae), ("rmse", rmse), ("mase", mase):
                reported = float(row.iloc[0][nome])
                denom = max(abs(calc), 1e-6)
                max_rel = max(max_rel, abs(reported - calc) / denom)
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
    code = [p for p in code if "check_task1" not in p.name]
    if not deps:
        return 0.0, "falta requirements.txt (ou pyproject/environment)"
    if not code:
        return 0.0, "falta script ou notebook na raiz do repo"
    return PESOS["repo"], "README + dependências + código para reproduzir"


def check_diagnostico(root: Path) -> tuple[float, str]:
    figs = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}
        and ".git" not in p.parts
    ]
    if len(figs) >= 3:
        return PESOS["diagnostico"], f"{len(figs)} figuras encontradas"
    texts = []
    for p in list(root.glob("*.ipynb")) + list(root.glob("*.md")):
        if p.name in {"README.md", "enunciado.md"}:
            continue
        texts.append(p.read_text(encoding="utf-8", errors="ignore").lower())
    blob = "\n".join(texts)
    if "acf" in blob and "pacf" in blob:
        return PESOS["diagnostico"], "ACF/PACF descritos no notebook ou notes"
    return 0.0, "faltam figuras ou ACF/PACF das três séries"


def check_ai_usage(root: Path) -> tuple[float, str]:
    path = root / "AI_USAGE.md"
    if not path.is_file():
        return 0.0, "falta AI_USAGE.md"
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) < 80:
        return 0.0, "AI_USAGE.md vazio ou curto demais"
    low = text.lower()
    tem_ferramenta = ("ferramenta" in low) or ("não usamos ia" in low) or ("nao usamos ia" in low)
    tem_prompt = "prompt" in low
    tem_erro = "erro" in low
    tem_resp = "responsab" in low
    if not (tem_ferramenta and tem_prompt and tem_erro and tem_resp):
        return 0.0, "AI_USAGE.md sem os itens mínimos (ferramentas, prompts, erros, responsabilidade)"
    return PESOS["ai_usage"], "AI_USAGE.md presente e no formato mínimo"


def evaluate(submission: Path, dados: Path) -> dict:
    prev = _read_csv(submission / "previsoes_validacao.csv")
    metricas = _read_csv(submission / "metricas.csv")
    val = pd.read_csv(dados / "validacao.csv", parse_dates=["date"])
    train = pd.read_csv(dados / "treino.csv", parse_dates=["date"])
    holdout = pd.read_csv(dados / "holdout_datas.csv", parse_dates=["date"])
    holdout_dates = set(holdout["date"])

    resultados = {
        "split": check_split(prev, holdout_dates),
        "metricas": check_metricas(metricas, prev, val, train),
        "sarima": check_sarima(prev, metricas),
        "baselines": check_baselines(prev),
        "repo": check_repo(submission),
        "diagnostico": check_diagnostico(submission),
        "ai_usage": check_ai_usage(submission),
    }
    nota = round(sum(v[0] for v in resultados.values()), 2)
    return {
        "nota": nota,
        "maximo": 10.0,
        "criterios": {
            k: {"pontos": round(v[0], 2), "max": PESOS[k], "detalhe": v[1]}
            for k, v in resultados.items()
        },
    }


def main() -> None:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Checks da Task 1 (crédito parcial).")
    parser.add_argument("--submission", type=Path, required=True, help="raiz do repo do grupo")
    parser.add_argument(
        "--dados",
        type=Path,
        default=here.parents[1] / "dados",
        help="pasta pública treino/validacao/holdout_datas",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.dados.is_dir():
        raise SystemExit(f"pasta de dados não encontrada: {args.dados}")
    rel = evaluate(args.submission.resolve(), args.dados.resolve())
    if args.json:
        json.dump(rel, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    print(f"Task 1  {rel['nota']:.1f} / {rel['maximo']:.0f}")
    for nome, c in rel["criterios"].items():
        print(f"  {nome:12} {c['pontos']:.1f}/{c['max']:.1f}  {c['detalhe']}")


if __name__ == "__main__":
    main()
