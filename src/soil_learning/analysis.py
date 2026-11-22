"""Análise das previsões gravadas em results/<configuração>/predictions/.

Convenções
- Unidade de análise: estação da população principal, rodadas a partir da 1. O MAE da estação é a média das cinco
  sementes.
- Razão sobre a persistência: média do MAE das estações ÷ média do MAE da persistência (estações com o mesmo peso).
- IC de 95%: percentis de 2000 reamostragens de estações. Os índices sorteados dependem só do número de estações e
  são os mesmos para todos os métodos de um conjunto, de modo que as diferenças entre métodos são pareadas.
- Testes de Wilcoxon bilaterais por estação; correção de Holm dentro de cada família, indicada nas tabelas.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .baselines import linear_reference
from .config import PROJECT_ROOT, ExperimentConfig
from .preparation import MIN_POST_EXAMPLES

RESULTS = PROJECT_ROOT / "results"
SEEDS = (42, 123, 456, 789, 2026)
METHODS = ("fedavg", "fedper", "finetune", "local", "central")
ARCHS = ("lstm", "tcn")
B = 2000
# Conjunto -> prefixo das configurações (<prefixo>_r01 e <prefixo>_r20) e pasta dos dados processados.
SETS = {"PED": ("ped", PROJECT_ROOT / "data" / "processed" / "ped"),
        "Curado": ("curated", PROJECT_ROOT / "data" / "processed" / "curated")}

_stations: dict = {}
_pred: dict = {}
_boot: dict = {}


def run_dir(name: str, rounds: int) -> Path:
    return RESULTS / f"{SETS[name][0]}_r{rounds:02d}"


def stations(name: str) -> pd.DataFrame:
    if name not in _stations:
        s = pd.read_parquet(SETS[name][1] / "stations.parquet")
        s["StationID"] = s["StationID"].astype(str)
        _stations[name] = s.set_index("StationID")
    return _stations[name]


def principal_ids(name: str) -> list[str]:
    return sorted(stations(name).index[stations(name)["principal"].astype(bool)])


def missing_runs(name: str, rounds_levels=(1, 20), methods=METHODS) -> list[Path]:
    return [run_dir(name, r) / "predictions" / f"{a}_{m}_seed{s}.parquet"
            for r in rounds_levels for a in ARCHS for m in methods for s in SEEDS
            if not (run_dir(name, r) / "predictions" / f"{a}_{m}_seed{s}.parquet").exists()]


def boot_idx(n: int) -> np.ndarray:
    """Índices de reamostragem de estações, os mesmos para todas as chamadas com o mesmo n."""
    if n not in _boot:
        _boot[n] = np.random.default_rng(20261105 + n).integers(0, n, (B, n))
    return _boot[n]


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(dtype=float)
    if not len(values):
        return pvalues
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return pd.Series(adjusted, index=pvalues.index)


def station_errors(name: str, rounds: int, arch: str, method: str) -> dict:
    """MAE por estação e semente (``e_seed``) e MAE da persistência por estação (``ep``)."""
    key = (name, rounds, arch, method)
    if key not in _pred:
        ids = set(principal_ids(name))
        xs = []
        for s in SEEDS:
            x = pd.read_parquet(run_dir(name, rounds) / "predictions" / f"{arch}_{method}_seed{s}.parquet",
                                columns=["StationID", "round", "pred", "y", "persistence"])
            x["StationID"] = x["StationID"].astype(str)
            x = x.loc[(x["round"] >= 1) & x["StationID"].isin(ids)].copy()
            x["seed"] = s
            x["e"] = (x["pred"] - x["y"]).abs()
            x["ep"] = (x["persistence"] - x["y"]).abs()
            xs.append(x)
        x = pd.concat(xs, ignore_index=True)
        e_seed = x.groupby(["StationID", "seed"])["e"].mean().unstack("seed")
        ep = x.groupby(["StationID", "seed"])["ep"].mean().unstack("seed").mean(axis=1)
        _pred[key] = {"e_seed": e_seed, "ep": ep}
    return _pred[key]


def aligned(name: str, rounds: int, arch: str, methods) -> tuple[pd.Index, pd.Series, dict[str, pd.DataFrame]]:
    """Estações comuns aos métodos, MAE da persistência e MAE por estação e semente de cada método."""
    runs = {m: station_errors(name, rounds, arch, m) for m in methods}
    ids = sorted(set.intersection(*(set(r["e_seed"].index) for r in runs.values())))
    return pd.Index(ids), runs[methods[0]]["ep"].loc[ids], {m: runs[m]["e_seed"].loc[ids] for m in methods}


def ratio_ci(e: np.ndarray, ep: np.ndarray) -> tuple[float, float, float]:
    idx = boot_idx(len(e))
    lo, hi = ci(e[idx].mean(1) / ep[idx].mean(1))
    return float(e.mean() / ep.mean()), lo, hi


def diff_ci(a: np.ndarray, b: np.ndarray, ep: np.ndarray) -> tuple[float, float, float]:
    """Diferença das razões (a − b); negativa quando a erra menos."""
    idx = boot_idx(len(a))
    d = (a[idx].mean(1) - b[idx].mean(1)) / ep[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float((a.mean() - b.mean()) / ep.mean()), float(lo), float(hi)


def performance() -> dict[str, pd.DataFrame]:
    """Razões com IC e variação entre sementes, comparações entre métodos e efeito do modelo inicial."""
    rows, pairs, init = [], [], []
    for name in SETS:
        for rounds in (1, 20):
            for arch in ARCHS:
                ids, ep, mae = aligned(name, rounds, arch, METHODS)
                epv = ep.to_numpy()
                for m in METHODS:
                    e = mae[m].mean(axis=1).to_numpy()
                    r, lo, hi = ratio_ci(e, epv)
                    seed_r = mae[m].mean(axis=0) / ep.mean()
                    station_ratio = e / epv
                    d = epv - e
                    rows.append({"conjunto": name, "rede": arch.upper(), "rodadas": rounds, "metodo": m, "n": len(ids),
                                 "razao": r, "ic_lo": lo, "ic_hi": hi, "dp_entre_sementes": float(seed_r.std(ddof=1)),
                                 "semente_min": float(seed_r.min()), "semente_max": float(seed_r.max()),
                                 "media_das_razoes_por_estacao": float(station_ratio.mean()),
                                 "mediana_das_razoes": float(np.median(station_ratio)),
                                 "frac_melhor_que_persistencia": float((d > 0).mean()),
                                 "p_vs_persistencia": float(stats.wilcoxon(d).pvalue),
                                 "mae_medio_estacao": float(e.mean()), "mae_persistencia": float(epv.mean())})
                for a, b in (("fedper", "fedavg"), ("finetune", "fedavg"), ("finetune", "fedper"),
                             ("local", "fedavg"), ("central", "fedavg")):
                    ea, eb = mae[a].mean(axis=1).to_numpy(), mae[b].mean(axis=1).to_numpy()
                    dr, lo, hi = diff_ci(ea, eb, epv)
                    per_seed = (mae[a].mean(axis=0) - mae[b].mean(axis=0)) / ep.mean()
                    pairs.append({"conjunto": name, "rede": arch.upper(), "rodadas": rounds, "a": a, "b": b,
                                  "razao_a_menos_b": dr, "ic_lo": lo, "ic_hi": hi,
                                  "frac_a_melhor_que_b": float((ea < eb).mean()),
                                  "p_wilcoxon_bruto": float(stats.wilcoxon(eb - ea).pvalue),
                                  "sementes_com_a_pior": int((per_seed > 0).sum()),
                                  "dif_por_semente_min": float(per_seed.min()), "dif_por_semente_max": float(per_seed.max())})
    rows = pd.DataFrame(rows)
    rows["p_vs_persistencia_holm"] = np.nan
    for _, idx in rows.groupby(["conjunto", "metodo"]).groups.items():   # família: 4 cenários por método
        rows.loc[idx, "p_vs_persistencia_holm"] = holm(rows.loc[idx, "p_vs_persistencia"]).to_numpy()
    pairs = pd.DataFrame(pairs)
    # Famílias de Holm: 12 testes por conjunto (FedPer × FedAvg, ajuste fino × FedAvg e × FedPer, 2 redes, 2 modelos
    # iniciais) e 8 cenários por contraste com o só local e com o centralizado (4 da PED e 4 do conjunto curado).
    pairs["p_holm_12"] = np.nan
    for name in SETS:
        m = pairs["conjunto"].eq(name) & pairs["a"].isin(["fedper", "finetune"])
        pairs.loc[m, "p_holm_12"] = holm(pairs.loc[m, "p_wilcoxon_bruto"]).to_numpy()
    for a in ("local", "central"):
        m = pairs["a"].eq(a)
        pairs.loc[m, "p_holm_8"] = holm(pairs.loc[m, "p_wilcoxon_bruto"]).to_numpy()
    pairs["familia"] = np.where(pairs["a"].isin(["fedper", "finetune"]),
                                "12 testes por conjunto (coluna p_holm_12)", "8 cenários por contraste (p_holm_8)")
    for name in SETS:
        for arch in ARCHS:
            for m in METHODS:
                a1, a20 = aligned(name, 1, arch, (m,)), aligned(name, 20, arch, (m,))
                ids = a1[0].intersection(a20[0])
                ep = a1[1].loc[ids].to_numpy()
                e1, e20 = a1[2][m].loc[ids].mean(axis=1).to_numpy(), a20[2][m].loc[ids].mean(axis=1).to_numpy()
                dr, lo, hi = diff_ci(e1, e20, ep)
                init.append({"conjunto": name, "rede": arch.upper(), "metodo": m, "n": len(ids),
                             "razao_1_rodada": e1.mean() / ep.mean(), "razao_20_rodadas": e20.mean() / ep.mean(),
                             "reducao_1_para_20": dr, "ic_lo": lo, "ic_hi": hi,
                             "p_wilcoxon": float(stats.wilcoxon(e1 - e20).pvalue)})
    return {"performance": rows, "method_pairs": pairs, "initial_model": pd.DataFrame(init)}


def spearman_bca(x, y, seed: int = 42, n_resamples: int = B) -> tuple[float, float]:
    """IC BCa de 95% da correlação de Spearman, por reamostragem pareada de estações."""
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()

    def rho(a, b, axis=-1):
        ra, rb = stats.rankdata(a, axis=axis), stats.rankdata(b, axis=axis)
        ra = ra - ra.mean(axis=axis, keepdims=True)
        rb = rb - rb.mean(axis=axis, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (ra * rb).sum(axis=axis) / np.sqrt((ra ** 2).sum(axis=axis) * (rb ** 2).sum(axis=axis))

    interval = stats.bootstrap((frame["x"].to_numpy(), frame["y"].to_numpy()), rho, paired=True, vectorized=True,
                               confidence_level=0.95, method="BCa", n_resamples=n_resamples,
                               random_state=np.random.default_rng(seed)).confidence_interval
    return float(interval.low), float(interval.high)


def heterogeneity() -> dict[str, pd.DataFrame]:
    """Questão 2: Spearman entre H e ΔMAE = MAE FedAvg − MAE FedPer, com controles e quartis de H.

    - IC BCa, p unilateral (ρ > 0) e bilateral, ρ por semente;
    - permutação de H dentro de cada estado (2000 permutações);
    - controle negativo: a mesma correlação com a diferença entre duas sementes do mesmo método (10 pares);
    - quartis de H: fração de estações com FedPer melhor e teste exato de Fisher do quartil mais alto contra os demais.
    """
    assoc, ctrl, quart = [], [], []
    for name in SETS:
        h_all = stations(name)["H_w1_balanced"]
        for rounds in (20, 1):
            for arch in ARCHS:
                ra, rp = station_errors(name, rounds, arch, "fedavg"), station_errors(name, rounds, arch, "fedper")
                ids = ra["e_seed"].index.intersection(rp["e_seed"].index).intersection(h_all.dropna().index)
                h = h_all.loc[ids]
                ea, ep_ = ra["e_seed"].loc[ids], rp["e_seed"].loc[ids]
                delta = ea.mean(axis=1) - ep_.mean(axis=1)
                rho, p2 = stats.spearmanr(h, delta)
                p1 = float(stats.spearmanr(h, delta, alternative="greater").pvalue)
                lo, hi = spearman_bca(h, delta)
                per_seed = [float(stats.spearmanr(h, ea[s] - ep_[s]).statistic) for s in SEEDS]
                ctl = [{"conjunto": name, "rodadas": rounds, "rede": arch.upper(), "metodo": method,
                        "semente_a": s1, "semente_b": s2, "rho": float(stats.spearmanr(h, e[s1] - e[s2]).statistic)}
                       for method, e in (("fedavg", ea), ("fedper", ep_)) for s1, s2 in combinations(SEEDS, 2)]
                ctrl.extend(ctl)
                c = pd.DataFrame(ctl)
                cf = c.loc[c["metodo"].eq("fedavg"), "rho"].to_numpy()
                both = np.concatenate([cf, -cf])   # a ordem das sementes na diferença é arbitrária
                state = stations(name).loc[ids, "State"]
                rng = np.random.default_rng(7)
                groups = [np.flatnonzero(state.to_numpy() == s) for s in state.unique()]
                hv, dv = h.to_numpy(), delta.to_numpy()
                null = []
                for _ in range(2000):
                    hp = hv.copy()
                    for g in groups:
                        hp[g] = rng.permutation(hv[g])
                    null.append(stats.spearmanr(hp, dv).statistic)
                perm_p = (np.sum(np.array(null) >= rho) + 1) / (len(null) + 1)
                pos = 100 * h.rank(pct=True)
                q4, better = pos.gt(75), delta.gt(0)
                tab = [[int((better & q4).sum()), int((~better & q4).sum())],
                       [int((better & ~q4).sum()), int((~better & ~q4).sum())]]
                fo, fg = stats.fisher_exact(tab), stats.fisher_exact(tab, alternative="greater")
                for k in range(4):
                    g = pos.gt(25 * k) & pos.le(25 * (k + 1))
                    quart.append({"conjunto": name, "rodadas": rounds, "rede": arch.upper(), "quartil_H": k + 1,
                                  "n": int(g.sum()), "frac_fedper_melhor": float(better[g].mean()),
                                  "mediana_delta_1e4": float(1e4 * delta[g].median()),
                                  "p_wilcoxon_no_quartil": float(stats.wilcoxon(delta[g]).pvalue)})
                assoc.append({
                    "conjunto": name, "rodadas": rounds, "rede": arch.upper(), "n": len(ids),
                    "rho_spearman": float(rho), "ic95_bca_lo": lo, "ic95_bca_hi": hi, "p_bilateral": float(p2),
                    "p_unilateral": p1, "rho_por_semente_min": min(per_seed), "rho_por_semente_max": max(per_seed),
                    "p_permutacao_dentro_do_estado_unilateral": float(perm_p),
                    "controle_fedavg_min": float(cf.min()), "controle_fedavg_max": float(cf.max()),
                    "controle_fedavg_media": float(cf.mean()), "controle_fedavg_dp": float(cf.std(ddof=1)),
                    "n_pares_controle_com_rho_ge_observado": int((both >= rho).sum()), "n_pares_controle": len(both),
                    "p_empirico_unilateral_controle": float(((both >= rho).sum() + 1) / (len(both) + 1)),
                    "controle_fedper_min": float(c.loc[c["metodo"].eq("fedper"), "rho"].min()),
                    "controle_fedper_max": float(c.loc[c["metodo"].eq("fedper"), "rho"].max()),
                    "frac_fedper_melhor_Q4": float(better[q4].mean()), "frac_fedper_melhor_Q1a3": float(better[~q4].mean()),
                    "fisher_Q4_vs_resto_p_bilateral": float(fo.pvalue), "fisher_Q4_vs_resto_p_unilateral": float(fg.pvalue),
                    "wilcoxon_p_dentro_Q4": float(stats.wilcoxon(delta[q4]).pvalue),
                    "wilcoxon_p_dentro_Q1a3": float(stats.wilcoxon(delta[~q4]).pvalue)})
    return {"heterogeneity": pd.DataFrame(assoc), "heterogeneity_seed_control": pd.DataFrame(ctrl),
            "heterogeneity_quartiles": pd.DataFrame(quart)}


def linear_comparison() -> dict[str, pd.DataFrame]:
    """Regressão linear de referência contra a persistência e contra FedAvg, FedPer e ajuste fino."""
    rows, out = [], {}
    for name in SETS:
        config = ExperimentConfig.from_json(PROJECT_ROOT / "configs" / f"{SETS[name][0]}_r20.json")
        per_round = linear_reference(config)
        out[f"linear_reference_rounds_{SETS[name][0]}"] = per_round
        st = stations(name)
        evaluated = set(st.index[st["principal"].astype(bool) & (st["post_examples"] >= MIN_POST_EXAMPLES)])
        w = per_round.loc[per_round["round"] >= 1].assign(e=lambda d: d["mae"] * d["n"]).groupby(
            ["StationID", "method"])[["e", "n"]].sum()
        lin = (w["e"] / w["n"]).unstack("method")
        lin = lin.loc[lin.index.isin(evaluated)]
        ids0, _, _ = aligned(name, 20, "lstm", ("fedavg",))
        for var, label in (("persistence", "persistência"), ("linear", "regressão linear"),
                           ("linear_station_bias", "regressão linear + viés por estação")):
            ids = lin.index.intersection(ids0)
            e, ep = lin.loc[ids, var].to_numpy(), lin.loc[ids, "persistence"].to_numpy()
            r, lo, hi = ratio_ci(e, ep)
            d = ep - e
            rows.append({"conjunto": name, "tipo": "referência", "variante": label, "rodadas": np.nan, "rede": "",
                         "metodo": "", "n": len(ids), "razao": r, "ic_lo": lo, "ic_hi": hi,
                         "mediana_razoes": float(np.median(e / ep)), "frac_melhor_que_persistencia": float((d > 0).mean()),
                         "p_vs_persistencia": float(stats.wilcoxon(d).pvalue) if var != "persistence" else np.nan})
        for rounds in (1, 20):
            for arch in ARCHS:
                ids_, ep_, mae = aligned(name, rounds, arch, ("fedavg", "fedper", "finetune"))
                ids = ids_.intersection(lin.index)
                ep = ep_.loc[ids].to_numpy()
                for m in ("fedavg", "fedper", "finetune"):
                    net = mae[m].loc[ids].mean(axis=1).to_numpy()
                    r, lo, hi = ratio_ci(net, ep)
                    row = {"conjunto": name, "tipo": "rede", "variante": "", "rodadas": rounds, "rede": arch.upper(),
                           "metodo": m, "n": len(ids), "razao": r, "ic_lo": lo, "ic_hi": hi,
                           "mediana_razoes": float(np.median(net / ep)),
                           "frac_melhor_que_persistencia": float((ep - net > 0).mean()),
                           "p_vs_persistencia": float(stats.wilcoxon(ep - net).pvalue)}
                    for var, tag in (("linear", "lin"), ("linear_station_bias", "linb")):
                        l = lin.loc[ids, var].to_numpy()
                        dd, dlo, dhi = diff_ci(net, l, ep)
                        row.update({f"{tag}_menos_rede_razao": -dd, f"{tag}_menos_rede_lo": -dhi,
                                    f"{tag}_menos_rede_hi": -dlo, f"{tag}_melhor_que_rede": float((l < net).mean()),
                                    f"p_{tag}_vs_rede": float(stats.wilcoxon(net - l).pvalue)})
                    rows.append(row)
    df = pd.DataFrame(rows)
    for name in SETS:   # família de Holm: 3 métodos × 2 redes × 2 modelos iniciais = 12 por conjunto
        m = df["conjunto"].eq(name) & df["tipo"].eq("rede")
        df.loc[m, "p_lin_vs_rede_holm"] = holm(df.loc[m, "p_lin_vs_rede"]).to_numpy()
    df["familia_holm"] = "12 comparações rede × regressão linear por conjunto"
    out["linear_reference"] = df
    return out


def _fedavg_runs(run: str, arch: str, seeds) -> list[Path]:
    return [RESULTS / run / "predictions" / f"{arch}_fedavg_seed{s}.parquet" for s in seeds]


def future_rain_and_budget() -> dict[str, pd.DataFrame]:
    """FedAvg com a chuva futura conhecida (PED, 20 rodadas, 5 sementes) e teste de orçamento de treino (LSTM, 2 sementes).

    Orçamento: as quatro execuções (adotado e maior, sem e com chuva futura) são comparadas nos mesmos pares
    (estação, origem), com a razão por estação, as razões das somas por faixa da chuva dos 14 dias seguintes e a
    inclinação da variação prevista sobre a observada nas origens com pelo menos 30 mm. A chuva futura de cada origem é
    a soma das 14 linhas seguintes do painel diário da estação (com pelo menos 10 dias medidos).

    Os ICs das duas tabelas usam a mesma sequência de sorteios (gerador com semente 0, primeiro o orçamento), por
    isso são calculados juntos.
    """
    data = SETS["PED"][1]
    ids = set(principal_ids("PED"))
    pan = pd.read_parquet(data / "daily_panel.parquet", columns=["StationID", "date", "precip_mm"])
    pan["StationID"] = pan["StationID"].astype(str)
    pan["date"] = pd.to_datetime(pan["date"])
    pan = pan.sort_values(["StationID", "date"])
    pan["fut"] = pan.groupby("StationID")["precip_mm"].transform(
        lambda s: s[::-1].rolling(14, min_periods=10).sum()[::-1].shift(-1))
    fut = pan[["StationID", "date", "fut"]].rename(columns={"date": "t0"})

    def load(run: str, arch: str = "lstm", seeds=(42, 123)) -> pd.DataFrame:
        x = pd.concat([pd.read_parquet(p) for p in _fedavg_runs(run, arch, seeds)])
        x["StationID"] = x["StationID"].astype(str)
        x = x[(x["round"] >= 1) & x["StationID"].isin(ids)]
        x["t0"] = pd.to_datetime(x["t0"])
        return x.groupby(["StationID", "t0"]).agg(pred=("pred", "mean"), y=("y", "first"),
                                                  p=("persistence", "first")).reset_index()

    runs = {"final_sem": "ped_r20", "final_com": "ped_r20_future_rain",
            "orc_sem": "budget_no_rain", "orc_com": "budget_future_rain"}
    G = {k: load(v) for k, v in runs.items()}
    keys = G["orc_sem"][["StationID", "t0"]]
    for k in ("orc_com", "final_sem", "final_com"):
        keys = keys.merge(G[k][["StationID", "t0"]])
    rows, M = [], {}
    for k, g in G.items():
        gg = g.merge(keys, on=["StationID", "t0"]).assign(e=lambda d: (d.pred - d.y).abs(), ep=lambda d: (d.p - d.y).abs())
        M[k] = gg.groupby("StationID")[["e", "ep"]].mean()
        gg = gg.merge(fut, on=["StationID", "t0"], how="left")
        big = gg[gg.fut >= 30]
        slope = np.polyfit(big.y - big.p, big.pred - big.p, 1)[0] if len(big) > 10 else np.nan
        bands = {"chuva>=30": gg.fut >= 30, "10-30": (gg.fut >= 10) & (gg.fut < 30), "<10": gg.fut < 10}
        rows.append({"execucao": k, "razao": M[k].e.mean() / M[k].ep.mean(), "inclinacao_30mm": slope,
                     "n_30mm": len(big), "n_pares": len(keys),
                     **{f"razao_soma_{b}": float(gg.e[m].sum() / gg.ep[m].sum()) for b, m in bands.items()}})

    rng = np.random.default_rng(0)
    sid = M["orc_sem"].index.values
    R = lambda k, ix: M[k].loc[ix].e.mean() / M[k].loc[ix].ep.mean()
    ic = []
    for a, b in (("orc_sem", "orc_com"), ("final_com", "orc_com"), ("final_sem", "orc_sem"), ("final_sem", "final_com")):
        d = [R(a, ix) - R(b, ix) for ix in (rng.choice(sid, len(sid)) for _ in range(2000))]
        ic.append({"a": a, "b": b, "dif": R(a, sid) - R(b, sid), "ic_inf": np.percentile(d, 2.5),
                   "ic_sup": np.percentile(d, 97.5)})

    rain = []
    for arch in ARCHS:
        def load5(run: str) -> pd.DataFrame:
            x = pd.concat([pd.read_parquet(p) for p in _fedavg_runs(run, arch, SEEDS)])
            x["StationID"] = x["StationID"].astype(str)
            x = x[(x["round"] >= 1) & x["StationID"].isin(ids)]
            x = x.assign(e=(x.pred - x.y).abs(), ep=(x.persistence - x.y).abs())
            return x.groupby("StationID")[["e", "ep"]].mean()
        a, b = load5(runs["final_sem"]), load5(runs["final_com"])
        sid5 = a.index.intersection(b.index)
        Rr = lambda m, ix: m.loc[ix].e.mean() / m.loc[ix].ep.mean()
        d = [Rr(a, ix) - Rr(b, ix) for ix in (rng.choice(sid5, len(sid5)) for _ in range(2000))]
        rain.append({"arch": arch, "razao_sem": Rr(a, sid5), "razao_com": Rr(b, sid5), "dif": Rr(a, sid5) - Rr(b, sid5),
                     "ic_inf": np.percentile(d, 2.5), "ic_sup": np.percentile(d, 97.5), "estacoes": len(sid5)})
    return {"budget": pd.DataFrame(rows), "budget_ci": pd.DataFrame(ic), "future_rain": pd.DataFrame(rain)}


def future_rain_and_budget_missing() -> list[Path]:
    paths = [p for run in ("ped_r20", "ped_r20_future_rain") for arch in ARCHS for p in _fedavg_runs(run, arch, SEEDS)]
    paths += [p for run in ("budget_no_rain", "budget_future_rain") for p in _fedavg_runs(run, "lstm", (42, 123))]
    return [p for p in paths if not p.exists()]


def station_metrics() -> pd.DataFrame:
    """MAE por configuração, arquitetura, método, semente e estação (rodadas a partir da 1), de todas as previsões
    disponíveis; serve para conferir execuções isoladas contra a referência."""
    frames = []
    for path in sorted(RESULTS.glob("*/predictions/*.parquet")):
        arch, method, seed = path.stem.split("_")
        x = pd.read_parquet(path, columns=["StationID", "round", "pred", "y", "persistence"])
        x = x.loc[x["round"] >= 1]
        g = x.assign(e=(x["pred"] - x["y"]).abs(), ep=(x["persistence"] - x["y"]).abs()).groupby("StationID")
        frames.append(pd.DataFrame({"n": g.size(), "mae": g["e"].mean(), "mae_persistence": g["ep"].mean()})
                      .reset_index().assign(config=path.parents[1].name, architecture=arch, method=method,
                                            seed=int(seed.removeprefix("seed"))))
    if not frames:
        return pd.DataFrame()
    cols = ["config", "architecture", "method", "seed", "StationID", "n", "mae", "mae_persistence"]
    return pd.concat(frames)[cols].sort_values(cols[:5]).reset_index(drop=True)


def compare(new: Path, reference: Path, rtol: float = 1e-5) -> list[str]:
    """Compara cada CSV de ``reference`` com o de mesmo nome em ``new``; devolve as linhas do relatório."""
    report, missing = [], []
    for ref_path in sorted(Path(reference).glob("*.csv")):
        new_path = Path(new) / ref_path.name
        if not new_path.exists():
            missing.append(ref_path.name)
            continue
        ref, cur = pd.read_csv(ref_path, dtype={"StationID": str}), pd.read_csv(new_path, dtype={"StationID": str})
        if ref_path.name == "station_metrics.csv":
            keys = ["config", "architecture", "method", "seed", "StationID"]
            both = ref.merge(cur, on=keys, suffixes=("", "_new"))
            values = ["n", "mae", "mae_persistence"]
            a, b = both[values].to_numpy(float), both[[f"{v}_new" for v in values]].to_numpy(float)
            scope = f"{len(both)} de {len(ref)} linhas da referência"
        else:
            if ref.shape != cur.shape or list(ref.columns) != list(cur.columns):
                report.append(f"{ref_path.name}: formato diferente ({ref.shape} × {cur.shape})")
                continue
            numeric = ref.select_dtypes("number").columns
            text = [c for c in ref.columns if c not in numeric]
            if not ref[text].fillna("").astype(str).equals(cur[text].fillna("").astype(str)):
                report.append(f"{ref_path.name}: rótulos diferentes")
                continue
            a, b = ref[numeric].to_numpy(float), cur[numeric].to_numpy(float)
            scope = f"{len(ref)} linhas"
        bad = ~np.isclose(a, b, rtol=rtol, atol=1e-12, equal_nan=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12), initial=0.0)
        report.append(f"{ref_path.name}: {scope}, {int(bad.sum())} valores divergentes (dif. relativa máx. {rel:.1e})")
    if missing:
        report.append(f"não gerados (faltam execuções): {', '.join(missing)}")
    return report
