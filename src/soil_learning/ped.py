"""Preparação dos dados da Plataforma de Entrega de Dados (PED) do CEMADEN.

Converte os arquivos baixados (um zip por pedido, um JSON por estação) no painel diário do estudo:

1. controle horário da umidade a 20 cm (``hourly_qc``): faixa de 0 a 0,6 m³/m³, picos que voltam na hora
   seguinte e zeros logo depois de valor alto;
2. regras diárias (``daily_qc``): ano hidrológico com falha crônica, sensor travado apesar de chuva, deriva do
   sensor e ano com total de chuva implausível;
3. regra por trecho (``segment_qc``): zeros que não respondem à chuva e platôs altos sem chuva;
4. desenho do estudo (``build_design``): elegibilidade, populações, heterogeneidade e agenda de rodadas de 28 dias,
   com a mesma implantação do conjunto curado (29/07/2016).

As regras usam a série inteira de cada estação (controle retrospectivo, como a curadoria manual).
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from . import preparation as P

SENSORS = {340: "US20cm", 330: "US10cm", 10: "Precip"}
JUMP = 0.05                 # m³/m³ entre horas consecutivas: picos e quedas bruscas
DRIFT_SHIFT = 0.08          # m³/m³: afastamento do piso seco que caracteriza deriva
CADENCE_DAYS = 28


def hourly_qc(raw: pd.Series) -> dict[str, pd.Series]:
    """Controle da umidade horária: faixa, picos que voltam e zeros por queda brusca.

    Devolve a série válida e as marcas de cada regra no índice horário contínuo da série.
    """
    full = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))
    dx, nxt = full.diff(), full.shift(-1) - full
    spike = (dx.abs() > JUMP) & (nxt.abs() > JUMP) & (np.sign(dx) != np.sign(nxt))
    last_nonzero = full.where(full != 0).ffill()
    dropout = (full == 0) & (last_nonzero > JUMP)            # zero logo depois de um valor alto: falha de leitura
    out_of_range = full.notna() & ~full.between(*P.SM_RANGE)
    valid = full.where(~out_of_range & ~spike & ~dropout)
    return {"valid": valid, "above06": full > P.SM_RANGE[1], "spike": spike & full.notna(),
            "dropout": dropout, "out_of_range": out_of_range}


def station_daily(records: list[dict]) -> pd.DataFrame | None:
    """Série diária de uma estação a partir das leituras do JSON da PED."""
    d = pd.DataFrame([r for r in records if r["id_sensor"] in SENSORS], columns=["datahora", "id_sensor", "valor"])
    if d.empty:
        return None
    d["hour"] = pd.to_datetime(d["datahora"]).dt.floor("h")
    sm = d.loc[d["id_sensor"] != 10].groupby(["id_sensor", "hour"])["valor"].mean().unstack("id_sensor")
    rain = d.loc[d["id_sensor"] == 10].groupby("hour")["valor"].sum()
    hourly = pd.DataFrame(index=sm.index.union(rain.index))
    for sensor in (340, 330):
        hourly[SENSORS[sensor]] = sm[sensor] if sensor in sm else np.nan
    # A PED não registra a maior parte das horas sem chuva: a hora sem registro vale zero quando a estação
    # transmitiu umidade (a 10 ou a 20 cm) naquela hora.
    transmitted = hourly[["US20cm", "US10cm"]].notna().any(axis=1)
    hourly["Precip"] = rain.reindex(hourly.index).where(lambda v: v >= 0)
    hourly.loc[transmitted & hourly["Precip"].isna(), "Precip"] = 0.0
    day = hourly.index.normalize()
    days = day.unique()
    raw = hourly["US20cm"]
    in_range = raw.where(raw.between(*P.SM_RANGE))
    out = {
        "US20cm_mean": in_range.groupby(day).mean().where(in_range.groupby(day).count() >= P.DAILY_MIN_HOURS),
        "US20cm_hours": raw.groupby(day).count(),
        "US20cm_zero_hours": (raw == 0).groupby(day).sum(),
    }
    if raw.notna().any():
        qc = hourly_qc(raw.dropna())
        qday = qc["valid"].index.normalize()
        mean = qc["valid"].groupby(qday).mean().where(qc["valid"].groupby(qday).count() >= P.DAILY_MIN_HOURS)
        out["US20cm_mean_qc"] = mean.reindex(days)
        for flag in ("above06", "spike", "dropout"):
            out[f"US20cm_{flag}_hours"] = qc[flag].groupby(qday).sum().reindex(days)
    precip = hourly["Precip"]
    # Hora com mais de 100 mm é implausível: o dia fica sem chuva.
    implausible = (precip > P.PRECIP_HOURLY_IMPLAUSIBLE_MM).groupby(day).any()
    out["precip_mm"] = (precip.groupby(day).sum(min_count=1)
                        .where(precip.groupby(day).count() >= P.DAILY_MIN_HOURS_PRECIP).where(~implausible))
    out["precip_max_hourly"] = precip.groupby(day).max()
    return pd.DataFrame(out).rename_axis("date").reset_index()


def read_raw(raw_dir: Path) -> pd.DataFrame:
    """Painel diário de todas as estações dos arquivos ``*.zip`` de ``raw_dir``."""
    frames = []
    for path in sorted(Path(raw_dir).glob("*.zip")):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                records = json.loads(z.read(name).decode("utf-8", errors="replace"))
                if not records:
                    continue
                daily = station_daily(records)
                if daily is not None:
                    frames.append(daily.assign(StationID=str(records[0]["codestacao"])))
        print(path.name, flush=True)
    panel = pd.concat(frames, ignore_index=True)
    # Pedidos de períodos vizinhos não se sobrepõem; se houver repetição, fica a linha com mais horas.
    panel = panel.sort_values("US20cm_hours", na_position="first").drop_duplicates(["StationID", "date"], keep="last")
    return panel.sort_values(["StationID", "date"]).reset_index(drop=True)


def drift_mask(panel: pd.DataFrame, col: str, hyear: pd.Series) -> pd.Series:
    """Dias com deriva do sensor.

    O piso seco de referência é o percentil 5 das médias diárias do primeiro ano hidrológico (maio a abril) com
    pelo menos 180 dias válidos. Um dia sai quando o percentil 5 dos 365 dias seguintes (com pelo menos 180
    válidos; no fim da série, dos 365 anteriores) se afasta do de referência em mais de ``DRIFT_SHIFT``, para cima
    ou para baixo. O limiar foi calibrado contra os cortes feitos pelos curadores do conjunto do Mendeley.
    """
    flag = pd.Series(False, index=panel.index)
    for _, g in panel.groupby("StationID"):
        x = g.set_index("date")[col]
        hy = hyear.loc[g.index].to_numpy()
        days_per_year = x.groupby(hy).count()
        years = days_per_year.index[days_per_year >= 180]
        if not len(years):
            continue
        ref = x[hy == years.min()].quantile(0.05)
        s = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D"))
        forward = s[::-1].rolling(365, min_periods=180).quantile(0.05)[::-1]
        backward = s.rolling(365, min_periods=180).quantile(0.05)
        floor = forward.fillna(backward)
        bad = ((floor - ref).abs() > DRIFT_SHIFT).reindex(x.index).fillna(False).to_numpy()
        flag.loc[g.index[bad]] = True
    return flag


def daily_qc(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regras diárias sobre ``US20cm_mean_qc`` e a chuva, com o relatório do que cada regra remove.

    - ano hidrológico com mais de 20% das horas acima de 0,6 m³/m³ (falha crônica): o ano inteiro sai;
    - sensor travado: 30 dias ou mais com a mesma média diária (|Δ| < 1e-4) apesar de pelo menos 50 mm de chuva,
      só em estações com pluviômetro confiável;
    - deriva do sensor (``drift_mask``);
    - chuva do ano hidrológico com total fora de 1/3 a 3 vezes a mediana da rede (pluviômetro parado ou em excesso).
    """
    panel = panel.copy()
    col = "US20cm_mean_qc"
    hyear = (panel["date"] - pd.DateOffset(months=4)).dt.year
    g = panel.groupby("StationID")
    span_years = ((g["date"].max() - g["date"].min()).dt.days + 1) / 365.25
    annual = g["precip_mm"].sum() / span_years
    reliable_station = ~((g["precip_max_hourly"].max() > P.PRECIP_HOURLY_IMPLAUSIBLE_MM)
                         | (annual / annual.median() > P.PRECIP_ANNUAL_RATIO_SUSPECT))
    reliable = panel["StationID"].map(reliable_station).fillna(False).astype(bool)

    before = panel[col].notna()
    share = (panel.groupby(["StationID", hyear])["US20cm_above06_hours"].transform("sum")
             / panel.groupby(["StationID", hyear])["US20cm_hours"].transform("sum").replace(0, np.nan))
    chronic = share > 0.2
    panel.loc[chronic, col] = np.nan
    removed_chronic = before & chronic

    stuck = pd.Series(False, index=panel.index)
    for _, g in panel.loc[reliable].groupby("StationID"):
        x = g.set_index("date")[[col, "precip_mm"]]
        x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D"))
        same = x[col].diff().abs().lt(1e-4) & x[col].notna() & x[col].shift().notna()
        run = (~same).cumsum()
        for _, r in x.loc[same].groupby(run[same]):
            days = r.index.union([r.index[0] - pd.Timedelta(days=1)])
            if len(days) >= 30 and x.loc[days, "precip_mm"].sum() >= 50:
                stuck.loc[g.index[g["date"].isin(days)]] = True
    removed_stuck = panel[col].notna() & stuck
    panel.loc[stuck, col] = np.nan

    drift = drift_mask(panel, col, hyear)
    removed_drift = panel[col].notna() & drift
    panel.loc[drift, col] = np.nan

    report = []
    for rule, mask in (("ano com falha crônica (> 20% das horas acima de 0,6)", removed_chronic),
                       ("sensor travado com chuva (≥ 30 dias, ≥ 50 mm)", removed_stuck),
                       (f"deriva do sensor (piso seco a mais de {DRIFT_SHIFT} do primeiro ano)", removed_drift)):
        report.append({"variable": "US20cm", "rule": rule, "days_removed": int(mask.sum()),
                       "stations": int(panel.loc[mask, "StationID"].nunique())})
    for flag, rule in (("above06", "leitura horária acima de 0,6 (faixa)"), ("spike", "pico que volta na hora seguinte"),
                       ("dropout", "zero logo depois de valor alto (falha de leitura)")):
        report.append({"variable": "US20cm", "rule": rule, "hours_removed": int(panel[f"US20cm_{flag}_hours"].sum()),
                       "stations": int(panel.loc[panel[f"US20cm_{flag}_hours"] > 0, "StationID"].nunique())})
    report.append({"variable": "US20cm", "rule": "dias válidos: só faixa e horas × controle ampliado",
                   "days_study": int(panel["US20cm_mean"].notna().sum()), "days_qc": int(panel[col].notna().sum())})

    rain = panel.groupby(["StationID", hyear])["precip_mm"].agg(["sum", "count"])
    rain = rain.loc[rain["count"] >= 180]
    ratio = rain["sum"] / rain["sum"].groupby(level=1).transform("median")
    bad = set(ratio.index[~ratio.between(1 / 3, 3)])
    bad_rain = pd.Series([(s, y) in bad for s, y in zip(panel["StationID"], hyear)], index=panel.index)
    report.append({"variable": "precip", "rule": "ano com chuva fora de [1/3; 3] da mediana da rede",
                   "days_removed": int((bad_rain & panel["precip_mm"].notna()).sum()),
                   "stations": int(panel.loc[bad_rain, "StationID"].nunique())})
    panel.loc[bad_rain, "precip_mm"] = np.nan
    return panel, pd.DataFrame(report)


def segment_qc(panel: pd.DataFrame, col: str = "US20cm_mean_qc") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regra por trecho, baseada no critério de validação por chuva dos autores da rede (Zeri et al., 2018).

    - zero (θ ≤ 0,005) que não responde a pelo menos 20 mm de chuva em 3 dias dentro do trecho;
    - platô acima de 0,45 por 5 dias ou mais com menos de 5 mm de chuva no trecho.

    Trechos com chuva medida em menos da metade dos dias ficam como estão.
    """
    panel = panel.sort_values(["StationID", "date"]).copy()
    zero_rule, plateau_rule = "zero que não responde a ≥ 20 mm em 3 dias", "platô > 0,45 por ≥ 5 dias sem chuva (< 5 mm)"
    masks = {zero_rule: pd.Series(False, index=panel.index), plateau_rule: pd.Series(False, index=panel.index)}
    for _, g in panel.groupby("StationID"):
        x = g.set_index("date")[[col, "precip_mm"]]
        x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D"))
        r3 = x["precip_mm"].rolling(3, min_periods=1).sum()
        for name, cond in ((zero_rule, x[col].le(0.005)), (plateau_rule, x[col].gt(0.45))):
            run = (cond != cond.shift()).cumsum()
            for _, r in x.loc[cond].groupby(run[cond]):
                if r["precip_mm"].notna().mean() < 0.5:
                    continue
                if name == zero_rule:
                    bad = r3.loc[r.index].max() >= 20
                else:
                    bad = len(r) >= 5 and r["precip_mm"].sum() < 5
                if bad:
                    masks[name].loc[g.index[g["date"].isin(r.index)]] = True
    report = []
    for name, m in masks.items():
        report.append({"variable": "US20cm", "rule": f"trecho: {name}", "days_removed": int(m.sum()),
                       "stations": int(panel.loc[m, "StationID"].nunique())})
        panel.loc[m, col] = np.nan
    return panel, pd.DataFrame(report)


def station_metadata(ids, registry: pd.DataFrame, curated: pd.DataFrame) -> pd.DataFrame:
    """Metadados do cadastro da PED; estações que saíram do cadastro usam os do conjunto curado.

    A textura do solo e o grupo de desenvolvimento vêm do conjunto curado.
    """
    meta = registry.rename(columns={"codestacao": "StationID", "cidade": "City", "uf": "State",
                                    "latitude": "Latitude", "longitude": "Longitude"})
    meta["station_type"] = np.where(meta["id_tipoestacao"] == 4, "Agro", "Acqua")
    columns = ["StationID", "City", "State", "Latitude", "Longitude", "station_type"]
    fallback = curated.loc[curated["StationID"].isin(sorted(set(ids) - set(meta["StationID"]))), columns]
    meta = pd.concat([meta[columns], fallback])
    meta = meta.loc[meta["StationID"].isin(set(ids))]
    stations = meta.merge(curated[["StationID", "sand_pct", "silt_pct", "clay_pct", "development"]],
                          on="StationID", how="left")
    stations["development"] = stations["development"].fillna(False).astype(bool)
    stations["in_mendeley"] = stations["StationID"].isin(curated["StationID"])
    stations["principal_original"] = stations["StationID"].isin(curated.loc[curated["principal"], "StationID"])
    return stations


def build_design(panel: pd.DataFrame, registry: pd.DataFrame, curated: pd.DataFrame, deployment: pd.Timestamp
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Painel diário, estações (elegibilidade, populações, H) e agenda de rodadas de ``CADENCE_DAYS`` dias.

    Elegibilidade: pelo menos 365 dias válidos e até 30% de leituras horárias iguais a zero. Ao contrário do conjunto
    curado, não se exige cobertura mínima do período, que puniria os trechos retirados pelo controle em 12 anos.
    """
    window = panel.copy()
    window["US20cm_mean"] = window["US20cm_mean_qc"]
    end = window["date"].max()
    stations = station_metadata(window["StationID"].unique(), registry, curated)

    g = window.groupby("StationID")
    span = (g["date"].max() - g["date"].min()).dt.days + 1
    stats = pd.DataFrame({
        "valid_days": g["US20cm_mean"].count(), "coverage": g["US20cm_mean"].count() / span,
        "exact_zero_fraction": g["US20cm_zero_hours"].sum() / g["US20cm_hours"].sum().replace(0, np.nan),
        "precip_annual_mm": g["precip_mm"].sum() / (span / 365.25), "precip_max_hourly_mm": g["precip_max_hourly"].max(),
    }).reset_index()
    stations = stations.merge(stats, on="StationID", how="left")
    ratio = stations["precip_annual_mm"] / stations["precip_annual_mm"].median()
    stations["precip_reliable"] = ~((stations["precip_max_hourly_mm"] > P.PRECIP_HOURLY_IMPLAUSIBLE_MM)
                                    | (ratio > P.PRECIP_ANNUAL_RATIO_SUSPECT))
    stations["eligible"] = ((stations["valid_days"] >= P.MIN_VALID_DAYS)
                            & (stations["exact_zero_fraction"].fillna(1) <= P.MAX_ZERO_FRACTION))

    eligible = set(stations.loc[stations["eligible"], "StationID"])
    daily = window.loc[window["StationID"].isin(eligible), ["date", "StationID", "US20cm_mean", "precip_mm"]]
    examples = P.example_index(daily)
    counts = pd.DataFrame({
        "pre_examples": examples.loc[examples["target_date"] <= deployment].groupby("StationID").size(),
        "post_examples": examples.loc[examples["t0"] > deployment].groupby("StationID").size(),
    }).reset_index(names="StationID")
    pre = daily.loc[daily["date"] <= deployment]
    samples = {sid: v["US20cm_mean"].dropna().to_numpy(dtype=float) for sid, v in pre.groupby("StationID")}
    with np.errstate(invalid="ignore", divide="ignore"):
        heterogeneity = P.heterogeneity({sid: v for sid, v in samples.items() if v.size})
    stations = (stations.merge(heterogeneity, on="StationID", how="left")
                        .merge(P.moisture_profiles(pre), on="StationID", how="left")
                        .merge(counts, on="StationID", how="left"))
    stations[["pre_examples", "post_examples"]] = stations[["pre_examples", "post_examples"]].fillna(0).astype(int)
    stations["full_network"] = stations["eligible"] & (stations["post_examples"] >= P.MIN_POST_EXAMPLES)
    stations["principal"] = stations["full_network"] & ~stations["development"]

    rows, r = [], 0
    last_origin = end - pd.Timedelta(days=P.HORIZON_DAYS)
    while True:
        cutoff = deployment + pd.Timedelta(days=r * CADENCE_DAYS)
        eval_start = cutoff + pd.Timedelta(days=1)
        eval_end = min(cutoff + pd.Timedelta(days=CADENCE_DAYS), last_origin)
        if eval_end < eval_start:
            break
        rows.append({"round": r, "cutoff": cutoff, "eval_start": eval_start, "eval_end": eval_end})
        r += 1
    return daily, stations, pd.DataFrame(rows)
