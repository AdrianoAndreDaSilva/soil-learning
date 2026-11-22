"""Preparação do conjunto curado (Mendeley Data) e funções comuns aos dois conjuntos.

Leitura dos arquivos horários, controle de qualidade, agregação diária, elegibilidade das estações,
heterogeneidade anterior à implantação, populações e agenda de rodadas.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

TARGET = "US20cm"
SM_RANGE = (0.0, 0.6)                 # m³/m³; fora da faixa vira ausente
DAILY_MIN_HOURS = 18                  # leituras horárias válidas para a média diária de umidade
DAILY_MIN_HOURS_PRECIP = 20           # leituras horárias válidas para o total diário de chuva
MIN_VALID_DAYS = 365
MIN_COVERAGE = 0.40
MAX_ZERO_FRACTION = 0.30
PRECIP_HOURLY_IMPLAUSIBLE_MM = 100.0
PRECIP_ANNUAL_RATIO_SUSPECT = 3.0

LOOKBACK_DAYS = 30
HORIZON_DAYS = 14
MIN_WINDOW_COVERAGE = 0.80
DEPLOYMENT_DAYS = 365                 # implantação: 365 dias após a primeira origem de previsão
ROUND_DAYS = 7
MIN_POST_EXAMPLES = 90
DEVELOPMENT_STATIONS = 16
MIN_DEVELOPMENT_EXAMPLES = 180
REFERENCE_POINTS = 512                # observações por estação na referência balanceada
HISTOGRAM_BINS = 30
SEED = 42

_STATION_ID = re.compile(r"_([0-9A-Z]+)\.xlsx$")
EARTH_RADIUS_KM = 6371.0


def read_hourly(path: Path) -> pd.DataFrame:
    """Série horária de uma estação em grade contínua, sem carimbos de tempo duplicados."""
    frame = pd.read_excel(path, sheet_name="Data")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
    measurements = [c for c in frame.columns if c != "Time"]
    for column in measurements:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["Time"].notna()].copy()
    frame["_filled"] = frame[measurements].notna().sum(axis=1)
    frame = (frame.sort_values(["Time", "_filled"], ascending=[True, False])
                  .drop_duplicates("Time", keep="first").drop(columns="_filled")
                  .set_index("Time").sort_index())
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h"))


def read_texture(path: Path) -> dict[str, float]:
    """Frações de areia, silte e argila (%) da aba ``Info``; o arquivo informa g/kg."""
    info = pd.read_excel(path, sheet_name="Info", header=None).iloc[:, :2].dropna()
    values = {str(k).strip(): v for k, v in zip(info.iloc[:, 0], info.iloc[:, 1])}
    return {f"{name.lower()}_pct": pd.to_numeric(values.get(name, np.nan), errors="coerce") / 10.0
            for name in ("Sand", "Silt", "Clay")}


def station_record(hourly: pd.DataFrame) -> dict[str, float]:
    """Diagnósticos da série bruta usados na elegibilidade e na confiabilidade do pluviômetro."""
    sm = hourly[TARGET] if TARGET in hourly else pd.Series(dtype=float)
    precip = hourly["Precip"] if "Precip" in hourly else pd.Series(dtype=float)
    years = (hourly.index.max() - hourly.index.min()).total_seconds() / (365.25 * 24 * 3600)
    valid = int(sm.notna().sum())
    return {
        "exact_zero_fraction": float((sm == 0).sum() / valid) if valid else np.nan,
        "precip_annual_mm": float(precip.sum(min_count=1)) / years if len(precip) and years > 0 else np.nan,
        "precip_max_hourly_mm": float(precip.max()) if precip.notna().any() else np.nan,
    }


def daily_series(hourly: pd.DataFrame) -> pd.DataFrame:
    """Controle de qualidade e agregação diária da umidade a 20 cm e da precipitação."""
    out = {}
    if TARGET in hourly:
        sm = hourly[TARGET].where(hourly[TARGET].between(*SM_RANGE))
        counts = sm.resample("D").count()
        out["US20cm_mean"] = sm.resample("D").mean().where(counts >= DAILY_MIN_HOURS)
    if "Precip" in hourly:
        precip = hourly["Precip"].where(~(hourly["Precip"] < 0))
        counts = precip.resample("D").count()
        out["precip_mm"] = precip.resample("D").sum(min_count=1).where(counts >= DAILY_MIN_HOURS_PRECIP)
    daily = pd.DataFrame(out).reindex(columns=["US20cm_mean", "precip_mm"])
    return daily.rename_axis("date").reset_index()


def build_panel(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Painel diário de todas as estações e tabela de estações com os critérios de elegibilidade."""
    catalog = pd.read_excel(raw_dir / "stations_catalog.xlsx")
    catalog["StationID"] = catalog["StationID"].astype(str)
    panels, records = [], []
    for path in sorted((raw_dir / "stations").glob("*.xlsx")):
        station_id = _STATION_ID.search(path.name).group(1)
        hourly = read_hourly(path)
        daily = daily_series(hourly)
        daily.insert(1, "StationID", station_id)
        panels.append(daily)
        records.append({
            "StationID": station_id, "station_type": "Agro" if "_Agro_" in path.name else "Acqua",
            **read_texture(path), **station_record(hourly),
            "valid_days": int(daily["US20cm_mean"].notna().sum()),
            "coverage": float(daily["US20cm_mean"].notna().mean()),
        })
    stations = catalog[["StationID", "City", "State", "Latitude", "Longitude"]].merge(
        pd.DataFrame(records), on="StationID", how="outer", validate="one_to_one").sort_values("StationID")

    ratio = stations["precip_annual_mm"] / stations["precip_annual_mm"].median()
    stations["precip_reliable"] = ~((stations["precip_max_hourly_mm"] > PRECIP_HOURLY_IMPLAUSIBLE_MM)
                                    | (ratio > PRECIP_ANNUAL_RATIO_SUSPECT))
    stations["eligible"] = (
        (stations["valid_days"] >= MIN_VALID_DAYS)
        & (stations["coverage"] >= MIN_COVERAGE)
        & (stations["exact_zero_fraction"].fillna(1) <= MAX_ZERO_FRACTION)
        & stations[["sand_pct", "silt_pct", "clay_pct"]].notna().all(axis=1)
    )
    return pd.concat(panels, ignore_index=True), stations.reset_index(drop=True)


def example_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Origens ``t0`` e datas-alvo dos exemplos válidos de cada estação.

    Um exemplo exige umidade em ``t0`` e em ``t0 + H`` e ao menos 80% da janela de 30 dias observada.
    """
    frames = []
    for station_id, group in panel.groupby("StationID", sort=False):
        y = group.set_index("date")["US20cm_mean"].sort_index()
        y = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq="D"))
        observed = y.notna().rolling(LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).mean()
        keep = y.shift(-HORIZON_DAYS).notna() & y.notna() & observed.ge(MIN_WINDOW_COVERAGE)
        t0 = y.index[keep]
        frames.append(pd.DataFrame({"StationID": station_id, "t0": t0,
                                    "target_date": t0 + pd.Timedelta(days=HORIZON_DAYS)}))
    return pd.concat(frames, ignore_index=True)


def balanced_reference(samples: dict[str, np.ndarray], exclude: str, seed: int) -> np.ndarray:
    """Referência da rede com a mesma quantidade de observações de cada estação, exceto ``exclude``."""
    rng = np.random.default_rng(seed)
    chunks = []
    for station_id in sorted(samples):
        if station_id == exclude:
            continue
        values = samples[station_id]
        chunks.append(values[rng.choice(values.size, size=REFERENCE_POINTS, replace=values.size < REFERENCE_POINTS)])
    return np.concatenate(chunks)


def heterogeneity(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    """Distância de cada estação à referência balanceada: Wasserstein-1, cosseno e Jensen-Shannon."""
    edges = np.linspace(SM_RANGE[0], SM_RANGE[1], HISTOGRAM_BINS + 1)
    rows = []
    for station_id in sorted(samples):
        local = samples[station_id]
        reference = balanced_reference(samples, station_id, SEED + sum(ord(c) for c in station_id))
        p = np.histogram(local, bins=edges)[0].astype(float)
        q = np.histogram(reference, bins=edges)[0].astype(float)
        norm = float(np.linalg.norm(p / p.sum()) * np.linalg.norm(q / q.sum()))
        cosine = float(1.0 - np.clip(np.dot(p / p.sum(), q / q.sum()) / norm, -1.0, 1.0))
        pj, qj = p + 1e-12, q + 1e-12
        rows.append({
            "StationID": station_id,
            "H_w1_balanced": float(wasserstein_distance(local, reference)),
            "H_cosine_balanced": cosine,
            "H_js_balanced": float(jensenshannon(pj / pj.sum(), qj / qj.sum(), base=2)),
        })
    return pd.DataFrame(rows)


def moisture_profiles(panel: pd.DataFrame) -> pd.DataFrame:
    """Média, desvio padrão, amplitude sazonal e autocorrelação de 7 dias da umidade por estação."""
    rows = []
    for station_id, group in panel.groupby("StationID", sort=False):
        series = group.set_index("date")["US20cm_mean"].sort_index()
        valid = series.dropna()
        if valid.empty:
            continue
        seasonal = valid.groupby(valid.index.dayofyear).mean()
        rows.append({
            "StationID": station_id, "mean": float(valid.mean()), "std": float(valid.std()),
            "seasonal_amplitude": float(seasonal.max() - seasonal.min()) if len(seasonal) > 1 else np.nan,
            "acf_7d": float(series.autocorr(7)),
        })
    return pd.DataFrame(rows)


def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latitudes.astype(float)), np.radians(longitudes.astype(float))
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _standardize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values, dtype=float)


def select_development_group(pool: pd.DataFrame, size: int = DEVELOPMENT_STATIONS, seed: int = SEED) -> list[str]:
    """Seleciona o grupo de desenvolvimento sem usar desempenho de modelo.

    Cobre primeiro os estados e os quartis de ``H_w1_balanced``, priorizando estações com mais
    exemplos pré-implantação, e completa por *farthest-point sampling* sobre a distância
    geográfica e a diferença de heterogeneidade.
    """
    pool = pool.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    quality = _standardize(pool["pre_examples"].to_numpy(dtype=float)) + 0.05 * rng.random(len(pool))
    pool = pool.assign(_quality=quality)
    selected: list[str] = []

    def best(group: pd.DataFrame) -> str:
        remaining = group.loc[~group["StationID"].isin(selected)]
        return str(remaining.sort_values("_quality", ascending=False).iloc[0]["StationID"])

    for column in ("State", "H_quartile"):
        for value in sorted(pool[column].astype(str).unique()):
            group = pool.loc[pool[column].astype(str).eq(value)]
            if len(selected) < size and not group["StationID"].isin(selected).all():
                selected.append(best(group))

    ids = pool["StationID"].tolist()
    index = {station_id: i for i, station_id in enumerate(ids)}
    geo = haversine_matrix(pool["Latitude"].to_numpy(), pool["Longitude"].to_numpy())
    h = pool["H_w1_balanced"].to_numpy(dtype=float)
    hdist = np.abs(h[:, None] - h[None, :])
    joint = 0.5 * geo / max(geo.max(), 1.0) + 0.5 * hdist / max(hdist.max(), 1e-12)
    while len(selected) < size:
        scores = [(min(joint[index[c], index[s]] for s in selected) + 0.05 * quality[index[c]], c)
                  for c in ids if c not in selected]
        selected.append(max(scores)[1])
    return selected


def round_schedule(deployment: pd.Timestamp, final_target_date: pd.Timestamp) -> pd.DataFrame:
    """Rodada 0 na implantação e uma rodada a cada ``ROUND_DAYS``; cada rodada avalia a janela seguinte."""
    last_origin = final_target_date - pd.Timedelta(days=HORIZON_DAYS)
    rows, r = [], 0
    while True:
        cutoff = deployment + pd.Timedelta(days=r * ROUND_DAYS)
        eval_start = cutoff + pd.Timedelta(days=1)
        eval_end = min(cutoff + pd.Timedelta(days=ROUND_DAYS), last_origin)
        if eval_end < eval_start:
            return pd.DataFrame(rows)
        rows.append({"round": r, "cutoff": cutoff, "eval_start": eval_start, "eval_end": eval_end})
        r += 1


def prepare(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa toda a preparação e retorna painel diário, tabela de estações e agenda de rodadas."""
    panel, stations = build_panel(raw_dir)
    eligible = set(stations.loc[stations["eligible"], "StationID"])
    eligible_panel = panel.loc[panel["StationID"].isin(eligible)]

    examples = example_index(eligible_panel)
    deployment = examples["t0"].min().normalize() + pd.Timedelta(days=DEPLOYMENT_DAYS)
    counts = pd.DataFrame({
        "pre_examples": examples.loc[examples["target_date"] <= deployment].groupby("StationID").size(),
        "post_examples": examples.loc[examples["t0"] > deployment].groupby("StationID").size(),
    }).reset_index(names="StationID")

    pre = eligible_panel.loc[eligible_panel["date"] <= deployment]
    samples = {sid: g["US20cm_mean"].dropna().to_numpy(dtype=float) for sid, g in pre.groupby("StationID")}
    samples = {sid: values for sid, values in samples.items() if values.size}
    stations = (stations.merge(heterogeneity(samples), on="StationID", how="left")
                        .merge(moisture_profiles(pre), on="StationID", how="left")
                        .merge(counts, on="StationID", how="left"))
    stations[["pre_examples", "post_examples"]] = stations[["pre_examples", "post_examples"]].fillna(0).astype(int)

    with_h = stations.loc[stations["H_w1_balanced"].notna()].copy()
    with_h["H_quartile"] = pd.qcut(with_h["H_w1_balanced"], 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    candidates = with_h.loc[with_h["pre_examples"] >= MIN_DEVELOPMENT_EXAMPLES]
    development = select_development_group(candidates)

    stations["development"] = stations["StationID"].isin(development)
    stations["full_network"] = stations["eligible"] & (stations["post_examples"] >= MIN_POST_EXAMPLES)
    stations["principal"] = stations["full_network"] & ~stations["development"]

    network = set(stations.loc[stations["full_network"], "StationID"])
    final_target = examples.loc[examples["StationID"].isin(network), "target_date"].max()
    return panel, stations, round_schedule(deployment, final_target)
