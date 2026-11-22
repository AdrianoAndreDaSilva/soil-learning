"""Regressão linear de referência: causal, sobre descritores da mesma janela de 30 dias das redes."""
from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import PRECIP_SCALE, SM_MAX, SoilData

LINEAR_FEATURES = ["y0", "m30_gap", "m7_gap", "trend7", "rain3", "rain7", "rain30", "rain_avail",
                   "sin", "cos", "sin_y0", "cos_y0"]
LINEAR_RIDGE = 1.0          # sobre descritores padronizados
STATION_BIAS_SHRINK = 30    # exemplos equivalentes no encolhimento do intercepto por estação


def window_features(store: SoilData, station_id: str) -> pd.DataFrame:
    """Descritores da janela de cada exemplo: nível, médias, tendência, chuva e sazonalidade.

    Exige ``store`` com ``input_variant="absolute"``, em que o primeiro canal é a umidade ÷ 0,6.
    """
    ex = store.all_examples(station_id)
    d = ex.dynamic
    sm = np.where(d[:, :, 2] > 0, d[:, :, 0] * SM_MAX, np.nan)
    rain = np.where(d[:, :, 3] > 0, np.expm1(d[:, :, 1] * PRECIP_SCALE), np.nan)
    y0 = ex.persistence.astype(float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m30, m7, previous = np.nanmean(sm, axis=1), np.nanmean(sm[:, -7:], axis=1), np.nanmean(sm[:, -14:-7], axis=1)
    sin, cos = d[:, -1, 4].astype(float), d[:, -1, 5].astype(float)
    return pd.DataFrame({
        "StationID": station_id, "t0": ex.t0, "target_date": ex.target_date, "y": ex.y.astype(float), "y0": y0,
        "m30_gap": m30 - y0, "m7_gap": np.where(np.isnan(m7), 0, m7 - y0),
        "trend7": np.where(np.isnan(previous), 0, y0 - previous),
        "rain3": np.log1p(np.nansum(rain[:, -3:], axis=1)), "rain7": np.log1p(np.nansum(rain[:, -7:], axis=1)),
        "rain30": np.log1p(np.nansum(rain, axis=1)), "rain_avail": d[:, :, 3].mean(axis=1).astype(float),
        "sin": sin, "cos": cos, "sin_y0": sin * y0, "cos_y0": cos * y0,
    })


def huber_ridge(X: np.ndarray, z: np.ndarray, delta: float, ridge: float, iterations: int = 6):
    """Regressão de Huber por mínimos quadrados reponderados, com ridge sobre descritores padronizados.

    Cada iteração só precisa de X'WX e X'Wz somados entre clientes, de modo que o ajuste tem versão federada exata.
    """
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    penalty = np.eye(Xs.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.ones(len(z))
    for _ in range(iterations):
        coef = np.linalg.solve(Xs.T @ (Xs * weights[:, None]) + penalty, Xs.T @ (weights * z))
        weights = np.minimum(1.0, delta / np.maximum(np.abs(z - Xs @ coef), 1e-12))
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ coef


def linear_reference(config: ExperimentConfig) -> pd.DataFrame:
    """MAE por estação e rodada da regressão reajustada a cada corte, com o mesmo calendário das redes.

    Treino: exemplos com alvo já observado, do período anterior à implantação (estações do modelo inicial) e do
    posterior (população principal). Alvo: variação em 14 dias, com a perda de Huber das redes. ``linear`` é comum à
    rede; ``linear_station_bias`` soma a mediana encolhida dos resíduos de cada estação.
    """
    store = SoilData(replace(config, input_variant="absolute", future_rain=False))
    clients = sorted(store.client_ids)
    initial = [sid for sid in store.eligible_ids if len(store.train_examples(
        sid, store.deployment, postdeployment_only=False)) >= config.min_train_examples]
    feats = pd.concat([window_features(store, sid) for sid in sorted(set(clients) | set(initial))], ignore_index=True)
    feats["z"] = feats["y"] - feats["y0"]
    deployment = np.datetime64(store.deployment)
    pre = feats["target_date"].le(deployment) & feats["StationID"].isin(initial)
    post = feats["t0"].gt(deployment) & feats["StationID"].isin(clients)
    rows = []
    for _, r in store.schedule.loc[store.schedule["round"] >= 1].iterrows():
        train = feats.loc[pre | (post & feats["target_date"].le(np.datetime64(pd.Timestamp(r["cutoff"]))))]
        X, z = train[LINEAR_FEATURES].to_numpy(), train["z"].to_numpy()
        model = huber_ridge(X, z, config.huber_delta, LINEAR_RIDGE)
        residuals = pd.Series(z - model(X)).groupby(train["StationID"].to_numpy())
        bias = (residuals.median() * residuals.count() / (residuals.count() + STATION_BIAS_SHRINK)).to_dict()
        ev = feats.loc[feats["StationID"].isin(clients)
                       & feats["t0"].ge(np.datetime64(pd.Timestamp(r["eval_start"])))
                       & feats["t0"].le(np.datetime64(pd.Timestamp(r["eval_end"])))]
        if ev.empty:
            continue
        linear = ev["y0"].to_numpy() + model(ev[LINEAR_FEATURES].to_numpy())
        predictions = {"linear": linear,
                       "linear_station_bias": linear + ev["StationID"].map(bias).fillna(0.0).to_numpy(),
                       "persistence": ev["y0"].to_numpy()}
        for name, pred in predictions.items():
            err = pd.Series(np.abs(pred - ev["y"].to_numpy())).groupby(ev["StationID"].to_numpy())
            for sid, e in err:
                rows.append({"StationID": sid, "round": int(r["round"]), "method": name, "n": len(e),
                             "mae": float(e.mean())})
    return pd.DataFrame(rows)
