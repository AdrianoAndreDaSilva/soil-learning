"""Exemplos supervisionados por estação, construídos a partir do painel diário processado."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, ExperimentConfig
from .models import DYNAMIC_DIM, STATIC_DIM

TARGET = "US20cm_mean"
SM_MAX = 0.6                            # escala do canal absoluto (m³/m³)
PRECIP_SCALE = math.log1p(100.0)        # log1p(chuva) / log1p(100): 100 mm vale 1
FUTURE_RAIN_SCALE = math.log1p(300.0)
MIN_WINDOW_COVERAGE = 0.80              # 24 dos 30 dias da janela com umidade


def future_rain_features(future_precip: np.ndarray) -> np.ndarray:
    """Chuva observada em (t0, t0 + H]: total, primeira metade, segunda metade e fração de dias medidos."""
    valid = ~np.isnan(future_precip)
    f = np.where(valid, np.maximum(future_precip, 0.0), 0.0)
    half = len(f) // 2
    return np.asarray([np.log1p(f.sum()) / FUTURE_RAIN_SCALE, np.log1p(f[:half].sum()) / FUTURE_RAIN_SCALE,
                       np.log1p(f[half:].sum()) / FUTURE_RAIN_SCALE, float(valid.mean()) if len(f) else 0.0],
                      dtype=np.float32)


@dataclass
class Examples:
    dynamic: np.ndarray        # (n, lookback, 6)
    static: np.ndarray         # (n, 4)
    y: np.ndarray              # umidade em t0 + H
    persistence: np.ndarray    # umidade em t0
    t0: np.ndarray
    target_date: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def mask(self, mask: np.ndarray) -> "Examples":
        return Examples(self.dynamic[mask], self.static[mask], self.y[mask], self.persistence[mask],
                        self.t0[mask], self.target_date[mask])


class SoilData:
    """Painel, estações e agenda de rodadas de uma pasta processada, com os exemplos de cada estação.

    Canais por dia: umidade (relativa à origem ou absoluta), chuva em escala log, máscaras de umidade e de
    chuva e seno e cosseno do dia do ano. Ausências valem zero e são indicadas pelas máscaras.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        data_dir = PROJECT_ROOT / config.data_dir
        self.panel = pd.read_parquet(data_dir / "daily_panel.parquet")
        self.panel["date"] = pd.to_datetime(self.panel["date"])
        self.stations = pd.read_parquet(data_dir / "stations.parquet")
        self.stations["StationID"] = self.stations["StationID"].astype(str)
        self.schedule = pd.read_csv(data_dir / "round_schedule.csv", parse_dates=["cutoff", "eval_start", "eval_end"])
        if config.max_rounds is not None:
            self.schedule = self.schedule.loc[self.schedule["round"] <= config.max_rounds].reset_index(drop=True)
        self.deployment = pd.Timestamp(self.schedule.iloc[0]["cutoff"])
        self._cache: dict[str, Examples] = {}

    def _ids(self, column: str) -> list[str]:
        return self.stations.loc[self.stations[column].astype(bool), "StationID"].tolist()

    @property
    def client_ids(self) -> list[str]:
        """População principal: as estações avaliadas."""
        return self._ids("principal")

    @property
    def eligible_ids(self) -> list[str]:
        return self._ids("eligible")

    @property
    def development_ids(self) -> list[str]:
        return self._ids("development")

    def all_examples(self, station_id: str) -> Examples:
        station_id = str(station_id)
        if station_id in self._cache:
            return self._cache[station_id]

        g = self.panel.loc[self.panel["StationID"].astype(str).eq(station_id)]
        if g.empty:
            raise KeyError(f"estação sem painel diário: {station_id}")
        g = g.sort_values("date").set_index("date")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        sm = g[TARGET].astype(float)
        precip = g["precip_mm"].astype(float)
        reliable = self.stations.loc[self.stations["StationID"].eq(station_id), "precip_reliable"].iloc[0]
        if not bool(reliable):
            precip = pd.Series(np.nan, index=g.index, dtype=float)
        precip_values = precip.to_numpy(dtype=float)
        zeros = np.zeros(STATIC_DIM, dtype=np.float32)
        relative = self.config.input_variant == "relative"

        dynamic_rows, static_rows, ys, persist, t0s, targets = [], [], [], [], [], []
        L, H = self.config.lookback_days, self.config.horizon_days
        for pos in range(L - 1, len(g) - H):
            y, p0 = sm.iloc[pos + H], sm.iloc[pos]
            window_sm = sm.iloc[pos - L + 1:pos + 1]
            if np.isnan(y) or np.isnan(p0) or window_sm.notna().mean() < MIN_WINDOW_COVERAGE:
                continue
            window_p = precip.iloc[pos - L + 1:pos + 1]
            if relative:
                moisture = (window_sm.to_numpy(dtype=np.float32) - np.float32(p0)) / self.config.residual_scale
            else:
                moisture = window_sm.to_numpy(dtype=np.float32) / SM_MAX
            p_values = np.maximum(window_p.to_numpy(dtype=np.float32), 0.0)
            angle = 2.0 * np.pi * window_sm.index.dayofyear.to_numpy(dtype=np.float32) / 365.25
            dynamic_rows.append(np.column_stack([
                np.nan_to_num(moisture, nan=0.0), np.nan_to_num(np.log1p(p_values) / PRECIP_SCALE, nan=0.0),
                window_sm.notna().to_numpy(dtype=np.float32), window_p.notna().to_numpy(dtype=np.float32),
                np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32),
            ]).astype(np.float32))
            static_rows.append(future_rain_features(precip_values[pos + 1:pos + H + 1])
                               if self.config.future_rain else zeros)
            ys.append(float(y)); persist.append(float(p0))
            t0s.append(g.index[pos]); targets.append(g.index[pos + H])

        examples = Examples(
            dynamic=np.asarray(dynamic_rows, dtype=np.float32).reshape(-1, L, DYNAMIC_DIM),
            static=np.asarray(static_rows, dtype=np.float32).reshape(-1, STATIC_DIM),
            y=np.asarray(ys, dtype=np.float32),
            persistence=np.asarray(persist, dtype=np.float32),
            t0=np.asarray(t0s, dtype="datetime64[ns]"),
            target_date=np.asarray(targets, dtype="datetime64[ns]"),
        )
        self._cache[station_id] = examples
        return examples

    def train_examples(self, station_id: str, cutoff: pd.Timestamp, *, postdeployment_only: bool) -> Examples:
        """Exemplos cujo alvo já foi observado em ``cutoff``."""
        ex = self.all_examples(station_id)
        mask = ex.target_date <= np.datetime64(pd.Timestamp(cutoff))
        if postdeployment_only:
            mask &= ex.t0 > np.datetime64(self.deployment)
        return ex.mask(mask)

    def eval_examples(self, station_id: str, start: pd.Timestamp, end: pd.Timestamp) -> Examples:
        ex = self.all_examples(station_id)
        mask = (ex.t0 >= np.datetime64(pd.Timestamp(start))) & (ex.t0 <= np.datetime64(pd.Timestamp(end)))
        return ex.mask(mask)
