"""Verificações sobre o conjunto curado processado (data/processed/curated)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from soil_learning.analysis import holm
from soil_learning.config import ExperimentConfig
from soil_learning.data import SM_MAX, SoilData
from soil_learning.ped import hourly_qc
from soil_learning.preparation import heterogeneity, round_schedule

CONFIG = ExperimentConfig(data_dir="data/processed/curated", max_rounds=12)


@pytest.fixture(scope="module")
def store():
    return SoilData(CONFIG)


def test_populations(store):
    assert len(store.eligible_ids) == 174
    assert len(store.development_ids) == 16
    assert len(store.client_ids) == 156
    assert not set(store.development_ids) & set(store.client_ids)


def test_relative_input_ends_at_zero_and_target_is_14_days_ahead(store):
    ex = store.all_examples(store.client_ids[0])
    assert np.all(ex.target_date - ex.t0 == np.timedelta64(14, "D"))
    assert np.all(ex.dynamic[:, -1, 0] == 0)            # (θ(t0) − θ0) / s
    assert np.all(ex.static == 0)


def test_absolute_input_reproduces_persistence():
    store = SoilData(replace(CONFIG, input_variant="absolute"))
    ex = store.all_examples(store.client_ids[0])
    assert np.allclose(ex.dynamic[:, -1, 0] * SM_MAX, ex.persistence, atol=1e-6)


def test_training_uses_only_observed_targets(store):
    for _, row in store.schedule.iterrows():
        cutoff = np.datetime64(row["cutoff"])
        for sid in store.client_ids[:20]:
            train = store.train_examples(sid, row["cutoff"], postdeployment_only=True)
            evaluation = store.eval_examples(sid, row["eval_start"], row["eval_end"])
            if len(train):
                assert train.target_date.max() <= cutoff
                assert train.t0.min() > np.datetime64(store.deployment)
            if len(evaluation):
                assert evaluation.t0.min() > cutoff
                assert not set(evaluation.t0.tolist()) & set(train.t0.tolist())


def test_unreliable_rain_gauge_is_masked(store):
    unreliable = store.stations.loc[store.stations["eligible"] & ~store.stations["precip_reliable"], "StationID"]
    for sid in unreliable:
        ex = store.all_examples(sid)
        assert np.all(ex.dynamic[:, :, 1] == 0) and np.all(ex.dynamic[:, :, 3] == 0)


def test_heterogeneity_identifies_shifted_station():
    rng = np.random.default_rng(0)
    samples = {"A": rng.normal(0.20, 0.02, 800), "B": rng.normal(0.22, 0.02, 600),
               "C": rng.normal(0.21, 0.03, 700), "D": rng.normal(0.45, 0.03, 900)}
    scores = heterogeneity(samples).set_index("StationID")
    for metric in ("H_w1_balanced", "H_cosine_balanced", "H_js_balanced"):
        assert scores[metric].idxmax() == "D"


def test_round_schedule_is_weekly_and_evaluates_after_cutoff():
    schedule = round_schedule(pd.Timestamp("2016-07-29"), pd.Timestamp("2016-09-30"))
    assert (schedule["cutoff"].diff().dropna() == pd.Timedelta(days=7)).all()
    assert (schedule["eval_start"] == schedule["cutoff"] + pd.Timedelta(days=1)).all()
    assert schedule["eval_end"].max() == pd.Timestamp("2016-09-16")


def test_hourly_qc_rules():
    index = pd.date_range("2020-01-01", periods=12, freq="h")
    raw = pd.Series([0.20, 0.20, 0.40, 0.20, 0.20, 0.0, 0.0, 0.10, 0.04, 0.03, 0.0, 0.70], index=index)
    qc = hourly_qc(raw)
    valid = qc["valid"]
    assert np.isnan(valid.iloc[2]) and qc["spike"].iloc[2]          # pico que volta na hora seguinte
    assert qc["dropout"].iloc[5] and qc["dropout"].iloc[6]           # zero logo depois de valor alto
    assert valid.iloc[10] == 0.0 and not qc["dropout"].iloc[10]      # zero por secagem gradual fica
    assert np.isnan(valid.iloc[11]) and qc["above06"].iloc[11]       # fora da faixa
    assert valid.iloc[0] == 0.20 and valid.iloc[9] == 0.03


def test_holm_adjustment():
    assert np.allclose(holm(pd.Series([0.01, 0.04])), [0.02, 0.04])
