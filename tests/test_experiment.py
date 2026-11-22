from dataclasses import replace

import torch

from soil_learning import experiment as E
from soil_learning.config import ExperimentConfig


def test_initial_model_is_reused_only_with_the_same_configuration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(E, "train_initial_model",
                        lambda *a, **k: calls.append(1) or {"state": {}, "config": {}, "rounds": 1})
    config = ExperimentConfig(initial_rounds=1)
    path = tmp_path / "lstm_seed42.pt"
    torch.save({"state": {}, "config": config.initial_model_config(), "rounds": 1}, path)

    E.initial_model(None, "lstm", 42, config, path)
    assert calls == []                                              # mesma configuração: reaproveita
    E.initial_model(None, "lstm", 42, replace(config, learning_rate=1e-3), path)
    assert calls == []                                              # a fase longitudinal não muda o modelo inicial
    E.initial_model(None, "lstm", 42, replace(config, data_dir="data/processed/curated"), path)
    assert calls == [1]                                             # outros dados: treina de novo
