"""Modelo inicial e simulação longitudinal: FedAvg, FedPer, ajuste fino local, só local e centralizado."""
from __future__ import annotations

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from .config import PROJECT_ROOT, ExperimentConfig
from .data import Examples, SoilData
from .federated import State, central_round, client_seed, cpu_state, fedavg_round, fedper_round, predict, train_local
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_name(architecture: str, method: str, seed: int) -> str:
    return f"{architecture}_{method}_seed{seed}"


def assert_causal(examples: Examples, cutoff: pd.Timestamp, station_id: str,
                  deployment: pd.Timestamp | None = None) -> None:
    """Falha se algum exemplo de treino tiver alvo posterior ao corte (ou origem anterior à implantação)."""
    if len(examples) == 0:
        return
    if examples.target_date.max() > np.datetime64(pd.Timestamp(cutoff)):
        raise AssertionError(f"alvo posterior ao corte {cutoff} em {station_id}")
    if deployment is not None and examples.t0.min() <= np.datetime64(pd.Timestamp(deployment)):
        raise AssertionError(f"exemplo pré-implantação no treino longitudinal de {station_id}")


def train_initial_model(store: SoilData, architecture: str, seed: int, config: ExperimentConfig) -> dict:
    """FedAvg por ``config.initial_rounds`` rodadas com os exemplos anteriores à implantação.

    Participam as estações elegíveis com pelo menos ``min_train_examples`` exemplos nesse período, inclusive as
    do grupo de desenvolvimento e as que ficaram fora da população principal.
    """
    factory = lambda: build_model(architecture)
    data = {sid: store.train_examples(sid, store.deployment, postdeployment_only=False) for sid in store.eligible_ids}
    data = {sid: ex for sid, ex in data.items() if len(ex) >= config.min_train_examples}
    for sid, ex in data.items():
        assert_causal(ex, store.deployment, sid)
    set_seed(seed)
    state = cpu_state(factory())
    for r in range(1, config.initial_rounds + 1):
        state, _ = fedavg_round(factory, state, data, config, seed=seed, round_idx=-r,
                                lr=config.initial_learning_rate, epochs=config.initial_local_epochs)
    return {"architecture": architecture, "seed": seed, "state": state, "rounds": config.initial_rounds,
            "stations": len(data), "examples": int(sum(len(ex) for ex in data.values())),
            "config": config.initial_model_config()}


def initial_model(store: SoilData, architecture: str, seed: int, config: ExperimentConfig, path: Path) -> dict:
    """Carrega o modelo inicial de ``path`` ou o treina (e grava) se a configuração salva for outra."""
    path = Path(path)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"] == config.initial_model_config():
            return saved
    model = train_initial_model(store, architecture, seed, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    return model


def run_longitudinal(store: SoilData, architecture: str, method: str, seed: int, config: ExperimentConfig,
                     initial_state: State) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Percorre a agenda: em cada rodada, treino com os alvos já observados, agregação e previsão do período seguinte.

    - ``fedavg``: modelo global único;
    - ``fedper``: a cabeça de cada estação é privada e preservada entre rodadas;
    - ``finetune``: agrega como o FedAvg e, antes de prever, cada estação que treinou na rodada ajusta uma cópia
      do modelo global com os próprios exemplos (a cópia é descartada);
    - ``local``: cada estação mantém um modelo completo, treinado só com os próprios dados;
    - ``central``: exemplos de todas as estações reunidos em um único treino.

    Devolve as métricas por estação e rodada, o registro de treino e as previsões por exemplo.
    """
    set_seed(seed)
    client_ids = sorted(store.client_ids)
    factory = lambda: build_model(architecture)
    train_kw = {"seed": seed, "lr": config.learning_rate, "epochs": config.local_epochs}

    split = factory(); split.load_state_dict(initial_state)
    global_state = split.shared_state_dict() if method == "fedper" else initial_state
    initial_head = split.private_state_dict()
    private_heads: dict[str, State] = {}     # fedper: cabeça de cada estação
    local_states: dict[str, State] = {}      # local: modelo completo de cada estação

    def client_model(sid: str):
        model = factory()
        if method == "fedper":
            model.load_partial_state_dict(global_state)
            model.load_partial_state_dict(private_heads.get(sid, initial_head))
        elif method == "local":
            model.load_state_dict(local_states.get(sid, initial_state))
        else:
            model.load_state_dict(global_state)
        return model

    metrics, train_log, predictions = [], [], []
    for _, round_row in store.schedule.iterrows():
        r, cutoff = int(round_row["round"]), pd.Timestamp(round_row["cutoff"])
        train: dict[str, Examples] = {}

        if r > 0:
            available = {sid: store.train_examples(sid, cutoff, postdeployment_only=True) for sid in client_ids}
            for sid, examples in available.items():
                assert_causal(examples, cutoff, sid, store.deployment)
            train = {sid: ex for sid, ex in available.items() if len(ex) >= config.min_train_examples}
            if method in ("fedavg", "finetune"):
                global_state, losses = fedavg_round(factory, global_state, train, config, round_idx=r, **train_kw)
            elif method == "central":
                global_state, losses = central_round(factory, global_state, train, config, round_idx=r, **train_kw)
            elif method == "fedper":
                global_state, losses = fedper_round(factory, global_state, private_heads, initial_head,
                                                    train, config, round_idx=r, **train_kw)
            else:
                losses = {}
                for sid in sorted(train):
                    local_states[sid], loss = fedavg_round(factory, local_states.get(sid, initial_state),
                                                           {sid: train[sid]}, config, round_idx=r, **train_kw)
                    losses.update(loss)
            for sid, examples in available.items():
                if len(examples):
                    train_log.append({
                        "StationID": sid, "round": r, "cutoff": cutoff.date().isoformat(),
                        "available_examples": len(examples), "trained": sid in losses,
                        "max_target_date": pd.Timestamp(examples.target_date.max()).date().isoformat(),
                        "train_loss": losses.get(sid, float("nan")),
                    })

        for sid in client_ids:
            ex = store.eval_examples(sid, round_row["eval_start"], round_row["eval_end"])
            if len(ex) == 0:
                continue
            if ex.t0.min() <= np.datetime64(cutoff):
                raise AssertionError(f"avaliação de {sid} começa antes do corte da rodada {r}")
            model = client_model(sid)
            if method == "finetune" and sid in train:
                train_local(model, train[sid], config, seed=client_seed(seed, sid, r) + 1,
                            lr=config.learning_rate, epochs=config.local_epochs)
            pred = predict(model, ex, config)
            err, perr = pred - ex.y, ex.persistence - ex.y
            metrics.append({
                "StationID": sid, "architecture": architecture, "method": method, "seed": seed, "round": r,
                "n": len(ex), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae_persistence": float(np.mean(np.abs(perr))),
                "rmse_persistence": float(np.sqrt(np.mean(perr ** 2))),
            })
            predictions.append(pd.DataFrame({"StationID": sid, "round": r, "t0": ex.t0, "y": ex.y,
                                             "pred": pred, "persistence": ex.persistence}))

    return pd.DataFrame(metrics), pd.DataFrame(train_log), pd.concat(predictions, ignore_index=True)


def run_single(config: ExperimentConfig, name: str, architecture: str, method: str, seed: int) -> str:
    """Executa uma combinação (arquitetura, método, semente) e grava os resultados em ``results/<name>/``."""
    torch.set_num_threads(1)
    out = PROJECT_ROOT / "results" / name
    run = run_name(architecture, method, seed)
    if (out / "runs" / f"{run}.csv").exists():
        return f"{run}: já existe"
    store = SoilData(config)
    model = initial_model(store, architecture, seed, config,
                          PROJECT_ROOT / "checkpoints" / name / f"{architecture}_seed{seed}.pt")
    started = time.perf_counter()
    metrics, train_log, predictions = run_longitudinal(store, architecture, method, seed, config, model["state"])
    runtime = time.perf_counter() - started
    if not np.isfinite(metrics[["mae", "rmse"]].to_numpy()).all():
        raise RuntimeError(f"{run}: métricas não finitas")
    for folder in ("runs", "train_log", "predictions"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    train_log.to_csv(out / "train_log" / f"{run}.csv", index=False)
    predictions.to_parquet(out / "predictions" / f"{run}.parquet", index=False)
    (out / "runs" / f"{run}.json").write_text(json.dumps({
        "architecture": architecture, "method": method, "seed": seed,
        "runtime_seconds": runtime, "initial_model_rounds": model["rounds"], "config": config.to_dict(),
    }, indent=2), encoding="utf-8")
    metrics.to_csv(out / "runs" / f"{run}.csv", index=False)
    return f"{run}: {len(metrics)} linhas em {runtime / 60:.1f} min"
