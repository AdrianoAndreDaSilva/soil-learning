"""Treino local, agregação e rodadas de FedAvg, FedPer e do treino centralizado."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Callable

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .data import Examples
from .models import PersonalizableRegressor

State = OrderedDict[str, torch.Tensor]
ModelFactory = Callable[[], PersonalizableRegressor]


def client_seed(seed: int, station_id: str, round_idx: int) -> int:
    token = f"{seed}|{station_id}|{round_idx}".encode()
    return (seed + int(hashlib.sha1(token).hexdigest()[:8], 16)) % (2**31 - 1)


def cpu_state(model: nn.Module) -> State:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def weighted_average(states: list[tuple[State, int]]) -> State:
    """Média dos estados ponderada pelo número de exemplos de cada cliente (acumulada em float64)."""
    total = float(sum(weight for _, weight in states))
    out = OrderedDict()
    for key, ref in states[0][0].items():
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for state, weight in states:
            acc += state[key].to(torch.float64) * (weight / total)
        out[key] = acc.to(ref.dtype)
    return out


def _shuffled_batches(n: int, batch_size: int) -> list[torch.Tensor]:
    # Mesma sequência de sorteios de um DataLoader com shuffle=True, sem montar exemplo a exemplo.
    torch.empty((), dtype=torch.int64).random_()
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    order = torch.randperm(n, generator=generator)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def train_local(model: PersonalizableRegressor, examples: Examples, config: ExperimentConfig, *,
                seed: int, lr: float, epochs: int) -> float:
    """Treino com alvo residual ``(y − persistência) / residual_scale``; devolve a perda média.

    O otimizador (Adam) é criado a cada chamada, isto é, reiniciado a cada cliente e rodada.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    loss_fn = nn.HuberLoss(delta=config.huber_delta / config.residual_scale)
    dynamic = torch.from_numpy(examples.dynamic)
    static = torch.from_numpy(examples.static)
    target = torch.from_numpy(((examples.y - examples.persistence) / config.residual_scale).astype(np.float32))
    losses = []
    for _ in range(epochs):
        for idx in _shuffled_batches(len(examples), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(dynamic[idx], static[idx]), target[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
    return float(np.mean(torch.stack(losses).numpy().astype(np.float64)))


@torch.no_grad()
def predict(model: PersonalizableRegressor, examples: Examples, config: ExperimentConfig) -> np.ndarray:
    """Umidade prevista em t0 + H (m³/m³): persistência mais o resíduo estimado."""
    if len(examples) == 0:
        return np.empty(0, dtype=np.float32)
    model.eval()
    dynamic, static = torch.from_numpy(examples.dynamic), torch.from_numpy(examples.static)
    out = torch.cat([model(dynamic[i:i + 256], static[i:i + 256]) for i in range(0, len(examples), 256)])
    residual = out.numpy().astype(np.float32)
    return (residual * config.residual_scale + examples.persistence).astype(np.float32)


def fedavg_round(factory: ModelFactory, global_state: State, client_data: dict[str, Examples],
                 config: ExperimentConfig, *, seed: int, round_idx: int, lr: float, epochs: int
                 ) -> tuple[State, dict[str, float]]:
    """Cada cliente treina o modelo inteiro; o servidor agrega todos os parâmetros."""
    updates, losses = [], {}
    for station_id in sorted(client_data):
        model = factory(); model.load_state_dict(global_state)
        losses[station_id] = train_local(model, client_data[station_id], config, lr=lr, epochs=epochs,
                                         seed=client_seed(seed, station_id, round_idx))
        updates.append((cpu_state(model), len(client_data[station_id])))
    return (weighted_average(updates) if updates else global_state), losses


def fedper_round(factory: ModelFactory, global_shared: State, private_heads: dict[str, State],
                 initial_head: State, client_data: dict[str, Examples], config: ExperimentConfig,
                 *, seed: int, round_idx: int, lr: float, epochs: int) -> tuple[State, dict[str, float]]:
    """Só a base compartilhada é agregada; a cabeça de cada cliente fica em ``private_heads``."""
    updates, losses = [], {}
    for station_id in sorted(client_data):
        model = factory()
        model.load_partial_state_dict(global_shared)
        model.load_partial_state_dict(private_heads.get(station_id, initial_head))
        losses[station_id] = train_local(model, client_data[station_id], config, lr=lr, epochs=epochs,
                                         seed=client_seed(seed, station_id, round_idx))
        updates.append((model.shared_state_dict(), len(client_data[station_id])))
        private_heads[station_id] = model.private_state_dict()
    return (weighted_average(updates) if updates else global_shared), losses


def central_round(factory: ModelFactory, global_state: State, client_data: dict[str, Examples],
                  config: ExperimentConfig, *, seed: int, round_idx: int, lr: float, epochs: int
                  ) -> tuple[State, dict[str, float]]:
    """Referência sem federação: os exemplos de todas as estações são reunidos e um único modelo treina sobre
    eles, com a mesma taxa, as mesmas épocas e os mesmos exemplos do FedAvg."""
    if not client_data:
        return global_state, {}
    ids = sorted(client_data)
    parts = [client_data[sid] for sid in ids]
    pooled = Examples(*(np.concatenate([getattr(p, f) for p in parts])
                        for f in ("dynamic", "static", "y", "persistence", "t0", "target_date")))
    model = factory(); model.load_state_dict(global_state)
    loss = train_local(model, pooled, config, lr=lr, epochs=epochs, seed=client_seed(seed, "central", round_idx))
    return cpu_state(model), {sid: loss for sid in ids}
