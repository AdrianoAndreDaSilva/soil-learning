"""LSTM e TCN divididas em base compartilhada e cabeça de regressão (a parte privada no FedPer)."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

DYNAMIC_DIM = 6      # canais por dia da janela
STATIC_DIM = 4       # entradas estáticas (chuva futura; zeradas nas execuções principais)
PRIVATE_PREFIX = "head."


def is_private(key: str) -> bool:
    return key.startswith(PRIVATE_PREFIX)


class PersonalizableRegressor(nn.Module):
    def shared_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.state_dict().items() if not is_private(k))

    def private_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.state_dict().items() if is_private(k))

    def load_partial_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        current.update({k: v for k, v in state.items() if k in current})
        self.load_state_dict(current)


class LSTMRegressor(PersonalizableRegressor):
    def __init__(self, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=DYNAMIC_DIM, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.static_proj = nn.Sequential(nn.Linear(STATIC_DIM, 8), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_size + 8, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(dynamic)
        z = self.fusion(torch.cat([h[-1], self.static_proj(static)], dim=-1))
        return self.head(z).squeeze(-1)


class CausalConv1d(nn.Conv1d):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__(channels, channels, kernel_size, dilation=dilation, padding=0)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.relu(self.norm1(self.conv1(x))))
        x = self.dropout(F.relu(self.norm2(self.conv2(x))))
        return F.relu(x + residual)


class TCNRegressor(PersonalizableRegressor):
    def __init__(self, channels: int = 27, kernel_size: int = 3, dilations: tuple[int, ...] = (1, 2, 4),
                 dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.input_proj = nn.Conv1d(DYNAMIC_DIM, channels, kernel_size=1)
        self.encoder = nn.Sequential(*[TemporalBlock(channels, kernel_size, d, dropout) for d in dilations])
        self.static_proj = nn.Sequential(nn.Linear(STATIC_DIM, 8), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(channels + 8, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.input_proj(dynamic.transpose(1, 2)))[:, :, -1]
        z = self.fusion(torch.cat([x, self.static_proj(static)], dim=-1))
        return self.head(z).squeeze(-1)


def build_model(name: str) -> PersonalizableRegressor:
    if name == "lstm":
        return LSTMRegressor()
    if name == "tcn":
        return TCNRegressor()
    raise ValueError(f"arquitetura desconhecida: {name}")


def count_parameters(model: PersonalizableRegressor) -> tuple[int, int]:
    """Número de parâmetros compartilhados e privados."""
    shared = sum(v.numel() for v in model.shared_state_dict().values())
    private = sum(v.numel() for v in model.private_state_dict().values())
    return shared, private
