from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Campos que definem o modelo inicial; um modelo salvo só é reutilizado se todos coincidirem.
INITIAL_MODEL_KEYS = (
    "data_dir", "input_variant", "future_rain", "lookback_days", "horizon_days", "min_train_examples",
    "batch_size", "weight_decay", "huber_delta", "residual_scale", "initial_rounds", "initial_local_epochs",
    "initial_learning_rate",
)
INPUT_VARIANTS = ("relative", "absolute")
METHODS = ("fedavg", "fedper", "finetune", "local", "central")


@dataclass(frozen=True)
class ExperimentConfig:
    # Pasta dos dados processados, relativa à raiz do projeto.
    data_dir: str = "data/processed/ped"
    # Primeiro canal da janela: "relative" = (θ − θ0) / residual_scale, sem o nível absoluto (redes do estudo);
    # "absolute" = θ / 0,6 (usado só para extrair os descritores da regressão linear de referência).
    input_variant: str = "relative"
    # Teste com a chuva futura: a chuva observada em (t0, t0 + H] entra nas quatro entradas estáticas.
    future_rain: bool = False

    lookback_days: int = 30
    horizon_days: int = 14
    min_train_examples: int = 24
    batch_size: int = 32
    weight_decay: float = 1e-5
    huber_delta: float = 0.005      # m³/m³
    residual_scale: float = 0.05    # m³/m³

    # Modelo inicial: rodadas de FedAvg com os exemplos anteriores à implantação.
    initial_rounds: int = 20
    initial_local_epochs: int = 3
    initial_learning_rate: float = 1e-3

    # Fase longitudinal: épocas locais e taxa de aprendizagem por rodada.
    local_epochs: int = 1
    learning_rate: float = 1e-4

    seeds: tuple[int, ...] = (42, 123, 456, 789, 2026)
    architectures: tuple[str, ...] = ("lstm", "tcn")
    methods: tuple[str, ...] = METHODS
    # Interrompe a agenda nesta rodada (verificações rápidas).
    max_rounds: int | None = None

    def __post_init__(self) -> None:
        if self.input_variant not in INPUT_VARIANTS:
            raise ValueError(f"input_variant deve ser um de {INPUT_VARIANTS}")
        if unknown := set(self.methods) - set(METHODS):
            raise ValueError(f"métodos desconhecidos: {sorted(unknown)}")

    def initial_model_config(self) -> dict:
        return {key: getattr(self, key) for key in INITIAL_MODEL_KEYS}

    def to_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("seeds", "architectures", "methods"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
