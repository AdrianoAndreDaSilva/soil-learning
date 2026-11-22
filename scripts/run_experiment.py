"""Executa a grade de uma configuração (arquiteturas × métodos × sementes) em processos paralelos.

    python scripts/run_experiment.py --config configs/ped_r20.json --workers 8

Os resultados vão para results/<configuração>/ e os modelos iniciais para checkpoints/<configuração>/.
Execuções já concluídas são mantidas; para refazer uma, apague o seu arquivo em results/<configuração>/runs/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
import os
from pathlib import Path

from soil_learning.config import PROJECT_ROOT, ExperimentConfig


def build_initial_model(config: ExperimentConfig, name: str, architecture: str, seed: int) -> str:
    import torch
    from soil_learning.data import SoilData
    from soil_learning.experiment import initial_model

    torch.set_num_threads(1)
    path = PROJECT_ROOT / "checkpoints" / name / f"{architecture}_seed{seed}.pt"
    model = initial_model(SoilData(config), architecture, seed, config, path)
    return f"modelo inicial {architecture} semente {seed}: {model['rounds']} rodadas"


def run(config: ExperimentConfig, name: str, architecture: str, method: str, seed: int) -> str:
    from soil_learning.experiment import run_single
    return run_single(config, name, architecture, method, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-rounds", type=int, help="interrompe a agenda nesta rodada (verificação rápida)")
    parser.add_argument("--name", help="pasta de saída em results/ e checkpoints/ (padrão: nome da configuração, "
                                       "com o sufixo _max<N> quando --max-rounds é usado)")
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    overrides = {key: tuple(value) for key in ("architectures", "methods", "seeds") if (value := getattr(args, key))}
    name = args.name or Path(args.config).stem
    if args.max_rounds is not None:
        overrides["max_rounds"] = args.max_rounds
        name = args.name or f"{name}_max{args.max_rounds}"
    config = replace(config, **overrides)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        models = [pool.submit(build_initial_model, config, name, a, s)
                  for a, s in product(config.architectures, config.seeds)]
        for future in models:
            print(future.result(), flush=True)
        runs = [pool.submit(run, config, name, a, m, s)
                for a, s, m in product(config.architectures, config.seeds, config.methods)]
        for future in runs:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
