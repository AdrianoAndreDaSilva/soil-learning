"""Tabelas de resultados a partir das previsões em results/<configuração>/predictions/.

    python scripts/analyze.py                 # todas as análises com previsões disponíveis
    python scripts/analyze.py --check         # e compara com results/reference/

Saída em results/analysis/:

- performance.csv, method_pairs.csv, initial_model.csv: razões sobre a persistência, comparações entre métodos e
  efeito do modelo inicial (ped_r01, ped_r20, curated_r01, curated_r20);
- heterogeneity*.csv: questão 2 (mesmas configurações);
- linear_reference*.csv: regressão linear de referência (reajustada aqui, alguns minutos);
- future_rain.csv, budget.csv, budget_ci.csv: chuva futura e teste de orçamento (ped_r20, ped_r20_future_rain,
  budget_no_rain, budget_future_rain);
- station_metrics.csv: MAE por execução e estação de todas as previsões encontradas.
"""
from __future__ import annotations

import argparse

import pandas as pd

from soil_learning import analysis as A
from soil_learning.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "analysis"
REFERENCE = PROJECT_ROOT / "results" / "reference"


def save(tables: dict[str, pd.DataFrame]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
        print(f"  {name}.csv ({len(df)} linhas)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compara as tabelas geradas com results/reference/")
    args = parser.parse_args()

    missing = A.missing_runs("PED") + A.missing_runs("Curado")
    if missing:
        print(f"desempenho, heterogeneidade e regressão linear: pulados, faltam {len(missing)} execuções "
              f"(por exemplo {missing[0].relative_to(PROJECT_ROOT)})")
    else:
        print("desempenho e comparações entre métodos", flush=True)
        save(A.performance())
        print("heterogeneidade (questão 2)", flush=True)
        save(A.heterogeneity())
        print("regressão linear de referência", flush=True)
        save(A.linear_comparison())

    missing = A.future_rain_and_budget_missing()
    if missing:
        print(f"chuva futura e orçamento: pulados, faltam {len(missing)} execuções")
    else:
        print("chuva futura e teste de orçamento", flush=True)
        save(A.future_rain_and_budget())

    print("métricas por execução e estação", flush=True)
    save({"station_metrics": A.station_metrics()})

    if args.check:
        print("\ncomparação com results/reference/")
        for line in A.compare(OUT, REFERENCE):
            print(f"  {line}")


if __name__ == "__main__":
    main()
