"""Gera os dados processados a partir dos arquivos brutos.

    python scripts/prepare_data.py curated            # data/raw/mendeley -> data/processed/curated
    python scripts/prepare_data.py ped [--reuse]      # data/raw/ped -> data/processed/ped

O conjunto curado deve ser preparado antes: a PED usa dele a data de implantação, a textura do solo e o grupo
de desenvolvimento. Com ``--reuse``, o painel diário lido dos zips (data/interim/ped_daily.parquet) é reaproveitado.
"""
from __future__ import annotations

import argparse

import pandas as pd

from soil_learning import ped
from soil_learning.config import PROJECT_ROOT
from soil_learning.preparation import prepare

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
INTERIM = PROJECT_ROOT / "data" / "interim" / "ped_daily.parquet"


def summary(stations: pd.DataFrame, schedule: pd.DataFrame) -> str:
    flags = stations[["eligible", "development", "full_network", "principal"]].sum()
    return (f"estações: {len(stations)} | " + " | ".join(f"{k}: {v}" for k, v in flags.items())
            + f" | rodadas: {len(schedule)} | implantação: {schedule['cutoff'].iloc[0]:%Y-%m-%d}"
            + f" | último corte: {schedule['cutoff'].iloc[-1]:%Y-%m-%d}")


def prepare_curated() -> None:
    out = PROCESSED / "curated"
    panel, stations, schedule = prepare(RAW / "mendeley")
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "daily_panel.parquet", index=False)
    stations.to_parquet(out / "stations.parquet", index=False)
    schedule.to_csv(out / "round_schedule.csv", index=False)
    print(summary(stations, schedule))


def prepare_ped(reuse: bool) -> None:
    out = PROCESSED / "ped"
    curated = pd.read_parquet(PROCESSED / "curated" / "stations.parquet")
    curated["StationID"] = curated["StationID"].astype(str)
    deployment = pd.Timestamp(pd.read_csv(PROCESSED / "curated" / "round_schedule.csv").iloc[0]["cutoff"])
    registry = pd.read_csv(RAW / "ped_stations.csv", dtype={"codestacao": str})

    if reuse and INTERIM.exists():
        daily = pd.read_parquet(INTERIM)
    else:
        daily = ped.read_raw(RAW / "ped")
        INTERIM.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(INTERIM, index=False)
    panel, report = ped.daily_qc(daily)
    panel, segment_report = ped.segment_qc(panel)
    daily_panel, stations, schedule = ped.build_design(panel, registry, curated, deployment)

    out.mkdir(parents=True, exist_ok=True)
    daily_panel.to_parquet(out / "daily_panel.parquet", index=False)
    stations.to_parquet(out / "stations.parquet", index=False)
    schedule.to_csv(out / "round_schedule.csv", index=False, date_format="%Y-%m-%d")
    report.to_csv(out / "qc_report.csv", index=False)
    segment_report.to_csv(out / "qc_report_segment.csv", index=False)
    print(summary(stations, schedule))
    print(f"controle de qualidade: {out / 'qc_report.csv'} e {out / 'qc_report_segment.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("curated", "ped"))
    parser.add_argument("--reuse", action="store_true", help="reaproveita data/interim/ped_daily.parquet")
    args = parser.parse_args()
    prepare_curated() if args.dataset == "curated" else prepare_ped(args.reuse)


if __name__ == "__main__":
    main()
