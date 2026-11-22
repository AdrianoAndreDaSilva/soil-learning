# soil-learning

[Português](README.md) · [Español](README.es.md)

Code, data and reference results of the undergraduate thesis (Trabalho de Conclusão de Curso) *Personalização no
aprendizado federado para previsão da umidade do solo: FedAvg x FedPer na rede do CEMADEN* (Personalization in
federated learning for soil moisture forecasting: FedAvg vs. FedPer in the CEMADEN network), by Adriano André da
Silva and Juliano Varella de Carvalho, Computer Science, Universidade Feevale, 2026. The thesis is written in
Portuguese; the main README is [README.md](README.md).

## Project overview

Forecasting soil moisture two weeks ahead helps anticipate agricultural droughts in the Brazilian semiarid region.
Forecasting models improve with data from many stations, but pooling those data is not always possible; federated
learning trains a shared model without the data leaving each station. Because stations differ in soil, sensor and
rainfall, a single model may serve some of them poorly, which motivates personalization.

The study compares **FedAvg**, which trains a single global model, with **FedPer**, which keeps the regression head
of the model private at each station, in forecasting volumetric soil moisture at 20 cm 14 days ahead. Each CEMADEN
monitoring station is a federated client, and the operation of the network is simulated over ten years. Two
questions are evaluated: (1) whether FedPer errs less than FedAvg and (2) whether the advantage of FedPer grows
with the heterogeneity of the station.

## Methodology

- **Data.** Hourly soil moisture at 20 cm and rainfall from the CEMADEN data delivery platform (PED, *Plataforma de
  Entrega de Dados*), November 2014 to October 2026, with a dedicated quality control (328 evaluated stations). The
  comparisons are repeated on the curated dataset of Zeri et al. (2020), 2015 to 2019 (156 stations). Details in
  [`data/README.md`](data/README.md) (Portuguese).
- **Task.** 30-day window with six channels: change in moisture relative to the forecast origin, rainfall, two
  availability masks and the sine and cosine of the day of year. The output is residual:
  ŷ(t0 + 14) = y(t0) + 0.05·f(X).
- **Models.** LSTM and TCN with about 15 thousand parameters; the 545-parameter head is the private part in FedPer.
- **Simulation.** Deployment on 2016-07-29. An initial model is trained by FedAvg on the earlier data, for 1 or for
  20 pre-training rounds. In the longitudinal phase (132 rounds of 28 days on the PED, 141 weekly rounds on the curated
  dataset), each station trains one epoch per round with learning rate 10⁻⁴, using only targets already observed, and
  forecasts the following period.
- **Baselines.** Persistence (repeating the current moisture), a causal linear regression, local fine-tuning,
  local-only training and centralized training.
- **Evaluation.** MAE per station, averaged over five seeds (42, 123, 456, 789, 2026), expressed as a ratio to the
  persistence MAE. Paired Wilcoxon tests per station with Holm correction, 95% confidence intervals by resampling
  stations, and the Spearman correlation between the heterogeneity *H* (Wasserstein distance) and the FedAvg − FedPer
  difference.
- **Complementary analyses included.** FedAvg with the rainfall of the following 14 days known, and a training budget
  test.

## Repository structure

```text
configs/                 one JSON configuration per experiment
data/                    raw and processed data (see data/README.md)
scripts/
  download_ped.py        schedules and downloads the PED data (requires an account)
  prepare_data.py        raw -> data/processed/{curated,ped}
  run_experiment.py      runs one configuration (architectures × methods × seeds)
  analyze.py             result tables and comparison with the reference
src/soil_learning/
  preparation.py, ped.py quality control, daily panel, eligibility, heterogeneity and round schedule
  data.py                supervised examples per station
  models.py              LSTM and TCN with shared base and private head
  federated.py           local training, aggregation, FedAvg, FedPer and centralized training
  experiment.py          initial model and longitudinal simulation
  baselines.py           linear reference regression
  analysis.py            metrics, intervals and tests
tests/                   package tests
results/reference/       reference tables (results of the thesis)
```

Runs write to `results/<configuration>/` and `checkpoints/<configuration>/`, and the analysis to
`results/analysis/`; these folders are not versioned. Code comments, messages and table column names are in
Portuguese.

## Requirements

- Python 3.11 or later. Dependencies pinned in [`pyproject.toml`](pyproject.toml): PyTorch 2.14.0 (CPU),
  NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, pyarrow 25.0.1 and openpyxl 3.1.5.
- CPU only, one thread per process; no GPU is used. This repository was checked with Python 3.11 on Windows 11 with
  16 cores.
- Cost of the longitudinal phase measured in the original runs: about 1 to 1.8 hours per run on the PED and 20 to
  45 minutes on the curated dataset, on one core; about 190 CPU hours for all configurations. The 20-round
  pre-training takes about 10 minutes per model on the PED.
- Disk space: about 1 GB for the outputs of the main configurations (predictions and training logs).

## Reproducing the experiments

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
pytest
```

### 2. Data (optional)

The processed data are already in `data/processed/`. To rebuild them from the raw files:

```bash
python scripts/prepare_data.py curated   # data/raw/mendeley -> data/processed/curated (about 6 min)
python scripts/prepare_data.py ped       # data/raw/ped -> data/processed/ped (requires the PED zip files)
```

The raw curated dataset is versioned. The PED files require an account on the platform; the procedure is described
in [`data/README.md`](data/README.md) and in the header of `scripts/download_ped.py`. Rebuilt from the same files,
both processed datasets are identical to the versioned ones.

### 3. Experiments

```bash
python scripts/run_experiment.py --config configs/ped_r01.json
python scripts/run_experiment.py --config configs/ped_r20.json
python scripts/run_experiment.py --config configs/curated_r01.json
python scripts/run_experiment.py --config configs/curated_r20.json
python scripts/run_experiment.py --config configs/ped_r20_future_rain.json
python scripts/run_experiment.py --config configs/budget_no_rain.json
python scripts/run_experiment.py --config configs/budget_future_rain.json
```

| Configuration | Data | Initial model | Methods | Use in the thesis |
| --- | --- | --- | --- | --- |
| `ped_r01`, `ped_r20` | PED | 1 and 20 rounds | FedAvg, FedPer, fine-tuning, local only, centralized | Main results |
| `curated_r01`, `curated_r20` | Curated dataset | 1 and 20 rounds | same | Verification |
| `ped_r20_future_rain` | PED | 20 rounds | FedAvg with future rainfall | Future rainfall ceiling |
| `budget_no_rain`, `budget_future_rain` | PED | 100 rounds of 10 epochs | FedAvg (LSTM, 2 seeds; 3 epochs and rate 10⁻³ per round) | Training budget test |

Each combination of architecture, method and seed runs in its own process; `--workers` sets how many run in
parallel (default: number of cores minus two).
Completed runs are kept; to redo one, delete its file in `results/<configuration>/runs/`. The options
`--architectures`, `--methods` and `--seeds` restrict the grid.

**Quick check.** `--max-rounds N` stops the simulation at round N and writes to `results/<configuration>_maxN/`.
Because the simulation is sequential, the predictions up to round N are the same as in the full run. For example,
to check one combination of the curated dataset in a few minutes:

```bash
python scripts/run_experiment.py --config configs/curated_r01.json --architectures lstm --seeds 42 --max-rounds 12
python scripts/analyze.py --check
```

The reference includes the metrics of this check (configuration `curated_r01_max12`, five methods).

### 4. Analysis and comparison with the reference

```bash
python scripts/analyze.py --check
```

The script writes the tables to `results/analysis/` from the available predictions and compares them with
`results/reference/`. The table `station_metrics.csv` holds the MAE of every run and station and allows checking
individual runs of any configuration; of the runs stopped with `--max-rounds`, the reference only covers the quick
check above. The other tables require all runs of the corresponding configurations.

### Determinism

The seeds fix the weight initialization, the batch order and dropout, and training runs on CPU with
`torch.use_deterministic_algorithms(True)` and one thread per process. With the pinned versions, the code in this
repository reproduced bit for bit, on the same computer, the initial models and the predictions of the original runs
(checked for the five methods, both architectures, both initial models and both datasets). On other hardware,
operating systems or PyTorch versions, floating-point rounding may change the last digits.

## Results

Ratio of the MAE to the persistence MAE (mean of five seeds; values below 1 mean a smaller error than repeating the
current moisture), with the 20-round initial model:

| Dataset | Model | FedAvg | FedPer | Fine-tuning | Local only | Centralized |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PED, 2016–2026 | LSTM | 0.989 | 0.991 | 0.988 | 1.007 | 0.995 |
| PED, 2016–2026 | TCN | 0.983 | 0.986 | 0.982 | 1.016 | 0.975 |
| Curated, 2016–2019 | LSTM | 0.958 | 0.964 | 0.957 | 0.989 | 0.945 |
| Curated, 2016–2019 | TCN | 0.941 | 0.948 | 0.939 | 0.985 | 0.952 |

- **Question 1.** FedPer did not outperform FedAvg in any of the four PED scenarios (2 models × 2 initial models)
  and was better at 40% to 43% of the stations; the difference favours FedAvg and is significant after Holm
  correction in two scenarios.
- **Question 2.** The correlation between heterogeneity and the advantage of FedPer is positive and weak
  (ρ = 0.105 for the LSTM, CI −0.002 to 0.216; ρ = 0.121 for the TCN) and, for the LSTM, cannot be distinguished from
  the variation between seeds.

The full tables are in [`results/reference/`](results/reference/):

| File | Content |
| --- | --- |
| `performance.csv` | Ratio per method, CI, standard deviation across seeds, share of stations better than persistence |
| `method_pairs.csv` | FedPer × FedAvg, fine-tuning × FedAvg and × FedPer, local only and centralized × FedAvg |
| `initial_model.csv` | Effect of the initial model (1 vs. 20 rounds) |
| `heterogeneity.csv`, `heterogeneity_quartiles.csv`, `heterogeneity_seed_control.csv` | Question 2, quartiles of *H* and negative control with seed pairs |
| `linear_reference.csv` | Linear reference regression |
| `future_rain.csv` | FedAvg without and with known future rainfall (2016–2026) |
| `budget.csv`, `budget_ci.csv` | Training budget test |
| `station_metrics.csv` | MAE per run and station |

The columns keep the names of the original analysis, in Portuguese (for example `razao` = ratio, `ic_lo`/`ic_hi` =
CI bounds, `p_holm_12` = Holm-adjusted p-value within a family of 12 tests); the Holm families are indicated in the
tables.

The thesis also reports diagnostic analyses that are not part of this repository: rainfall situations, semesters,
gradient boosting and a centralized LSTM with future rainfall, regional and soil context, and sensitivity to zero
moisture readings.

## Citation

Citation metadata are in [`CITATION.cff`](CITATION.cff) (GitHub offers *Cite this repository*). To cite the
software:

> da Silva, A. A. (2026). *soil-learning: FedAvg e FedPer na previsão da umidade do solo com estações do CEMADEN*
> (version 1.0.0) [software]. <https://github.com/AdrianoAndreDaSilva/soil-learning>

To cite the thesis:

> da Silva, A. A.; de Carvalho, J. V. (2026). *Personalização no aprendizado federado para previsão da umidade do
> solo: FedAvg x FedPer na rede do CEMADEN*. Undergraduate thesis (Computer Science), Universidade Feevale, Novo
> Hamburgo, Brazil.

## License

The code is made available under the [PolyForm Noncommercial License 1.0.0](LICENSE)
(SPDX: `PolyForm-Noncommercial-1.0.0`). The license allows using, studying, modifying and redistributing the code
for noncommercial purposes, such as research, study and personal use, and allows use by educational institutions,
public research organizations and government institutions, provided that copies carry the license terms and the
copyright notice (*Required Notice*). Uses not covered by the license, in particular commercial uses, require
separate permission from the copyright holder. Because it restricts commercial use, this is not an open source
license under the Open Source Initiative definition.

The license does not apply to the data. The curated dataset is under its authors' CC BY 4.0 license. The data
derived from the PED are distributed under the platform's Terms of Use, which provide them as a free public service
and ask that the source be cited, in Portuguese: **DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC**; Centro Nacional de Monitoramento e Alertas de Desastres
Naturais - CEMADEN/MCTI, <https://www.gov.br/cemaden/> (see [`data/README.md`](data/README.md)). The libraries
used have their own licenses (BSD-3-Clause for PyTorch, NumPy, pandas and SciPy; Apache-2.0 for pyarrow; MIT for
openpyxl and pytest).

## Acknowledgements

Funding statement, in its required original wording:

Este trabalho foi totalmente financiado pelo projeto Venture Science, suportado pelo Center for Embedded Devices and Research in Digital Agriculture (CEDRA) - SENAI-RS, com recursos financeiros oriundos do PPI IoT/Manufatura 4.0 / PPI HardwareBR do MCTI, através do Termo de Cooperação n.° 056/2023, firmado com a EMBRAPII.
