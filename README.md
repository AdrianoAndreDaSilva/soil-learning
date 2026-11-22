# soil-learning

[English](README.en.md) · [Español](README.es.md)

Código, dados e resultados de referência do Trabalho de Conclusão de Curso *Personalização no aprendizado
federado para previsão da umidade do solo: FedAvg x FedPer na rede do CEMADEN* (Adriano André da Silva e
Juliano Varella de Carvalho, Ciência da Computação, Universidade Feevale, 2026).

## Visão geral

Prever a umidade do solo com duas semanas de antecedência ajuda a antecipar secas agrícolas no semiárido. Modelos
de previsão melhoram com dados de muitas estações, mas reunir esses dados nem sempre é possível; o aprendizado
federado treina um modelo comum sem que os dados saiam de cada estação. Como as estações diferem em solo, sensor e
chuva, um modelo único pode atender mal parte delas, o que motiva a personalização.

O estudo compara o **FedAvg**, que treina um único modelo global, com o **FedPer**, que mantém privada em cada
estação a cabeça de regressão do modelo, na previsão da umidade volumétrica a 20 cm com 14 dias de antecedência.
Cada estação de monitoramento do CEMADEN é um cliente federado, e a operação da rede é simulada ao longo de dez
anos. Duas questões são avaliadas: (1) se o FedPer erra menos que o FedAvg e (2) se a vantagem do FedPer cresce com
a heterogeneidade da estação.

## Metodologia

- **Dados.** Séries horárias de umidade a 20 cm e de chuva da Plataforma de Entrega de Dados (PED) do CEMADEN,
  de novembro de 2014 a outubro de 2026, com controle de qualidade próprio (328 estações avaliadas). As comparações
  são repetidas no conjunto curado de Zeri et al. (2020), de 2015 a 2019 (156 estações). Detalhes em
  [`data/README.md`](data/README.md).
- **Tarefa.** Janela de 30 dias com seis canais: variação da umidade em relação à origem, chuva, duas máscaras de
  disponibilidade e seno e cosseno do dia do ano. A saída é residual: ŷ(t0 + 14) = y(t0) + 0,05·f(X).
- **Modelos.** LSTM e TCN com cerca de 15 mil parâmetros; a cabeça de 545 parâmetros é a parte privada no FedPer.
- **Simulação.** Implantação em 29/07/2016. Um modelo inicial é treinado por FedAvg com os dados anteriores, por 1 ou
  por 20 rodadas de pré-treino. Na fase longitudinal (132 rodadas de 28 dias na PED, 141 semanais no conjunto curado),
  cada estação treina uma época por rodada com taxa 10⁻⁴, só com alvos já observados, e prevê o período seguinte.
- **Referências.** Persistência (repetir a umidade atual), regressão linear causal, ajuste fino local, treinamento
  só local e treinamento centralizado.
- **Avaliação.** MAE por estação, média de cinco sementes (42, 123, 456, 789, 2026), expresso como razão sobre a
  persistência. Wilcoxon pareado por estação com correção de Holm, IC de 95% por reamostragem de estações e
  correlação de Spearman entre a heterogeneidade *H* (distância de Wasserstein) e a diferença FedAvg − FedPer.
- **Análises complementares incluídas.** FedAvg com a chuva dos 14 dias seguintes conhecida e teste de orçamento de
  treino.

## Estrutura do repositório

```text
configs/                 uma configuração JSON por experimento
data/                    dados brutos e processados (ver data/README.md)
scripts/
  download_ped.py        agenda e baixa os dados da PED (exige cadastro)
  prepare_data.py        brutos -> data/processed/{curated,ped}
  run_experiment.py      executa uma configuração (arquiteturas × métodos × sementes)
  analyze.py             tabelas de resultados e comparação com a referência
src/soil_learning/
  preparation.py, ped.py controle de qualidade, painel diário, elegibilidade, heterogeneidade e agenda
  data.py                exemplos supervisionados por estação
  models.py              LSTM e TCN com base compartilhada e cabeça privada
  federated.py           treino local, agregação, FedAvg, FedPer e centralizado
  experiment.py          modelo inicial e simulação longitudinal
  baselines.py           regressão linear de referência
  analysis.py            métricas, intervalos e testes
tests/                   testes do pacote
results/reference/       tabelas de referência (resultados do artigo)
```

As execuções gravam em `results/<configuração>/` e `checkpoints/<configuração>/`, e a análise em
`results/analysis/`; essas pastas não são versionadas.

## Requisitos

- Python 3.11 ou superior. Dependências fixadas em [`pyproject.toml`](pyproject.toml): PyTorch 2.14.0 (CPU),
  NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, pyarrow 25.0.1 e openpyxl 3.1.5.
- Só CPU, com um thread por processo; não há uso de GPU. A conferência deste repositório foi feita com Python 3.11
  em Windows 11, com 16 núcleos.
- Custo da fase longitudinal medido nas execuções originais: cerca de 1 a 1,8 hora por execução na PED e de 20 a
  45 minutos no conjunto curado, em um núcleo; cerca de 190 horas de CPU para todas as configurações. O pré-treino de
  20 rodadas leva cerca de 10 minutos por modelo na PED.
- Espaço em disco: cerca de 1 GB para as saídas das configurações principais (previsões e registros de treino).

## Reprodução dos experimentos

### 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
pytest
```

### 2. Dados (opcional)

Os dados processados já estão em `data/processed/`. Para refazê-los a partir dos brutos:

```bash
python scripts/prepare_data.py curated   # data/raw/mendeley -> data/processed/curated (cerca de 6 min)
python scripts/prepare_data.py ped       # data/raw/ped -> data/processed/ped (exige os zips da PED)
```

O conjunto curado bruto está versionado. Os arquivos da PED exigem cadastro na plataforma; o procedimento está em
[`data/README.md`](data/README.md) e no cabeçalho de `scripts/download_ped.py`. Refeitos a partir dos mesmos
arquivos, os dois conjuntos processados são idênticos aos versionados.

### 3. Experimentos

```bash
python scripts/run_experiment.py --config configs/ped_r01.json
python scripts/run_experiment.py --config configs/ped_r20.json
python scripts/run_experiment.py --config configs/curated_r01.json
python scripts/run_experiment.py --config configs/curated_r20.json
python scripts/run_experiment.py --config configs/ped_r20_future_rain.json
python scripts/run_experiment.py --config configs/budget_no_rain.json
python scripts/run_experiment.py --config configs/budget_future_rain.json
```

| Configuração | Dados | Modelo inicial | Métodos | Uso no artigo |
| --- | --- | --- | --- | --- |
| `ped_r01`, `ped_r20` | PED | 1 e 20 rodadas | FedAvg, FedPer, ajuste fino, só local, centralizado | Resultados principais |
| `curated_r01`, `curated_r20` | Conjunto curado | 1 e 20 rodadas | os mesmos | Verificação |
| `ped_r20_future_rain` | PED | 20 rodadas | FedAvg com a chuva futura | Teto com chuva futura |
| `budget_no_rain`, `budget_future_rain` | PED | 100 rodadas de 10 épocas | FedAvg (LSTM, 2 sementes; 3 épocas e taxa 10⁻³ por rodada) | Teste de orçamento |

Cada combinação de arquitetura, método e semente roda em um processo; `--workers` define quantos em paralelo
(padrão: número de núcleos menos dois).
Execuções já concluídas são mantidas; para refazer uma, apague o arquivo correspondente em
`results/<configuração>/runs/`. As opções `--architectures`, `--methods` e `--seeds` restringem a grade.

**Verificação rápida.** `--max-rounds N` interrompe a simulação na rodada N e grava em `results/<configuração>_maxN/`.
Como a simulação é sequencial, as previsões até a rodada N são as mesmas da execução completa. Por exemplo, para
conferir em poucos minutos uma combinação do conjunto curado:

```bash
python scripts/run_experiment.py --config configs/curated_r01.json --architectures lstm --seeds 42 --max-rounds 12
python scripts/analyze.py --check
```

A referência inclui as métricas dessa verificação (configuração `curated_r01_max12`, cinco métodos).

### 4. Análise e comparação com a referência

```bash
python scripts/analyze.py --check
```

O script gera as tabelas em `results/analysis/` com as previsões disponíveis e as compara com
`results/reference/`. A tabela `station_metrics.csv` tem o MAE de cada execução e estação e permite conferir
execuções isoladas de qualquer configuração; das interrompidas com `--max-rounds`, a referência só tem a
verificação rápida acima. As demais tabelas exigem todas as execuções das configurações correspondentes.

### Determinismo

As sementes fixam a inicialização dos pesos, a ordem dos lotes e o *dropout*, e o treino roda em CPU com
`torch.use_deterministic_algorithms(True)` e um thread por processo. Com as versões fixadas, o código deste
repositório reproduziu bit a bit, no mesmo computador, os modelos iniciais e as previsões das execuções originais
(conferidos os cinco métodos, as duas arquiteturas, os dois modelos iniciais e os dois conjuntos). Em outro
hardware, sistema operacional ou versão do PyTorch, diferenças de arredondamento em ponto flutuante podem alterar
os últimos dígitos.

## Resultados

Razão do MAE sobre o da persistência (média de cinco sementes; valores abaixo de 1 indicam erro menor que o de
repetir a umidade atual), com o modelo inicial de 20 rodadas:

| Conjunto | Modelo | FedAvg | FedPer | Ajuste fino | Só local | Centralizado |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PED, 2016–2026 | LSTM | 0,989 | 0,991 | 0,988 | 1,007 | 0,995 |
| PED, 2016–2026 | TCN | 0,983 | 0,986 | 0,982 | 1,016 | 0,975 |
| Conjunto curado, 2016–2019 | LSTM | 0,958 | 0,964 | 0,957 | 0,989 | 0,945 |
| Conjunto curado, 2016–2019 | TCN | 0,941 | 0,948 | 0,939 | 0,985 | 0,952 |

- **Questão 1.** O FedPer não superou o FedAvg em nenhum dos quatro cenários da PED (2 modelos × 2 modelos
  iniciais) e foi melhor em 40% a 43% das estações; a diferença favorece o FedAvg e é significativa após a correção
  de Holm em dois cenários.
- **Questão 2.** A correlação entre a heterogeneidade e a vantagem do FedPer é positiva e fraca
  (ρ = 0,105 na LSTM, IC de −0,002 a 0,216; ρ = 0,121 na TCN) e, na LSTM, não se distingue da variação entre sementes.

As tabelas completas estão em [`results/reference/`](results/reference/):

| Arquivo | Conteúdo (tabela do artigo) |
| --- | --- |
| `performance.csv` | Razão por método, IC, desvio entre sementes, fração de estações melhores que a persistência (desempenho; verificação no conjunto curado) |
| `method_pairs.csv` | FedPer × FedAvg, ajuste fino × FedAvg e × FedPer, só local e centralizado × FedAvg (questão 1, ajuste fino, colaboração) |
| `initial_model.csv` | Efeito do modelo inicial (1 contra 20 rodadas) |
| `heterogeneity.csv`, `heterogeneity_quartiles.csv`, `heterogeneity_seed_control.csv` | Questão 2, quartis de *H* e controle negativo com pares de sementes |
| `linear_reference.csv` | Regressão linear de referência |
| `future_rain.csv` | FedAvg sem e com a chuva futura conhecida (2016–2026) |
| `budget.csv`, `budget_ci.csv` | Teste de orçamento de treino |
| `station_metrics.csv` | MAE por execução e estação |

As colunas usam os nomes da análise original (por exemplo, `razao`, `ic_lo`, `ic_hi`, `p_holm_12`); as
famílias da correção de Holm estão indicadas nas próprias tabelas.

O artigo também traz análises diagnósticas que não fazem parte deste repositório: situações de chuva, semestres,
*gradient boosting* e LSTM centralizada com a chuva futura, contexto regional e do solo e sensibilidade aos zeros de
umidade.

## Citação

Os metadados de citação estão em [`CITATION.cff`](CITATION.cff) (o GitHub oferece a opção *Cite this repository*).
Para citar o software:

> da Silva, A. A. (2026). *soil-learning: FedAvg e FedPer na previsão da umidade do solo com estações do CEMADEN*
> (versão 1.0.0) [software]. <https://github.com/AdrianoAndreDaSilva/soil-learning>

Para citar o trabalho:

> da Silva, A. A.; de Carvalho, J. V. (2026). *Personalização no aprendizado federado para previsão da umidade do
> solo: FedAvg x FedPer na rede do CEMADEN*. Trabalho de Conclusão de Curso (Ciência da Computação), Universidade
> Feevale, Novo Hamburgo.

## Licença

O código é disponibilizado sob a [PolyForm Noncommercial License 1.0.0](LICENSE)
(SPDX: `PolyForm-Noncommercial-1.0.0`). A licença permite usar, estudar, modificar e redistribuir o código para
fins não comerciais, como pesquisa, estudo e uso pessoal, e permite o uso por instituições de ensino, organizações
públicas de pesquisa e órgãos governamentais, desde que a cópia seja acompanhada dos termos da licença e do aviso
de direitos autorais (*Required Notice*). Usos não contemplados por ela, em particular os comerciais, exigem
autorização separada do titular dos direitos. Por restringir o uso comercial, esta não é uma licença de código
aberto no sentido da definição da Open Source Initiative.

A licença não se aplica aos dados. O conjunto curado segue a licença CC BY 4.0 de seus autores. Os dados derivados
da PED são distribuídos conforme os Termos de Uso da plataforma, que os oferecem como serviço público gratuito e
pedem a citação da fonte: **DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC**; Centro Nacional de Monitoramento e Alertas de Desastres
Naturais - CEMADEN/MCTI, <https://www.gov.br/cemaden/> (ver [`data/README.md`](data/README.md)). As
bibliotecas usadas têm licenças próprias (BSD-3-Clause no PyTorch, NumPy, pandas e SciPy; Apache-2.0 no pyarrow;
MIT no openpyxl e no pytest).

## Agradecimentos

Este trabalho foi totalmente financiado pelo projeto Venture Science, suportado pelo Center for Embedded Devices and Research in Digital Agriculture (CEDRA) - SENAI-RS, com recursos financeiros oriundos do PPI IoT/Manufatura 4.0 / PPI HardwareBR do MCTI, através do Termo de Cooperação n.° 056/2023, firmado com a EMBRAPII.
