# Dados

O estudo usa dois conjuntos de séries horárias de umidade volumétrica do solo e de chuva das estações do
CEMADEN no semiárido brasileiro. Os dados processados dos dois estão versionados, de modo que os experimentos
podem ser executados sem baixar nada. Os brutos do conjunto curado também estão versionados; os da PED não.

```text
data/
├── raw/
│   ├── mendeley/            conjunto curado, como publicado (versionado)
│   ├── ped/                 arquivos baixados da PED (fora do git)
│   ├── ped_requests.csv     pedidos à PED que formam os dados do estudo
│   └── ped_stations.csv     cadastro das estações da PED
├── interim/                 painel intermediário da PED (fora do git)
└── processed/
    ├── curated/             conjunto curado processado
    └── ped/                 PED processada (desenho do artigo)
```

Unidades: umidade volumétrica em m³/m³, precipitação em mm, horários em UTC.

## Conjunto curado (`raw/mendeley/`)

Zeri, M.; Costa, J. M.; Urbano, D.; Cuartas, L. A.; Ivo, A.; Marengo, J.; Alvalá, R. C. S. (2020).
*A soil moisture dataset over the Brazilian semiarid region*. Mendeley Data, versão 2.
DOI: [10.17632/xrk5rfcpvg.2](https://doi.org/10.17632/xrk5rfcpvg.2). Licença: CC BY 4.0.

Os arquivos são cópia do conjunto original, sem alteração de conteúdo. Apenas o catálogo (`_Stations.xlsx`) e a
pasta das estações (`PCDs_`) foram renomeados.

| Arquivo | Conteúdo |
| --- | --- |
| `stations_catalog.xlsx` | Catálogo das 360 estações: arquivo, município, estado, coordenadas e período |
| `stations/PCD_<tipo>_<UF>_<id>.xlsx` | Uma estação por arquivo. Aba `Data`: série horária (`Time`, `Precip`, `US10cm`, `US20cm` e, nas estações Agro, `US30cm` e `US40cm`). Aba `Info`: metadados e textura do solo (g/kg) |

## PED (`raw/ped/`, `raw/ped_requests.csv`, `raw/ped_stations.csv`)

Séries da Plataforma de Entrega de Dados do CEMADEN (<https://ped.cemaden.gov.br/>), estações Acqua (tipo 5) e
Agro (tipo 4) dos nove estados do semiárido, de novembro de 2014 a outubro de 2026.

Fonte dos dados, como pedem a plataforma e seus Termos de Uso: **DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC**.
Centro Nacional de Monitoramento e Alertas de Desastres Naturais - CEMADEN/MCTI, <https://www.gov.br/cemaden/>.

O acesso exige cadastro na PED, e os arquivos brutos (cerca de 800 MB) não estão no repositório. Para obtê-los,
agende pedidos novos com `scripts/download_ped.py` (os links dos pedidos expiram em 30 dias) e grave os zips em
`raw/ped/`.

- `ped_requests.csv`: os 168 pedidos agendados em 22/10/2026 (estado, tipo de estação, início e fim, número do
  pedido). Não contém credenciais. 163 foram concluídos e baixados. Os cinco pedidos anuais da Bahia/Acqua de
  2015 a abril de 2019 ficaram pendentes e repetem o bloco 2014–abril de 2019 (pedido 154802), que foi baixado.
- `ped_stations.csv`: cadastro das estações (código, município, UF, coordenadas e tipo), obtido em 22/10/2026 do
  serviço `pcds-cadastro/dados-cadastrais` da PED. Estações que não constam do cadastro usam os metadados do
  conjunto curado.

## Dados processados (`processed/`)

Gerados por `scripts/prepare_data.py curated` e `scripts/prepare_data.py ped`, nesta ordem (a PED usa do conjunto
curado a data de implantação, a textura do solo e o grupo de desenvolvimento). Cada pasta tem três arquivos.

**`daily_panel.parquet`**: uma linha por estação e dia, com `date`, `StationID`, `US20cm_mean` (média diária da
umidade a 20 cm, ausente com menos de 18 leituras horárias válidas) e `precip_mm` (total diário de chuva, ausente
com menos de 20 horas registradas).

**`stations.parquet`**: uma linha por estação.

| Coluna | Descrição |
| --- | --- |
| `City`, `State`, `Latitude`, `Longitude`, `station_type` | Metadados; tipo Acqua ou Agro |
| `sand_pct`, `silt_pct`, `clay_pct` | Textura do solo (%), do conjunto curado |
| `valid_days`, `coverage`, `exact_zero_fraction` | Dias com umidade válida, fração de dias válidos no período e fração de leituras horárias iguais a zero |
| `precip_annual_mm`, `precip_max_hourly_mm`, `precip_reliable` | Diagnóstico do pluviômetro; não confiável se a taxa horária excede 100 mm ou o total anual excede 3 vezes a mediana da rede |
| `eligible` | Pelo menos 365 dias válidos e até 30% de zeros; no conjunto curado, também cobertura de pelo menos 40% e textura completa |
| `H_w1_balanced`, `H_cosine_balanced`, `H_js_balanced` | Heterogeneidade anterior à implantação em relação à referência balanceada da rede (a do estudo é `H_w1_balanced`) |
| `mean`, `std`, `seasonal_amplitude`, `acf_7d` | Descritores da umidade anterior à implantação |
| `pre_examples`, `post_examples` | Exemplos com alvo até a implantação e com origem posterior a ela |
| `development` | Grupo de desenvolvimento (16 estações do conjunto curado, usadas na escolha dos hiperparâmetros) |
| `full_network` | Elegível com pelo menos 90 exemplos depois da implantação |
| `principal` | População avaliada: `full_network` fora do grupo de desenvolvimento |
| `in_mendeley`, `principal_original` | Só na PED: estação presente no conjunto curado e estação principal nele |

**`round_schedule.csv`**: agenda das rodadas. `cutoff` é a data de corte do treino; `eval_start` e `eval_end`
delimitam as origens de previsão avaliadas na rodada. A rodada 0 tem corte na implantação simulada (29/07/2016) e
avalia o modelo inicial.

| | Conjunto curado | PED |
| --- | --- | --- |
| Estações em `stations.parquet` | 360 | 548 |
| Estações em `daily_panel.parquet` | 360 | 350 (só as elegíveis) |
| Elegíveis / população principal | 174 / 156 | 350 / 328 |
| Rodadas | 142 semanais (último corte em 12/04/2019) | 133 de 28 dias (último corte em 11/09/2026) |
| Controle de qualidade | Faixa de 0 a 0,6 m³/m³ e horas mínimas por dia | Também picos, zeros por queda brusca, falha crônica, sensor travado, deriva, chuva anual implausível e regra por trecho (`ped.py`) |

Na PED, `qc_report.csv` e `qc_report_segment.csv` registram o que cada regra do controle de qualidade remove.

## Licenças

O conjunto curado é distribuído sob CC BY 4.0 por seus autores; a cópia em `raw/mendeley/` e os dados derivados em
`processed/curated/` mantêm essa licença e a atribuição acima.

Os Termos de Uso da PED (<https://ped.cemaden.gov.br/suporte/termouso>) disponibilizam os dados "como serviço
público gratuito a qualquer pessoa física ou jurídica" e pedem a citação da fonte em eventos, publicações e ações,
indicando o nome "Centro Nacional de Monitoramento e Alertas de Desastres Naturais - CEMADEN/MCTI" e o endereço
<https://www.gov.br/cemaden/>; a página inicial da plataforma pede ainda a menção "DADOS DA REDE OBSERVACIONAL DO
CEMADEN/MCTIC" sempre que os dados forem usados. Os termos não restringem a redistribuição. Os dados derivados em
`processed/ped/` e o cadastro em `raw/ped_stations.csv` são distribuídos com essa atribuição; os arquivos brutos não
são redistribuídos e devem ser obtidos na própria plataforma.

A licença do código do repositório (PolyForm Noncommercial 1.0.0) não se aplica aos dados.
