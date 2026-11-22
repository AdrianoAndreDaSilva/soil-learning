# soil-learning

[Português](README.md) · [English](README.en.md)

Código, datos y resultados de referencia del Trabajo de Conclusión de Curso (Trabalho de Conclusão de Curso)
*Personalização no aprendizado federado para previsão da umidade do solo: FedAvg x FedPer na rede do CEMADEN*
(Personalización en el aprendizaje federado para la predicción de la humedad del suelo: FedAvg frente a FedPer en
la red del CEMADEN), de Adriano André da Silva y Juliano Varella de Carvalho, Ciencia de la Computación, Universidade
Feevale, 2026. El trabajo está escrito en portugués; el README principal es [README.md](README.md).

## Descripción general

Predecir la humedad del suelo con dos semanas de anticipación ayuda a anticipar sequías agrícolas en la región
semiárida de Brasil. Los modelos de predicción mejoran con datos de muchas estaciones, pero reunir esos datos no
siempre es posible; el aprendizaje federado entrena un modelo común sin que los datos salgan de cada estación. Como
las estaciones difieren en suelo, sensor y lluvia, un único modelo puede atender mal a parte de ellas, lo que motiva
la personalización.

El estudio compara **FedAvg**, que entrena un único modelo global, con **FedPer**, que mantiene privada en cada
estación la cabeza de regresión del modelo, en la predicción de la humedad volumétrica a 20 cm con 14 días de
anticipación. Cada estación de monitoreo del CEMADEN es un cliente federado, y la operación de la red se simula a lo
largo de diez años. Se evalúan dos preguntas: (1) si FedPer comete menos error que FedAvg y (2) si la ventaja de
FedPer crece con la heterogeneidad de la estación.

## Metodología

- **Datos.** Series horarias de humedad a 20 cm y de lluvia de la plataforma de entrega de datos del CEMADEN (PED,
  *Plataforma de Entrega de Dados*), de noviembre de 2014 a octubre de 2026, con un control de calidad propio
  (328 estaciones evaluadas). Las comparaciones se repiten en el conjunto curado de Zeri et al. (2020), de 2015 a
  2019 (156 estaciones). Detalles en [`data/README.md`](data/README.md) (en portugués).
- **Tarea.** Ventana de 30 días con seis canales: variación de la humedad respecto del origen de la predicción,
  lluvia, dos máscaras de disponibilidad y el seno y el coseno del día del año. La salida es residual:
  ŷ(t0 + 14) = y(t0) + 0,05·f(X).
- **Modelos.** LSTM y TCN con unos 15 mil parámetros; la cabeza de 545 parámetros es la parte privada en FedPer.
- **Simulación.** Implantación el 29/07/2016. Un modelo inicial se entrena con FedAvg con los datos anteriores,
  durante 1 o 20 rondas de preentrenamiento. En la fase longitudinal (132 rondas de 28 días en la PED, 141 semanales
  en el conjunto curado), cada estación entrena una época por ronda con tasa 10⁻⁴, solo con objetivos ya observados,
  y predice el período siguiente.
- **Referencias.** Persistencia (repetir la humedad actual), regresión lineal causal, ajuste fino local,
  entrenamiento solo local y entrenamiento centralizado.
- **Evaluación.** MAE por estación, promedio de cinco semillas (42, 123, 456, 789, 2026), expresado como razón sobre
  el MAE de la persistencia. Prueba de Wilcoxon pareada por estación con corrección de Holm, intervalos de confianza
  del 95% por remuestreo de estaciones y correlación de Spearman entre la heterogeneidad *H* (distancia de
  Wasserstein) y la diferencia FedAvg − FedPer.
- **Análisis complementarios incluidos.** FedAvg con la lluvia de los 14 días siguientes conocida y prueba de
  presupuesto de entrenamiento.

## Estructura del repositorio

```text
configs/                 una configuración JSON por experimento
data/                    datos brutos y procesados (ver data/README.md)
scripts/
  download_ped.py        programa y descarga los datos de la PED (requiere registro)
  prepare_data.py        brutos -> data/processed/{curated,ped}
  run_experiment.py      ejecuta una configuración (arquitecturas × métodos × semillas)
  analyze.py             tablas de resultados y comparación con la referencia
src/soil_learning/
  preparation.py, ped.py control de calidad, panel diario, elegibilidad, heterogeneidad y calendario de rondas
  data.py                ejemplos supervisados por estación
  models.py              LSTM y TCN con base compartida y cabeza privada
  federated.py           entrenamiento local, agregación, FedAvg, FedPer y centralizado
  experiment.py          modelo inicial y simulación longitudinal
  baselines.py           regresión lineal de referencia
  analysis.py            métricas, intervalos y pruebas
tests/                   pruebas del paquete
results/reference/       tablas de referencia (resultados del trabajo)
```

Las ejecuciones escriben en `results/<configuración>/` y `checkpoints/<configuración>/`, y el análisis en
`results/analysis/`; estas carpetas no se versionan. Los comentarios del código, los mensajes y los nombres de las
columnas de las tablas están en portugués.

## Requisitos

- Python 3.11 o superior. Dependencias fijadas en [`pyproject.toml`](pyproject.toml): PyTorch 2.14.0 (CPU),
  NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, pyarrow 25.0.1 y openpyxl 3.1.5.
- Solo CPU, con un hilo por proceso; no se usa GPU. Este repositorio se comprobó con Python 3.11 en Windows 11 con
  16 núcleos.
- Costo de la fase longitudinal medido en las ejecuciones originales: entre 1 y 1,8 horas por ejecución en la PED y
  entre 20 y 45 minutos en el conjunto curado, en un núcleo; unas 190 horas de CPU para todas las configuraciones. El
  preentrenamiento de 20 rondas tarda unos 10 minutos por modelo en la PED.
- Espacio en disco: cerca de 1 GB para las salidas de las configuraciones principales (predicciones y registros de
  entrenamiento).

## Reproducción de los experimentos

### 1. Entorno

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
pytest
```

### 2. Datos (opcional)

Los datos procesados ya están en `data/processed/`. Para rehacerlos a partir de los brutos:

```bash
python scripts/prepare_data.py curated   # data/raw/mendeley -> data/processed/curated (unos 6 min)
python scripts/prepare_data.py ped       # data/raw/ped -> data/processed/ped (requiere los zip de la PED)
```

El conjunto curado bruto está versionado. Los archivos de la PED requieren registro en la plataforma; el
procedimiento está en [`data/README.md`](data/README.md) y en el encabezado de `scripts/download_ped.py`. Rehechos a
partir de los mismos archivos, los dos conjuntos procesados son idénticos a los versionados.

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

| Configuración | Datos | Modelo inicial | Métodos | Uso en el trabajo |
| --- | --- | --- | --- | --- |
| `ped_r01`, `ped_r20` | PED | 1 y 20 rondas | FedAvg, FedPer, ajuste fino, solo local, centralizado | Resultados principales |
| `curated_r01`, `curated_r20` | Conjunto curado | 1 y 20 rondas | los mismos | Verificación |
| `ped_r20_future_rain` | PED | 20 rondas | FedAvg con la lluvia futura | Techo con lluvia futura |
| `budget_no_rain`, `budget_future_rain` | PED | 100 rondas de 10 épocas | FedAvg (LSTM, 2 semillas; 3 épocas y tasa 10⁻³ por ronda) | Prueba de presupuesto |

Cada combinación de arquitectura, método y semilla se ejecuta en un proceso; `--workers` define cuántos en
paralelo (por defecto, el número de núcleos menos dos). Las ejecuciones ya concluidas se conservan; para rehacer una, borre su archivo en
`results/<configuración>/runs/`. Las opciones `--architectures`, `--methods` y `--seeds` restringen la grilla.

**Verificación rápida.** `--max-rounds N` interrumpe la simulación en la ronda N y escribe en
`results/<configuración>_maxN/`. Como la simulación es secuencial, las predicciones hasta la ronda N son las mismas de
la ejecución completa. Por ejemplo, para comprobar en pocos minutos una combinación del conjunto curado:

```bash
python scripts/run_experiment.py --config configs/curated_r01.json --architectures lstm --seeds 42 --max-rounds 12
python scripts/analyze.py --check
```

La referencia incluye las métricas de esta verificación (configuración `curated_r01_max12`, cinco métodos).

### 4. Análisis y comparación con la referencia

```bash
python scripts/analyze.py --check
```

El script genera las tablas en `results/analysis/` con las predicciones disponibles y las compara con
`results/reference/`. La tabla `station_metrics.csv` contiene el MAE de cada ejecución y estación y permite comprobar
ejecuciones aisladas de cualquier configuración; de las interrumpidas con `--max-rounds`, la referencia solo tiene
la verificación rápida anterior. Las demás tablas requieren todas las ejecuciones de las configuraciones
correspondientes.

### Determinismo

Las semillas fijan la inicialización de los pesos, el orden de los lotes y el *dropout*, y el entrenamiento se
ejecuta en CPU con `torch.use_deterministic_algorithms(True)` y un hilo por proceso. Con las versiones fijadas, el
código de este repositorio reprodujo bit a bit, en el mismo computador, los modelos iniciales y las predicciones de
las ejecuciones originales (comprobados los cinco métodos, las dos arquitecturas, los dos modelos iniciales y los
dos conjuntos). En otro hardware, sistema operativo o versión de PyTorch, el redondeo en punto flotante puede alterar
los últimos dígitos.

## Resultados

Razón del MAE sobre el de la persistencia (promedio de cinco semillas; valores por debajo de 1 indican un error
menor que repetir la humedad actual), con el modelo inicial de 20 rondas:

| Conjunto | Modelo | FedAvg | FedPer | Ajuste fino | Solo local | Centralizado |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PED, 2016–2026 | LSTM | 0,989 | 0,991 | 0,988 | 1,007 | 0,995 |
| PED, 2016–2026 | TCN | 0,983 | 0,986 | 0,982 | 1,016 | 0,975 |
| Conjunto curado, 2016–2019 | LSTM | 0,958 | 0,964 | 0,957 | 0,989 | 0,945 |
| Conjunto curado, 2016–2019 | TCN | 0,941 | 0,948 | 0,939 | 0,985 | 0,952 |

- **Pregunta 1.** FedPer no superó a FedAvg en ninguno de los cuatro escenarios de la PED (2 modelos × 2 modelos
  iniciales) y fue mejor en el 40% al 43% de las estaciones; la diferencia favorece a FedAvg y es significativa tras
  la corrección de Holm en dos escenarios.
- **Pregunta 2.** La correlación entre la heterogeneidad y la ventaja de FedPer es positiva y débil
  (ρ = 0,105 en la LSTM, IC de −0,002 a 0,216; ρ = 0,121 en la TCN) y, en la LSTM, no se distingue de la variación
  entre semillas.

Las tablas completas están en [`results/reference/`](results/reference/):

| Archivo | Contenido |
| --- | --- |
| `performance.csv` | Razón por método, IC, desviación entre semillas, fracción de estaciones mejores que la persistencia |
| `method_pairs.csv` | FedPer × FedAvg, ajuste fino × FedAvg y × FedPer, solo local y centralizado × FedAvg |
| `initial_model.csv` | Efecto del modelo inicial (1 frente a 20 rondas) |
| `heterogeneity.csv`, `heterogeneity_quartiles.csv`, `heterogeneity_seed_control.csv` | Pregunta 2, cuartiles de *H* y control negativo con pares de semillas |
| `linear_reference.csv` | Regresión lineal de referencia |
| `future_rain.csv` | FedAvg sin y con la lluvia futura conocida (2016–2026) |
| `budget.csv`, `budget_ci.csv` | Prueba de presupuesto de entrenamiento |
| `station_metrics.csv` | MAE por ejecución y estación |

Las columnas conservan los nombres del análisis original, en portugués (por ejemplo, `razao` = razón,
`ic_lo`/`ic_hi` = límites del IC, `p_holm_12` = valor p ajustado por Holm en una familia de 12 pruebas); las familias
de la corrección de Holm se indican en las propias tablas.

El trabajo también presenta análisis diagnósticos que no forman parte de este repositorio: situaciones de lluvia,
semestres, *gradient boosting* y una LSTM centralizada con la lluvia futura, contexto regional y del suelo, y
sensibilidad a las lecturas de humedad iguales a cero.

## Cita

Los metadatos de cita están en [`CITATION.cff`](CITATION.cff) (GitHub ofrece la opción *Cite this repository*).
Para citar el software:

> da Silva, A. A. (2026). *soil-learning: FedAvg e FedPer na previsão da umidade do solo com estações do CEMADEN*
> (versión 1.0.0) [software]. <https://github.com/AdrianoAndreDaSilva/soil-learning>

Para citar el trabajo:

> da Silva, A. A.; de Carvalho, J. V. (2026). *Personalização no aprendizado federado para previsão da umidade do
> solo: FedAvg x FedPer na rede do CEMADEN*. Trabajo de Conclusión de Curso (Ciencia de la Computación), Universidade
> Feevale, Novo Hamburgo, Brasil.

## Licencia

El código se pone a disposición bajo la [PolyForm Noncommercial License 1.0.0](LICENSE)
(SPDX: `PolyForm-Noncommercial-1.0.0`). La licencia permite usar, estudiar, modificar y redistribuir el código con
fines no comerciales, como investigación, estudio y uso personal, y permite el uso por instituciones educativas,
organizaciones públicas de investigación e instituciones gubernamentales, siempre que las copias incluyan los
términos de la licencia y el aviso de derechos de autor (*Required Notice*). Los usos no contemplados por la
licencia, en particular los comerciales, requieren una autorización aparte del titular de los derechos. Por
restringir el uso comercial, no es una licencia de código abierto según la definición de la Open Source Initiative.

La licencia no se aplica a los datos. El conjunto curado sigue la licencia CC BY 4.0 de sus autores. Los datos
derivados de la PED se distribuyen según los Términos de Uso de la plataforma, que los ofrecen como servicio público
gratuito y piden citar la fuente, en portugués: **DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC**; Centro Nacional de Monitoramento e Alertas de Desastres
Naturais - CEMADEN/MCTI, <https://www.gov.br/cemaden/> (ver [`data/README.md`](data/README.md)). Las
bibliotecas utilizadas tienen licencias propias (BSD-3-Clause en PyTorch, NumPy, pandas y SciPy; Apache-2.0 en
pyarrow; MIT en openpyxl y pytest).

## Agradecimientos

Declaración de financiamiento, en su redacción original obligatoria:

Este trabalho foi totalmente financiado pelo projeto Venture Science, suportado pelo Center for Embedded Devices and Research in Digital Agriculture (CEDRA) - SENAI-RS, com recursos financeiros oriundos do PPI IoT/Manufatura 4.0 / PPI HardwareBR do MCTI, através do Termo de Cooperação n.° 056/2023, firmado com a EMBRAPII.
