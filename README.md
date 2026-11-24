# Predicción de duración de viajes — NYC Yellow Taxi (ene-abr 2026)

Proyecto final del curso de Machine Learning. **Estado: Avance 1** (sobre la entrega previa).
Resumen del avance: [`reports/avance_1.md`](reports/avance_1.md).

## Estructura

```
├── proposal.md                          # Propuesta (entrega previa)
├── requirements.txt
├── data/README.md                       # Cómo obtener y combinar los datos
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb     # EDA + baseline (entrega previa)
│   ├── 02_limpieza_features.ipynb       # Reglas de limpieza, faltantes, features, deriva
│   └── 03_modelos.ipynb                 # Comparación de modelos, sensibilidad, errores
├── src/
│   ├── data.py                          # Carga por mes, target, limpieza con reporte
│   ├── features.py                      # Features y preprocesamiento (Pipelines, sin leakage)
│   ├── evaluate.py                      # Métricas, errores por segmento, figuras
│   └── train.py                         # Pipeline principal (validación y test)
├── tests/make_fake_data.py              # Datos SINTÉTICOS para probar que el pipeline corre
├── reports/
│   ├── avance_1.md                      # Informe de avance
│   ├── resultados_avance1.md            # Tablas generadas por train.py
│   └── figures/
└── outputs/                             # metrics*.json, CSV de errores, importancia, limpieza
```

## Cómo reproducir

```bash
pip install -r requirements.txt
```

1. **Datos:** descargar los 4 parquet del TLC y combinarlos en `data/yellow_2026_jan_apr.parquet` (ver `data/README.md`).
2. **Exploración (entrega previa):**
   `jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploracion_inicial.ipynb`
3. **Pipeline completo del avance:**
   ```bash
   python src/train.py --etapa validacion   # compara 6 modelos: train ene-feb, validación marzo
   python src/train.py --etapa test         # reentrena el elegido con ene-mar y evalúa UNA vez en abril
   # o ambas: python src/train.py --etapa todo
   ```
   Genera `outputs/metrics.json`, los CSV de análisis, las figuras 06-13 y `reports/resultados_avance1.md`.
4. **Notebooks del avance:** `notebooks/02_limpieza_features.ipynb` y `notebooks/03_modelos.ipynb`.

**Memoria:** por defecto se usan 700k viajes por mes de train y 1M de marzo, y el test es abril completo, predicho por bloques.
Probado para ~4 GB de RAM. Para ajustar: `--n-por-mes`, `--n-val`, `--n-test`, `--rf-max-samples`.

**Prueba rápida sin los datos reales** (solo comprueba que el código corre, las métricas no significan nada):
```bash
python tests/make_fake_data.py --out data/fake_yellow.parquet
python src/train.py --etapa todo --rapido --data data/fake_yellow.parquet
```

## Partición y validación

| Conjunto | Meses | Uso |
|---|---|---|
| Train | enero-febrero 2026 | ajuste |
| Validación | marzo 2026 | elegir modelo |
| Test | abril 2026 | evaluación final única |

## Reproducibilidad

- `SEED = 42` en muestreo, modelos y target encoding.
- Todo el preprocesamiento vive en `sklearn.Pipeline`: lo que depende de los datos (imputación, escalado, target encoding) se ajusta solo con train.
- Los resultados en `outputs/` y `reports/` se regeneran con `src/train.py`.
