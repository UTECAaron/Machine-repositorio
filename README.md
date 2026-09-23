---

editor_options: 
  markdown: 
    wrap: 72
---

# Predicción de duración de viajes — NYC Yellow Taxi (ene-abr 2026)

Entrega previa (semana 1) del proyecto final del curso de Machine Learning.

## Estructura

```         
proyecto-final/
├── proposal.md                          # Propuesta completa (13 puntos)
├── README.md                            # Este archivo
├── requirements.txt
├── notebooks/
│   └── 01_exploracion_inicial.ipynb     # EDA + baseline (ejecutado, con outputs)
└── src/
    ├── data.py                          # Carga y combinación de datos
    ├── features.py                      # Definición de features (sin leakage)
    └── train.py                         # Entrena el baseline sobre el dataset completo
```

> El dataset (`yellow_tripdata_2026-{01,02,03,04}.parquet`, \~295MB combinado) no se incluye en este repositorio por su peso. Ver instrucciones abajo para regenerarlo a partir de los 4 archivos originales del TLC.

## Cómo reproducir

### 1. Instalar dependencias

``` bash
pip install -r requirements.txt
```

### 2. Combinar los datos

Los 4 archivos originales del TLC se combinan en uno solo con `source_month` como columna adicional:

``` python
from src.data import combine_monthly_parquets

files = {
    "2026-01": "ruta/a/yellow_tripdata_2026-01.parquet",
    "2026-02": "ruta/a/yellow_tripdata_2026-02.parquet",
    "2026-03": "ruta/a/yellow_tripdata_2026-03.parquet",
    "2026-04": "ruta/a/yellow_tripdata_2026-04.parquet",
}
combine_monthly_parquets(files, "data/yellow_2026_jan_apr.parquet")
```

### 3. Ejecutar la exploración inicial

``` bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploracion_inicial.ipynb
```

> **Nota sobre memoria**: el dataset completo tiene aproximadamente 14.9M filas. El notebook usa una muestra aleatoria representativa (15%, \~2.2M filas) para que la exploración corra cómodamente en entornos con RAM limitada (probado con \~4GB).

### 4. Ejecutar el baseline (dataset completo, no la muestra)

``` bash
python src/train.py
```

Imprime las métricas de los dos baselines (mediana global y regresión lineal), usando partición **temporal** (train: enero-marzo, test: abril) para evitar leakage temporal.

## Resultado del baseline (dataset completo)

| Modelo                                         | MAE (min) | RMSE (min) | R²     |
|------------------------------------------------|-----------|------------|--------|
| Mediana global                                 | 9.59      | 15.35      | —      |
| Regresión lineal (distancia, hora, día semana) | 6.02      | 9.41       | 0.5911 |

Ver `proposal.md` para la justificación de la métrica, el split y los riesgos de leakage identificados.

## Reproducibilidad

- Semilla fija: `SEED = 42` en todos los scripts.
- Split temporal (no aleatoria) para simular el escenario de producción real.
