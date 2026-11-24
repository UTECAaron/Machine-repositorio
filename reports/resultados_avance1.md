# Resultados generados automáticamente — Avance 1

_Archivo generado por `python src/train.py`. No editar a mano._

## Validación (train 2026-01, 2026-02 → validación 2026-03)
n_train = 1,329,344 · n_val = 953,449

| modelo | MAE_min | RMSE_min | R2 | MedAE_min | P90_AE_min | MAPE_pct | fit_seg |
|---|---|---|---|---|---|---|---|
| baseline_mediana | 9.208 | 14.881 | -0.0592 | 6.217 | 19.4 | 72.5 | 0.3 |
| baseline_lineal | 5.578 | 8.773 | 0.6318 | 4.029 | 10.688 | 47.88 | 1.3 |
| ridge | 4.351 | 6.989 | 0.7664 | 2.985 | 8.942 | 32.55 | 6.3 |
| random_forest | 3.496 | 5.877 | 0.8348 | 2.268 | 7.29 | 24.54 | 26.8 |
| hgb_mae | 3.288 | 5.831 | 0.8374 | 1.974 | 7.045 | 20.93 | 62.4 |
| hgb_mse | 3.397 | 5.731 | 0.8429 | 2.186 | 7.132 | 23.53 | 40.2 |

**Mejor modelo:** `hgb_mae` — mejora de MAE vs baseline lineal: **41.05%**

**Sensibilidad (HGB sin `trip_distance`):** MAE = 3.887 min, R² = 0.7891

## Limpieza (sobre las muestras cargadas)

| regla | elim_train | pct_train | elim_val | pct_val |
|---|---|---|---|---|
| pickup fuera del mes del archivo | 8 | 0.001 | 6 | 0.001 |
| duración < 1 min (incl. negativas) | 30851 | 2.204 | 22453 | 2.245 |
| duración > 180 min | 646 | 0.046 | 292 | 0.029 |
| distancia <= 0 o faltante | 38953 | 2.782 | 23655 | 2.365 |
| distancia > 100 mi | 72 | 0.005 | 20 | 0.002 |
| velocidad media > 80 mph | 126 | 0.009 | 125 | 0.013 |
| TOTAL (eliminadas | pct = % retenido) | 70656 | 94.953 | 46551 | 95.345 |

## Distribución del target por mes

| source_month | count | mean | std | min | 25% | 50% | 75% | 95% | max |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01 | 665424.0 | 17.19 | 13.81 | 1.0 | 8.37 | 13.55 | 21.4 | 43.25 | 179.97 |
| 2026-02 | 663920.0 | 17.86 | 14.06 | 1.0 | 8.72 | 14.22 | 22.42 | 44.32 | 179.82 |
| 2026-03 | 953449.0 | 17.4 | 14.46 | 1.0 | 8.27 | 13.5 | 21.47 | 45.22 | 179.7 |

## Errores por segmento (validación, mejor modelo vs baseline lineal)

| valor | segmento | n | pct_viajes | duracion_media | MAE | sesgo_medio | MAE_baseline | mejora_pct |
|---|---|---|---|---|---|---|---|---|
| 0-1 | distancia (mi) | 210739 | 22.1 | 6.8 | 1.768 | -0.191 | 5.092 | 65.3 |
| 1-2 | distancia (mi) | 290048 | 30.42 | 11.49 | 2.338 | -0.117 | 3.481 | 32.8 |
| 2-3 | distancia (mi) | 146850 | 15.4 | 16.64 | 2.949 | -0.1 | 3.93 | 25.0 |
| 3-5 | distancia (mi) | 122645 | 12.86 | 21.57 | 3.74 | -0.376 | 5.7 | 34.4 |
| 5-10 | distancia (mi) | 107744 | 11.3 | 28.59 | 5.197 | -1.124 | 8.022 | 35.2 |
| 10-20 | distancia (mi) | 67116 | 7.04 | 47.05 | 8.091 | -2.835 | 13.593 | 40.5 |
| 20-100 | distancia (mi) | 8307 | 0.87 | 59.9 | 10.71 | -4.09 | 21.977 | 51.3 |
| 0-5 | duración real (min) | 85353 | 8.95 | 3.66 | 1.375 | 1.174 | 6.399 | 78.5 |
| 5-10 | duración real (min) | 237498 | 24.91 | 7.56 | 1.695 | 0.908 | 4.073 | 58.4 |
| 10-20 | duración real (min) | 360379 | 37.8 | 14.43 | 2.48 | 0.289 | 2.904 | 14.6 |
| 20-30 | duración real (min) | 150993 | 15.84 | 24.18 | 3.936 | -1.196 | 6.326 | 37.8 |
| 30-60 | duración real (min) | 97151 | 10.19 | 40.19 | 7.343 | -3.548 | 11.455 | 35.9 |
| 60-180 | duración real (min) | 22075 | 2.32 | 78.25 | 18.724 | -16.981 | 31.254 | 40.1 |
| madrugada (0-5) | franja horaria | 89787 | 9.42 | 15.06 | 2.705 | -0.691 | 4.787 | 43.5 |
| mañana pico (6-9) | franja horaria | 130041 | 13.64 | 18.62 | 3.724 | -0.237 | 6.538 | 43.0 |
| día (10-15) | franja horaria | 286251 | 30.02 | 19.22 | 3.963 | -0.845 | 6.376 | 37.8 |
| tarde pico (16-19) | franja horaria | 233331 | 24.47 | 17.34 | 3.195 | -0.35 | 5.6 | 42.9 |
| noche (20-23) | franja horaria | 214039 | 22.45 | 15.28 | 2.465 | -0.298 | 4.235 | 41.8 |
| no | fin de semana | 678699 | 71.18 | 18.03 | 3.473 | -0.484 | 5.898 | 41.1 |
| sí | fin de semana | 274750 | 28.82 | 15.86 | 2.83 | -0.552 | 4.787 | 40.9 |
| desde aeropuerto | aeropuerto | 60884 | 6.39 | 36.87 | 5.45 | -2.465 | 10.249 | 46.8 |
| hacia aeropuerto | aeropuerto | 15136 | 1.59 | 39.82 | 6.118 | -2.713 | 11.134 | 45.1 |
| sin aeropuerto | aeropuerto | 877429 | 92.03 | 15.66 | 3.089 | -0.329 | 5.158 | 40.1 |
| 1 | VendorID | 190058 | 19.93 | 21.91 | 4.583 | -0.667 | 7.781 | 41.1 |
| 2 | VendorID | 761004 | 79.82 | 16.21 | 2.945 | -0.453 | 5.006 | 41.2 |
| 6 | VendorID | 2387 | 0.25 | 38.67 | 9.496 | -3.592 | 12.541 | 24.3 |
| no | RatecodeID faltante | 736313 | 77.23 | 17.34 | 3.36 | -0.585 | 5.859 | 42.7 |
| sí | RatecodeID faltante | 217136 | 22.77 | 17.6 | 3.041 | -0.226 | 4.624 | 34.2 |

## Importancia por permutación (top 10)

| feature | importancia_media | desv |
|---|---|---|
| trip_distance | 7.1289 | 0.0486 |
| route | 2.2449 | 0.0052 |
| hour_cos | 1.0951 | 0.0079 |
| hour_sin | 0.5495 | 0.0047 |
| RatecodeID_cat | 0.3335 | 0.006 |
| is_weekend | 0.31 | 0.0058 |
| PULocationID | 0.2933 | 0.0042 |
| DOLocationID | 0.2433 | 0.003 |
| pickup_dow | 0.1652 | 0.0011 |
| ratecode_missing | 0.036 | 0.0026 |

## Test (train ene-mar → test 2026-04) — evaluación única
n_train = 1,996,714 · n_test = 3,675,060

| modelo | MAE_min | RMSE_min | R2 | MedAE_min | P90_AE_min | MAPE_pct |
|---|---|---|---|---|---|---|
| baseline_mediana | 9.545 | 15.31 | -0.0882 | 6.317 | 20.85 | 69.76 |
| baseline_lineal | 5.858 | 9.053 | 0.6195 | 4.207 | 11.718 | 47.08 |
| hgb_mae | 3.596 | 6.18 | 0.8227 | 2.093 | 8.127 | 20.98 |

Mejora de MAE vs baseline lineal en test: **38.61%**
