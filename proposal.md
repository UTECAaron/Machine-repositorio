---

editor_options: 
  markdown: 
    wrap: 72
---

# Propuesta de proyecto: Predicción de duración de viajes en NYC Yellow Taxi

## 1. Título del proyecto

Predicción de la duración de viajes en taxis amarillos de Nueva York (enero–abril 2026)

## 2. Integrantes

- Romano Castro, Aaron Adriano
- Chahuara Galdos, Sebastian Ernesto
- Calderon Soto, Enzo Mathias
- Villanueva Lara, Carlos Armando

## 3. Dataset elegido

**NYC Yellow Taxi Trip Records** — enero, febrero, marzo y abril de 2026. Fuente: NYC Taxi & Limousine Commission (TLC). URL: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>

- 4 archivos parquet, uno por mes.
- **14,908,446 viajes** en total, 20 columnas originales.
- Tamaño combinado: aproximadamente 295 MB en disco (formato parquet comprimido).

## 4. Pregunta predictiva

Dado un viaje que está por iniciar (con su ubicación de recojo, hora, día de la semana y distancia estimada de ruta), **¿cuánto tiempo durará el viaje, en minutos?**

## 5. Variable objetivo

`duration_min` = `tpep_dropoff_datetime` − `tpep_pickup_datetime`, expresada en minutos.

## 6. Unidad de predicción

Un viaje individual (una fila del dataset = un viaje = una predicción).

## 7. Variables disponibles antes de la predicción

Al momento en que el viaje inicia (pickup), se conoce:

 - `trip_distance` (distancia estimada de la ruta solicitada y ver nota de riesgo en sección 8)

\- `PULocationID` (zona de recojo)

\- `DOLocationID` (zona de destino declarada)

\- Hora del día y día de la semana derivados de `tpep_pickup_datetime` - `passenger_count`, `RatecodeID`, `VendorID`

**No se usan**: `tpep_dropoff_datetime`, `fare_amount`, `tip_amount`, `total_amount`, `tolls_amount`, `payment_type`, `mta_tax`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee` porque todas estas se conocen recién al finalizar el viaje o al momento del cobro.

## 8. Riesgos de leakage

- **`trip_distance`**: en el dataset del TLC este valor corresponde a la distancia *real* recorrida (medida por el taxímetro al finalizar), no a una distancia estimada al momento de solicitar el viaje. Esto es una forma sutil de leakage: en un caso de uso real (app de viajes estimando duración antes de aceptar), solo se tendría una distancia *estimada* por un motor de rutas, no la real. **Decisión tomada**: se usa como proxy razonable para el baseline, pero se documenta explícitamente esta limitación y se discute en el informe final como una simplificación del problema, no como una solución lista para producción.
- **Cualquier variable relacionada al pago** (`fare_amount`, `tip_amount`, `payment_type`, etc.) tiene correlación directa con la duración/distancia del viaje y solo se conoce al finalizar — quedan excluidas por completo.
- **`DOLocationID`**: se asume conocido de antemano (el pasajero indica destino al pedir el viaje), lo cual es razonable para apps de e-hailing, pero no aplica igual para taxis de calle sin destino declarado. Se documenta como supuesto.

## 9. Métrica principal y métrica secundaria

- **Principal**: MAE (Mean Absolute Error, en minutos).
- **Secundaria**: RMSE (penaliza más los errores grandes, relevante porque hay una cola larga de viajes atípicamente largos) y R².

## 10. Plan de validación

**Partición temporal, no aleatoria**: entrenamiento con enero–marzo 2026 (aproximadamente 10.57M viajes), evaluación con abril 2026 (aproximadamente 3.69M viajes).

Justificación: en producción el modelo siempre predice viajes *futuros* respecto a los datos de entrenamiento. Un split aleatorio (ej. train_test_split estándar) filtraría información temporal, el modelo "vería" patrones de abril durante el entrenamiento y sobreestimaría el desempeño real. La partición temporal simula honestamente el escenario de uso real.

## 11. Modelo baseline

Dos baselines, de menor a mayor complejidad:

1\. **Predicción por mediana global** (constante): MAE = 9.56 min, RMSE = 15.32 min.

2. **Regresión lineal** (`trip_distance`, `pickup_hour`, `pickup_dow`): MAE = 6.02 min, RMSE = 9.42 min, R² = 0.589.

La regresión lineal ya reduce el error en aproximadamente 37% respecto a la mediana, lo cual da un piso razonable y honesto para comparar modelos más complejos (árboles, gradient boosting) en las siguientes semanas.

## 12. Riesgos técnicos

- **Memoria limitada** (entorno con 4GB RAM) frente a un dataset de aproximadamente 15M filas nos hace requerir trabajar con dtypes optimizados, carga selectiva de columnas y/o procesamiento por chunks en vez de cargar todo en pandas de una vez.

- **Calidad de datos**: aproximadamente 26% de valores faltantes en `RatecodeID`, `store_and_fwd_flag` y `passenger_count` (concentrados en un subconjunto de vendors);

  duraciones negativas (\~1.24% de los registros)

  y

  outliers extremos de distancia (máximo de 328,522 millas, claramente erróneo).

- **Cambios estructurales entre meses**: posible estacionalidad o eventos puntuales (clima, feriados) que afecten la distribución de abril de forma distinta a enero–marzo, lo que podría afectar la evaluación del modelo final.

- **Balance de la cola larga**: viajes muy largos (\>3h, 0.04% de los casos) son raros pero con alto error absoluto. (Tenemos que decidir si excluirlos, capearlos o modelarlos aparte.)

## 13. Plan de trabajo semanas restantes

+----------------------------------+---------------------------------------------------------------------------------------------------------------------------+
| Semana                           | Actividad                                                                                                                 |
+==================================+===========================================================================================================================+
| 6-7                              | Limpieza definitiva, feature engineering adicional (features de zona, festivos, clima si se integra fuente externa)       |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------+
| 7-8                              | Entrenamiento de al menos 3 familias de modelos (regresión lineal regularizada, árboles/Random Forest, Gradient Boosting) |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------+
| 8-9                              | Búsqueda de hiperparámetros, validación cruzada temporal                                                                  |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------+
| 9–10                             | Análisis de errores por segmento (zona, hora, duración real)                                                              |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------+
| 10–11                            | Interpretabilidad (feature importance, SHAP si aplica)                                                                    |
|                                  |                                                                                                                           |
|                                  | Redacción de informe final, discusión de sesgos y limitaciones                                                            |
|                                  |                                                                                                                           |
|                                  | Presentación, pulido de repositorio y reproducibilidad                                                                    |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------+
