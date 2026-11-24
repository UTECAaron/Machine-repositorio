# Avance 1 — Predicción de duración de viajes en NYC Yellow Taxi (ene-abr 2026)

**Integrantes:** Romano Castro, Aaron Adriano · Chahuara Galdos, Sebastian Ernesto · Calderon Soto, Enzo Mathias · Villanueva Lara, Carlos Armando

## 0. Qué cubre este avance

Respecto a la entrega previa (propuesta + EDA + baseline), este avance adelanta estos puntos del informe final:

| Punto del informe final | Estado | Dónde está |
|---|---|---|
| 4. Limpieza y preparación de datos | ✅ pipeline reproducible con reporte por regla | `src/data.py`, `notebooks/02_limpieza_features.ipynb` |
| 6. Estrategia de partición y validación | ✅ train / validación / test temporales | `src/train.py`, sección 3 |
| 7. Baselines | ✅ rehechos y corregidos | sección 4 |
| 8. Modelos entrenados | ✅ 3 familias (lineal regularizado, Random Forest, Gradient Boosting) | `notebooks/03_modelos.ipynb` |
| 11. Interpretación | 🟡 importancia por permutación | sección 6 |
| 12. Análisis de errores | 🟡 primer corte por segmentos | sección 7 |
| 9. Búsqueda de hiperparámetros | ⬜ siguiente avance | sección 9 |
| 13. Riesgos éticos y sesgos | 🟡 avances puntuales (leakage de distancia, sesgo por vendor) | secciones 5 y 8 |

## 1. Problema (sin cambios)

Predecir la duración en minutos de un viaje de taxi amarillo en NYC en el momento del pickup, con la zona
de origen, zona de destino declarada, hora, día y distancia. Métrica principal **MAE**; secundarias **RMSE** y **R²**.
En este avance agregamos MedAE, el percentil 90 del error absoluto y MAPE para ver la cola y el error relativo.

## 2. Limpieza y preparación

### 2.1 Reglas de limpieza

Se aplican de forma secuencial y se registra cuántas filas elimina cada una (`outputs/limpieza_reporte.csv`).

| Regla | Motivo | % eliminado (train) |
|---|---|---|
| pickup fuera del mes del archivo | fechas de otros años por relojes mal configurados; además romperían el split temporal | [completar] |
| duración < 1 min (incl. negativas) | errores de registro, cancelaciones | [completar] |
| duración > 180 min | dropoffs no cerrados | [completar] |
| distancia ≤ 0 | viaje sin sentido físico | [completar] |
| distancia > 100 mi | error del taxímetro (máx. observado 328,522 mi) | [completar] |
| velocidad media > 80 mph | combinación distancia/duración imposible | [completar] |
| **Total retenido** | | [completar] % |

Cambios respecto a la entrega previa: el umbral inferior de duración pasa de `> 0` a `≥ 1 min`, y se agregan dos
reglas (mes del pickup y velocidad). Las reglas usan la duración (el target) para detectar etiquetas corruptas, así que se
aplican igual en train, validación y test. **Limitación:** las métricas describen el desempeño sobre registros válidos.

### 2.2 Faltantes

`RatecodeID` y `passenger_count` faltan juntos (~26%) y se concentran en ciertos `VendorID`: no es faltante al azar.
No se eliminan esas filas. `RatecodeID` faltante pasa a ser una categoría propia más un indicador `ratecode_missing`, y
`passenger_count` se imputa con la mediana de train **dentro del Pipeline**.

### 2.3 Variables

Todas se conocen al momento del pickup:

- **Numéricas:** `trip_distance`, `log_distance`, hora cíclica (`hour_sin`, `hour_cos`), `is_weekend`, `is_holiday` (feriados federales),
  `same_zone`, `pu_airport`, `do_airport` (zonas 1, 132, 138), `unknown_zone` (264, 265), `passenger_count`, `ratecode_missing`.
- **One-hot:** hora, día de la semana, `RatecodeID`, `VendorID`.
- **Target encoding:** `PULocationID`, `DOLocationID` y `route` (par origen-destino, ≈70k combinaciones).

**Corrección del baseline de la entrega previa.** `BASE_FEATURES` incluía `PULocationID` y `DOLocationID` como números
en la regresión lineal (aunque el README decía que solo usaba distancia, hora y día). Tratar un ID de zona como número no
tiene sentido: la zona 200 no es "el doble" que la 100. El nuevo `baseline_lineal` usa distancia + hora y día en one-hot,
y las zonas se usan solo en los modelos completos, con target encoding.

## 3. Partición y validación

| Conjunto | Meses | Uso |
|---|---|---|
| Train | enero, febrero 2026 | ajuste |
| Validación | marzo 2026 | comparar modelos y tomar decisiones |
| Test | abril 2026 | evaluación **única** del modelo elegido, reentrenado con ene-mar |

En la entrega previa se evaluaba directamente en abril. Si eligiéramos el modelo mirando abril, ese número ya no sería
una estimación honesta (el enunciado lo prohíbe explícitamente). Por eso ahora marzo cumple el rol de validación y abril
queda reservado.

**Muestreo por memoria (~4 GB RAM):** 700k viajes por mes de train y 1M de marzo, muestreo aleatorio simple con semilla fija.
El test se evalúa sobre **abril completo**, prediciendo por bloques de 500k filas. Todo es configurable por línea de comandos.

**Control de leakage en el target encoding:** el encoder está dentro del `Pipeline` (solo ve train) y usa cross-fitting de
5 folds al transformar el propio train, así ninguna fila recibe la media calculada con su propio target.

## 4. Resultados en validación (marzo)

| Modelo | Familia | MAE | RMSE | R² | P90 error | Tiempo de ajuste |
|---|---|---|---|---|---|---|
| baseline_mediana | constante | [completar] | | | | |
| baseline_lineal | lineal | [completar] | | | | |
| ridge | lineal regularizado | [completar] | | | | |
| random_forest | bagging de árboles | [completar] | | | | |
| hgb_mae | boosting, pérdida absoluta | [completar] | | | | |
| hgb_mse | boosting, pérdida cuadrática | [completar] | | | | |

**Modelo elegido (menor MAE en validación):** [completar]. Mejora de MAE frente al baseline lineal: [completar] %.

Figura: `reports/figures/07_comparacion_modelos_validacion.png`.

**Lectura [completar con los números reales]:**
- ¿Cuánto aporta pasar de lineal a árboles? La relación distancia-duración no es lineal (la velocidad media sube con la
  distancia, `05_velocidad_por_distancia.png`), y las interacciones zona × hora no las captura un modelo aditivo.
- `hgb_mae` vs `hgb_mse`: se espera que el primero gane en MAE y el segundo en RMSE. Elegimos por MAE porque es la métrica principal.

## 5. Sensibilidad a `trip_distance`

`trip_distance` es la distancia **real** recorrida, conocida al terminar el viaje. Para medir cuánto depende el modelo de
ella, se entrena el mismo HGB sin `trip_distance` ni `log_distance`:

| Variante | MAE | R² |
|---|---|---|
| Mejor modelo, con distancia | [completar] | [completar] |
| HGB sin distancia (zonas + hora + resto) | [completar] | [completar] |

**Lectura:** la diferencia es la parte del desempeño que depende de conocer la distancia. Para una versión realista, el
siguiente avance reemplazará la distancia real por una **distancia estimada antes del viaje**, por ejemplo la mediana
histórica de `trip_distance` por ruta calculada solo con train.

## 6. Interpretación inicial

Importancia por permutación sobre variables originales, en 30k viajes de validación (`12_importancia_permutacion.png`).
Se lee como "cuántos minutos sube el MAE si desordenamos esta variable".

Top 5: [completar desde `outputs/importancia_variables.csv`].

## 7. Primer análisis de errores (validación)

Tabla completa en `outputs/errores_por_segmento_validacion.csv`; figuras 09, 10 y 11.

| Segmento | Observación |
|---|---|
| Distancia | [completar: ¿el MAE crece con la distancia? ¿y en relativo?] |
| Duración real | [completar: se espera subestimación de viajes largos (sesgo negativo)] |
| Franja horaria | [completar: ¿más error en horas pico?] |
| Aeropuerto | [completar] |
| VendorID / RatecodeID faltante | [completar: posible sesgo por proveedor] |

## 8. Evaluación en test (abril, una sola vez)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| baseline_mediana | [completar] | | |
| baseline_lineal | [completar] | | |
| modelo elegido | [completar] | | |

Si el MAE en test es parecido al de validación, la elección es estable frente al cambio de mes. Si es bastante peor,
hay deriva temporal (ver `06_deriva_target_por_mes.png` y `outputs/deriva_mensual.csv`).

## 9. Riesgos y limitaciones identificados en este avance

- **Distancia real como feature** (sección 5): es la principal amenaza a la validez del problema tal como está planteado.
- **Destino declarado:** se asume conocido (como en apps de e-hailing), no aplica a taxis tomados en la calle.
- **Faltantes por proveedor:** si un vendor tiene más error, el modelo no es igual de útil para todos los conductores.
- **Etiquetas corruptas fuera de la evaluación:** las métricas no cubren los ~4-5% de registros descartados.
- **Muestreo:** se entrena con una muestra (memoria). En el siguiente avance comprobaremos con una curva de aprendizaje si más datos mejoran el resultado.

## 10. Próximos pasos (siguiente avance)

1. **Búsqueda de hiperparámetros** del modelo elegido con validación temporal (ene→feb, ene-feb→mar), sin tocar abril.
2. **Distancia estimada pre-viaje** por ruta, para cerrar el riesgo de leakage.
3. **Análisis de errores completo** (`notebooks/04_analisis_errores.ipynb`): zonas con más error, casos extremos, sesgo por vendor.
4. **Curva de aprendizaje** (tamaño de muestra vs. MAE).
5. **Informe final**, sección ética y presentación.

## Reproducción

```bash
pip install -r requirements.txt
python src/train.py --etapa todo        # ~10-30 min en una laptop, según la RAM
```

Uso de IA: el código de este avance y la estructura del documento se desarrollaron con asistencia de Claude (Anthropic).
El equipo revisó las decisiones e interpretará los resultados.
