"""
Carga, combinación y limpieza de los datos de NYC Yellow Taxi (ene-abr 2026).

Cambios respecto a la entrega previa
------------------------------------
- `load_months`: carga mes por mes con filtro de pyarrow y muestrea en el
  momento, para no tener los ~15M de filas en memoria a la vez.
- `clean_for_modeling` ahora devuelve también un reporte de cuántas filas
  elimina cada regla (trazabilidad de la limpieza para el informe).
- Nuevas reglas: pickup fuera del mes del archivo y velocidad media imposible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "yellow_2026_jan_apr.parquet"

# Columnas necesarias. Las de pago NO se cargan: no se usan como features
# (leakage) y así se ahorra memoria.
COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "RatecodeID",
    "VendorID",
    "source_month",
]

# Columnas que usa el notebook 01 (EDA de la entrega previa)
COLUMNS_EDA = COLUMNS[:-1] + ["payment_type", "store_and_fwd_flag", "source_month"]

# Umbrales de limpieza (justificados en reports/avance_1.md, sección 2)
MIN_DURATION_MIN = 1.0      # < 1 min con distancia > 0: error de registro o viaje cancelado
MAX_DURATION_MIN = 180.0    # > 3 h: 0.04% de los datos, dropoffs mal cerrados
MAX_DISTANCE_MI = 100.0     # máximo observado 328,522 mi: error evidente del taxímetro
MAX_SPEED_MPH = 80.0        # velocidad media físicamente implausible en NYC


# ----------------------------------------------------------------------------
# Combinación de archivos mensuales (sin cambios respecto a la entrega previa)
# ----------------------------------------------------------------------------
def combine_monthly_parquets(files: dict, out_path: str) -> None:
    """
    Combina varios parquet mensuales en un solo archivo, agregando una
    columna 'source_month'. Usa pyarrow (no pandas) para evitar cargar
    todos los meses en memoria a la vez.

    Parameters
    ----------
    files : dict
        {nombre_mes: ruta_archivo}, ej. {'2026-01': 'path/to/file.parquet'}
    out_path : str
        Ruta de salida para el parquet combinado.
    """
    writer = None
    schema = None
    for name, f in files.items():
        table = pq.read_table(f)
        table = table.append_column("source_month", pa.array([name] * table.num_rows))
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out_path, schema)
        else:
            # Los meses pueden diferir levemente en tipos (int vs double); se
            # castea al esquema del primer mes para que el archivo sea válido.
            table = table.select(schema.names).cast(schema)
        writer.write_table(table)
        del table
    if writer is not None:
        writer.close()


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("trip_distance", "passenger_count", "RatecodeID"):
        if c in df.columns:
            df[c] = df[c].astype("float32")
    for c in ("PULocationID", "DOLocationID", "VendorID"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    if "source_month" in df.columns:
        df["source_month"] = df["source_month"].astype("category")
    return df


def load_months(
    path: str | Path = DEFAULT_DATA_PATH,
    months: list[str] | None = None,
    n_per_month: int | None = None,
    seed: int = 42,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carga uno o varios meses del parquet combinado.

    Si `n_per_month` está definido, toma una muestra aleatoria simple de ese
    tamaño en cada mes (estratificada por mes, misma semilla -> reproducible).
    Se lee un mes a la vez para que el pico de memoria sea ~1 mes, no 4.
    """
    cols = columns or COLUMNS
    if months is None:
        months = sorted(
            pq.read_table(path, columns=["source_month"]).column(0).unique().to_pylist()
        )
    parts = []
    for i, m in enumerate(months):
        table = pq.read_table(path, columns=cols, filters=[("source_month", "=", m)])
        df = table.to_pandas()
        del table
        if n_per_month is not None and len(df) > n_per_month:
            df = df.sample(n=n_per_month, random_state=seed + i)
        parts.append(_optimize_dtypes(df))
    out = pd.concat(parts, ignore_index=True)
    out["source_month"] = out["source_month"].astype(str)
    return out


def load_base_features(path: str, columns: list | None = None, optimize_dtypes: bool = True) -> pd.DataFrame:
    """Compatibilidad con la entrega previa (notebook 01): carga todo el archivo."""
    df = pd.read_parquet(path, columns=columns or COLUMNS_EDA)
    return _optimize_dtypes(df) if optimize_dtypes else df


# ----------------------------------------------------------------------------
# Target y limpieza
# ----------------------------------------------------------------------------
def add_target_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la variable objetivo (duración en minutos) y la hora/día del pickup.
    El dropoff se usa ÚNICAMENTE para construir el target.
    """
    df = df.copy()
    df["duration_min"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    ).astype("float32")
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("int8")
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek.astype("int8")
    return df


def clean_for_modeling(
    df: pd.DataFrame,
    min_duration_min: float = MIN_DURATION_MIN,
    max_duration_min: float = MAX_DURATION_MIN,
    max_distance_mi: float = MAX_DISTANCE_MI,
    max_speed_mph: float = MAX_SPEED_MPH,
    return_report: bool = False,
):
    """
    Excluye registros con errores evidentes. Cada regla se aplica de forma
    secuencial y se registra cuántas filas elimina, para documentarlo.

    Nota: las reglas usan el target (duración) para detectar etiquetas
    corruptas. Se aplican igual a train, validación y test porque en esos
    registros la etiqueta misma no es confiable; esto se reporta como
    limitación (el modelo no está evaluado sobre registros corruptos).
    """
    n0 = len(df)
    speed = df["trip_distance"] / (df["duration_min"].clip(lower=1e-3) / 60)
    pickup_month = df["tpep_pickup_datetime"].dt.strftime("%Y-%m")

    rules = {
        "pickup fuera del mes del archivo": pickup_month != df["source_month"],
        f"duración < {min_duration_min:g} min (incl. negativas)": df["duration_min"] < min_duration_min,
        f"duración > {max_duration_min:g} min": df["duration_min"] > max_duration_min,
        "distancia <= 0 o faltante": ~(df["trip_distance"] > 0),
        f"distancia > {max_distance_mi:g} mi": df["trip_distance"] > max_distance_mi,
        f"velocidad media > {max_speed_mph:g} mph": speed > max_speed_mph,
    }

    keep = pd.Series(True, index=df.index)
    report = []
    for name, bad in rules.items():
        removed = int((keep & bad).sum())
        report.append({
            "regla": name,
            "filas_eliminadas": removed,
            "pct_del_total": round(100 * removed / max(n0, 1), 3),
            "pct_marcadas_regla": round(100 * float(bad.mean()), 3),
        })
        keep &= ~bad

    out = df.loc[keep].copy()
    report.append({
        "regla": "TOTAL (eliminadas | pct = % retenido)",
        "filas_eliminadas": n0 - len(out),
        "pct_del_total": round(100 * len(out) / max(n0, 1), 3),
        "pct_marcadas_regla": np.nan,
    })
    if return_report:
        return out, pd.DataFrame(report)
    return out


def load_clean(
    months: list[str],
    n_per_month: int | None = None,
    path: str | Path = DEFAULT_DATA_PATH,
    seed: int = 42,
    return_report: bool = False,
):
    """Atajo: carga -> target -> limpieza."""
    df = load_months(path, months=months, n_per_month=n_per_month, seed=seed)
    df = add_target_and_time_features(df)
    return clean_for_modeling(df, return_report=return_report)
