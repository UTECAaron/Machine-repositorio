"""
Funciones para cargar y combinar los datos de NYC Yellow Taxi.
"""
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path


def combine_monthly_parquets(files: dict, out_path: str) -> None:
    """
    Combina varios parquet mensuales en un solo archivo, agregando una
    columna 'source_month'. Usa pyarrow (no pandas) para evitar cargar
    todos los meses en memoria a la vez.

    Parameters
    ----------
    files : dict
        Diccionario {nombre_mes: ruta_archivo}, ej. {'2026-01': 'path/to/file.parquet'}
    out_path : str
        Ruta de salida para el parquet combinado.
    """
    writer = None
    for name, f in files.items():
        table = pq.read_table(f)
        month_col = pa.array([name] * table.num_rows)
        table = table.append_column("source_month", month_col)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        del table
    if writer is not None:
        writer.close()


def load_base_features(path: str, columns: list | None = None, optimize_dtypes: bool = True) -> pd.DataFrame:
    """
    Carga el dataset combinado, opcionalmente solo un subconjunto de columnas
    (recomendado dado el volumen de datos y memoria limitada).

    Con optimize_dtypes=True (default) se reducen los tipos numéricos a
    float32/int32 para bajar el uso de memoria, importante con ~15M filas
    en un entorno con RAM limitada.
    """
    cols = columns or [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "RatecodeID",
        "VendorID",
        "payment_type",
        "store_and_fwd_flag",
        "source_month",
    ]
    df = pd.read_parquet(path, columns=cols)

    if optimize_dtypes:
        if "trip_distance" in df.columns:
            df["trip_distance"] = df["trip_distance"].astype("float32")
        if "passenger_count" in df.columns:
            df["passenger_count"] = df["passenger_count"].astype("float32")
        if "RatecodeID" in df.columns:
            df["RatecodeID"] = df["RatecodeID"].astype("float32")
        for c in ("PULocationID", "DOLocationID", "VendorID", "payment_type"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], downcast="integer")

    return df


def add_target_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la variable objetivo (duración en minutos) y features temporales
    derivadas exclusivamente del pickup (disponibles antes de la predicción).
    No usa ninguna columna derivada del dropoff salvo para calcular el target.
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
    max_duration_min: float = 180,
    max_distance_mi: float = 100,
) -> pd.DataFrame:
    """
    Aplica límites razonables y documentados para excluir registros con
    errores evidentes (duración negativa/cero, duración extrema, distancia
    cero o absurda). No es un filtro arbitrario: cada límite está justificado
    en proposal.md / informe final.
    """
    mask = (
        (df["duration_min"] > 0)
        & (df["duration_min"] <= max_duration_min)
        & (df["trip_distance"] > 0)
        & (df["trip_distance"] <= max_distance_mi)
    )
    return df.loc[mask].copy()
