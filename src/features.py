"""
Feature engineering para el modelo de predicción de duración de viaje.

IMPORTANTE (control de leakage):
Solo se incluyen aquí variables que estarían disponibles al momento de
iniciar el viaje. Explícitamente NO se usan: tpep_dropoff_datetime,
fare_amount, tip_amount, total_amount, tolls_amount, payment_type
(se conoce recién al finalizar el viaje), ni ninguna otra columna
relacionada con el cobro final.
"""
import pandas as pd

# Features seguras (conocidas antes o al momento del pickup)
BASE_FEATURES = [
    "trip_distance",   # distancia estimada de la ruta solicitada
    "pickup_hour",
    "pickup_dow",
    "PULocationID",
    "DOLocationID",
]

TARGET = "duration_min"


def build_feature_matrix(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Devuelve solo las columnas de features especificadas (default: BASE_FEATURES)."""
    cols = features or BASE_FEATURES
    return df[cols].copy()
