"""
Feature engineering y preprocesamiento (scikit-learn Pipelines).

CONTROL DE LEAKAGE
------------------
Solo se usan variables conocidas al momento de iniciar el viaje:
pickup (fecha/hora), zona de recojo, zona de destino declarada,
distancia, vendor, tarifa (RatecodeID) y nº de pasajeros.

NO se usan: tpep_dropoff_datetime (solo para el target), fare_amount,
tip_amount, total_amount, tolls_amount, payment_type, extra, mta_tax,
improvement_surcharge, congestion_surcharge, Airport_fee, cbd_congestion_fee.

Las codificaciones que miran el target (TargetEncoder para zonas y rutas)
se ajustan DENTRO del Pipeline, por lo que solo ven datos de entrenamiento.
Además TargetEncoder usa cross-fitting interno (cv=5) al transformar el
propio train, así el modelo no ve la media de su propia fila.

Limitación conocida: `trip_distance` es la distancia real medida al final
del viaje (ver proposal.md, sección 8). Se mantiene como proxy de una
distancia de ruta estimada; ver experimento de sensibilidad en train.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

TARGET = "duration_min"

# Zonas TLC de aeropuertos: 1 = Newark (EWR), 132 = JFK, 138 = LaGuardia
AIRPORT_ZONES = (1, 132, 138)
# Zonas "desconocidas" en el lookup oficial del TLC
UNKNOWN_ZONES = (264, 265)

# Features usadas en la entrega previa (se mantienen para el baseline)
BASE_FEATURES = ["trip_distance", "pickup_hour", "pickup_dow"]

NUMERIC_FEATURES = [
    "trip_distance",
    "log_distance",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_holiday",
    "same_zone",
    "pu_airport",
    "do_airport",
    "unknown_zone",
    "passenger_count",
    "ratecode_missing",
]
ONEHOT_FEATURES = ["pickup_hour", "pickup_dow", "RatecodeID_cat", "VendorID"]
TARGET_ENC_FEATURES = ["PULocationID", "DOLocationID", "route"]

ALL_FEATURES = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENC_FEATURES

# Sin distancia (experimento de sensibilidad al leakage de trip_distance)
NO_DISTANCE_NUMERIC = [c for c in NUMERIC_FEATURES if c not in ("trip_distance", "log_distance")]


def _holidays(years) -> set:
    cal = USFederalHolidayCalendar()
    hol = cal.holidays(start=f"{min(years)}-01-01", end=f"{max(years)}-12-31")
    return set(hol.date)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features derivadas. Todas son funciones deterministas de
    columnas disponibles al pickup (no dependen del target ni de otras filas),
    así que pueden calcularse antes del split sin riesgo de leakage.
    """
    df = df.copy()
    ts = df["tpep_pickup_datetime"]
    if "pickup_hour" not in df:
        df["pickup_hour"] = ts.dt.hour.astype("int8")
        df["pickup_dow"] = ts.dt.dayofweek.astype("int8")

    df["log_distance"] = np.log1p(df["trip_distance"]).astype("float32")
    minutes = ts.dt.hour * 60 + ts.dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * minutes / 1440).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * minutes / 1440).astype("float32")
    df["is_weekend"] = (df["pickup_dow"] >= 5).astype("int8")
    hol = _holidays(ts.dt.year.dropna().unique())
    df["is_holiday"] = ts.dt.date.isin(hol).astype("int8")

    df["same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype("int8")
    df["pu_airport"] = df["PULocationID"].isin(AIRPORT_ZONES).astype("int8")
    df["do_airport"] = df["DOLocationID"].isin(AIRPORT_ZONES).astype("int8")
    df["unknown_zone"] = (
        df["PULocationID"].isin(UNKNOWN_ZONES) | df["DOLocationID"].isin(UNKNOWN_ZONES)
    ).astype("int8")

    # Faltantes estructurales (~26%, concentrados en ciertos vendors):
    # se conserva la información de "faltante" como indicador explícito.
    df["ratecode_missing"] = df["RatecodeID"].isna().astype("int8")
    rc = df["RatecodeID"].fillna(-1).astype(int)
    rc = rc.where(rc.isin([1, 2, 3, 4, 5, 6, 99, -1]), 99)
    df["RatecodeID_cat"] = rc.astype(str)
    df["VendorID"] = df["VendorID"].astype(str)

    # Ruta origen-destino como categoría (alta cardinalidad: ~70k combinaciones)
    df["route"] = df["PULocationID"].astype(int) * 1000 + df["DOLocationID"].astype(int)
    return df


def _target_encoder(seed: int) -> TargetEncoder:
    """TargetEncoder con cross-fitting de 5 folds barajados (compatible sklearn 1.3 a 1.9+)."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=seed)


def build_preprocessor(kind: str = "full", scale: bool = False, seed: int = 42) -> ColumnTransformer:
    """
    kind:
      - "base": solo distancia + hora + día (one-hot), como baseline lineal.
      - "full": todas las features.
      - "no_distance": todo menos trip_distance/log_distance (sensibilidad).
    scale: estandariza las numéricas (necesario para modelos lineales regularizados).
    """
    num_scaler = [("scale", StandardScaler())] if scale else []

    if kind == "base":
        return ColumnTransformer([
            ("num", Pipeline(num_scaler or [("pass", "passthrough")]), ["trip_distance"]),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["pickup_hour", "pickup_dow"]),
        ])

    numeric = NUMERIC_FEATURES if kind == "full" else NO_DISTANCE_NUMERIC
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))] + num_scaler), numeric),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=100),
         ONEHOT_FEATURES),
        ("te", Pipeline([
            ("te", _target_encoder(seed)),
        ] + num_scaler), TARGET_ENC_FEATURES),
    ], verbose_feature_names_out=False)


def build_feature_matrix(df: pd.DataFrame, features: list | None = None) -> pd.DataFrame:
    """Compatibilidad con la entrega previa."""
    return df[features or BASE_FEATURES].copy()
