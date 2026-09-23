"""
Entrena los modelos baseline para predicción de duración de viaje
y guarda las métricas en outputs/metrics.json.

Uso:
    python src/train.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_base_features, add_target_and_time_features, clean_for_modeling
from features import BASE_FEATURES, TARGET

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "yellow_2026_jan_apr.parquet"
OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "metrics.json"


def main():
    df = load_base_features(str(DATA_PATH))
    df = add_target_and_time_features(df)
    df = clean_for_modeling(df)

    # Partición TEMPORAL (no aleatoria): entrenar con ene-mar, evaluar con abril.
    # Justificación: en producción el modelo siempre predice viajes futuros
    # respecto a los datos con los que fue entrenado. Un split aleatorio
    # filtraría información temporal (leakage) y sobreestimaría el desempeño real.
    train = df[df["source_month"] != "2026-04"]
    test = df[df["source_month"] == "2026-04"]

    X_train, y_train = train[BASE_FEATURES], train[TARGET]
    X_test, y_test = test[BASE_FEATURES], test[TARGET]

    # Baseline 1: mediana global (predicción constante)
    median_pred = np.full(len(y_test), y_train.median())
    mae_median = mean_absolute_error(y_test, median_pred)
    rmse_median = np.sqrt(mean_squared_error(y_test, median_pred))

    # Baseline 2: regresión lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, pred)
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred))
    r2_lr = r2_score(y_test, pred)

    results = {
        "baseline_mediana": {"MAE_min": round(mae_median, 3), "RMSE_min": round(rmse_median, 3)},
        "baseline_regresion_lineal": {
            "MAE_min": round(mae_lr, 3),
            "RMSE_min": round(rmse_lr, 3),
            "R2": round(r2_lr, 4),
        },
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features_usadas": BASE_FEATURES,
        "split": "temporal: train=ene-mar 2026, test=abril 2026",
        "seed": SEED,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
