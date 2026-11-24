"""
Métricas, análisis de errores por segmento y figuras.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

DIST_BINS = [0, 1, 2, 3, 5, 10, 20, 100]
DUR_BINS = [0, 5, 10, 20, 30, 60, 180]
HOUR_BANDS = {
    "madrugada (0-5)": range(0, 6),
    "mañana pico (6-9)": range(6, 10),
    "día (10-15)": range(10, 16),
    "tarde pico (16-19)": range(16, 20),
    "noche (20-23)": range(20, 24),
}


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "MAE_min": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE_min": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MedAE_min": round(float(median_absolute_error(y_true, y_pred)), 3),
        "P90_AE_min": round(float(np.quantile(err, 0.9)), 3),
        # Error relativo: un error de 5 min no pesa igual en un viaje de 6 que en uno de 60
        "MAPE_pct": round(float(100 * np.mean(err / np.maximum(y_true, 1))), 2),
    }


def _labels(bins) -> list[str]:
    return [f"{a:g}-{b:g}" for a, b in zip(bins[:-1], bins[1:])]


def _hour_band(h: pd.Series) -> pd.Series:
    out = pd.Series(index=h.index, dtype=object)
    for name, rng in HOUR_BANDS.items():
        out[h.isin(list(rng))] = name
    return out


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    seg = pd.DataFrame(index=df.index)
    seg["distancia (mi)"] = pd.cut(df["trip_distance"], DIST_BINS, include_lowest=True,
                                   labels=_labels(DIST_BINS))
    seg["duración real (min)"] = pd.cut(df["duration_min"], DUR_BINS, include_lowest=True,
                                        labels=_labels(DUR_BINS))
    seg["franja horaria"] = pd.Categorical(_hour_band(df["pickup_hour"]), categories=list(HOUR_BANDS),
                                           ordered=True)
    seg["fin de semana"] = np.where(df["pickup_dow"] >= 5, "sí", "no")
    airport = np.where(df["PULocationID"].isin([1, 132, 138]), "desde aeropuerto",
                       np.where(df["DOLocationID"].isin([1, 132, 138]), "hacia aeropuerto", "sin aeropuerto"))
    seg["aeropuerto"] = airport
    seg["VendorID"] = df["VendorID"].astype(str)
    seg["RatecodeID faltante"] = np.where(df["RatecodeID"].isna(), "sí", "no")
    return seg


def segment_errors(df: pd.DataFrame, y_pred, baseline_pred=None) -> pd.DataFrame:
    """
    MAE, sesgo medio (pred - real) y peso de cada segmento. Si se pasa la
    predicción del baseline, también la mejora relativa por segmento.
    """
    seg = add_segments(df)
    work = pd.DataFrame({
        "y": df["duration_min"].to_numpy(),
        "abs_err": np.abs(df["duration_min"].to_numpy() - y_pred),
        "bias": y_pred - df["duration_min"].to_numpy(),
    }, index=df.index)
    if baseline_pred is not None:
        work["abs_err_base"] = np.abs(df["duration_min"].to_numpy() - baseline_pred)

    rows = []
    for col in seg.columns:
        g = work.groupby(seg[col], observed=True)
        t = pd.DataFrame({
            "n": g.size(),
            "pct_viajes": (100 * g.size() / len(work)).round(2),
            "duracion_media": g["y"].mean().round(2),
            "MAE": g["abs_err"].mean().round(3),
            "sesgo_medio": g["bias"].mean().round(3),
        })
        if baseline_pred is not None:
            t["MAE_baseline"] = g["abs_err_base"].mean().round(3)
            t["mejora_pct"] = (100 * (1 - t["MAE"] / t["MAE_baseline"])).round(1)
        t.insert(0, "segmento", col)
        t.index.name = "valor"
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: dict, path: Path, metric: str = "MAE_min"):
    names = list(results)
    vals = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.2))
    colors = ["#9ca3af" if n.startswith("baseline") else "#2563eb" for n in names]
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.replace('_min', '')} en validación (minutos, menor es mejor)")
    ax.set_title("Comparación de modelos — validación (marzo 2026)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_pred_vs_real(y_true, y_pred, path: Path, title: str, max_points: int = 200_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) > max_points:
        idx = rng.choice(len(y_true), max_points, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, bins="log", cmap="Blues", extent=(0, 90, 0, 90))
    ax.plot([0, 90], [0, 90], color="#dc2626", lw=1, label="predicción perfecta")
    ax.set_xlim(0, 90); ax.set_ylim(0, 90)
    ax.set_xlabel("Duración real (min)"); ax.set_ylabel("Duración predicha (min)")
    ax.set_title(title); ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="nº de viajes (log)")
    _save(fig, path)


def plot_segment_mae(seg_df: pd.DataFrame, segment: str, path: Path, title: str):
    t = seg_df[seg_df["segmento"] == segment]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(t))
    if "MAE_baseline" in t:
        ax.bar(x - 0.2, t["MAE_baseline"], 0.4, label="baseline lineal", color="#9ca3af")
        ax.bar(x + 0.2, t["MAE"], 0.4, label="mejor modelo", color="#2563eb")
        ax.legend()
    else:
        ax.bar(x, t["MAE"], color="#2563eb")
    ax.set_xticks(x, t["valor"], rotation=20, ha="right")
    ax.set_ylabel("MAE (min)")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(x, t["pct_viajes"], "o--", color="#f59e0b", label="% de viajes")
    ax2.set_ylabel("% de viajes", color="#b45309")
    ax.spines[["top"]].set_visible(False)
    _save(fig, path)


def plot_residuals_by_hour(df: pd.DataFrame, y_pred, path: Path):
    res = y_pred - df["duration_min"].to_numpy()
    t = pd.DataFrame({"h": df["pickup_hour"].to_numpy(), "res": res, "abs": np.abs(res)}).groupby("h")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t["abs"].mean().index, t["abs"].mean(), "o-", color="#2563eb", label="MAE")
    ax.plot(t["res"].mean().index, t["res"].mean(), "s--", color="#dc2626", label="sesgo medio (pred − real)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(24)); ax.set_xlabel("Hora de pickup"); ax.set_ylabel("minutos")
    ax.set_title("Error del mejor modelo por hora (validación)")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_importance(imp: pd.DataFrame, path: Path, title: str):
    imp = imp.sort_values("importancia_media").tail(15)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.2))
    ax.barh(imp["feature"], imp["importancia_media"], xerr=imp["desv"], color="#2563eb")
    ax.set_xlabel("Aumento del MAE al permutar la variable (min)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_monthly_target(df: pd.DataFrame, path: Path):
    """Chequeo de deriva entre meses: distribución del target por mes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, g in df.groupby("source_month"):
        ax.hist(g["duration_min"].clip(upper=90), bins=90, histtype="step", density=True, lw=1.4, label=m)
    ax.set_xlabel("Duración (min, recortada a 90 para graficar)")
    ax.set_ylabel("densidad")
    ax.set_title("Distribución del target por mes (deriva temporal)")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)
