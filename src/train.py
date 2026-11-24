"""
Pipeline principal de entrenamiento y evaluación — Avance 1.

Esquema de partición (temporal, sin mezclar meses):

    enero + febrero  ->  TRAIN        (ajuste de modelos)
    marzo            ->  VALIDACIÓN   (comparar modelos / tomar decisiones)
    abril            ->  TEST         (se usa una sola vez, al final)

Uso:
    python src/train.py --etapa validacion   # compara modelos en marzo
    python src/train.py --etapa test         # reentrena en ene-mar con el modelo elegido y evalúa en abril
    python src/train.py --etapa todo         # ambas, en orden

    # prueba rápida con poca data (para revisar que todo corre):
    python src/train.py --etapa todo --rapido

Salidas:
    outputs/metrics_validacion.json, outputs/metrics_test.json, outputs/metrics.json
    outputs/errores_por_segmento_validacion.csv, outputs/limpieza_reporte.csv,
    outputs/importancia_variables.csv, outputs/deriva_mensual.csv
    reports/figures/*.png
    reports/resultados_avance1.md   (tablas listas para el informe)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

import data as D
import evaluate as E
from features import ALL_FEATURES, TARGET, add_features, build_preprocessor

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"

TRAIN_MONTHS = ["2026-01", "2026-02"]
VAL_MONTH = "2026-03"
TEST_MONTH = "2026-04"


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def make_models(n_train: int, rf_max_samples: int) -> dict:
    """
    Tres familias + dos baselines. Hiperparámetros razonables por defecto;
    la búsqueda sistemática queda para el siguiente avance.
    """
    rf_frac = min(1.0, rf_max_samples / max(n_train, 1))
    return {
        # --- baselines (los mismos de la entrega previa, versión honesta) ---
        "baseline_mediana": Pipeline([
            ("prep", build_preprocessor("base")),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "baseline_lineal": Pipeline([
            ("prep", build_preprocessor("base")),
            ("model", LinearRegression()),
        ]),
        # --- familia 1: lineal regularizado con todas las features ---
        "ridge": Pipeline([
            ("prep", build_preprocessor("full", scale=True, seed=SEED)),
            ("model", Ridge(alpha=1.0, random_state=SEED)),
        ]),
        # --- familia 2: bagging de árboles ---
        "random_forest": Pipeline([
            ("prep", build_preprocessor("full", seed=SEED)),
            ("model", RandomForestRegressor(
                n_estimators=80, max_depth=20, min_samples_leaf=20, max_features=0.5,
                max_samples=rf_frac, n_jobs=-1, random_state=SEED,
            )),
        ]),
        # --- familia 3: gradient boosting (dos funciones de pérdida) ---
        "hgb_mae": Pipeline([
            ("prep", build_preprocessor("full", seed=SEED)),
            ("model", HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.1, max_iter=400, max_leaf_nodes=63,
                min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20, random_state=SEED,
            )),
        ]),
        "hgb_mse": Pipeline([
            ("prep", build_preprocessor("full", seed=SEED)),
            ("model", HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.1, max_iter=400, max_leaf_nodes=63,
                min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20, random_state=SEED,
            )),
        ]),
    }


def sensitivity_model() -> Pipeline:
    """HGB sin trip_distance: cuánto depende el modelo de la variable con leakage parcial."""
    return Pipeline([
        ("prep", build_preprocessor("no_distance", seed=SEED)),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.1, max_iter=400, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=SEED,
        )),
    ])


def predict_in_chunks(model, df: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
    """Predice por bloques para no duplicar en memoria un mes completo."""
    preds = []
    for start in range(0, len(df), chunk):
        part = add_features(df.iloc[start:start + chunk])
        preds.append(model.predict(part[ALL_FEATURES]))
    return np.clip(np.concatenate(preds), 0, None)


# ----------------------------------------------------------------------------
# Etapa 1: validación (marzo)
# ----------------------------------------------------------------------------
def etapa_validacion(args) -> dict:
    print("== ETAPA VALIDACIÓN: train = ene-feb, validación = marzo ==")
    t0 = time.time()
    train, rep_train = D.load_clean(TRAIN_MONTHS, n_per_month=args.n_por_mes, path=args.data,
                                    seed=SEED, return_report=True)
    val, rep_val = D.load_clean([VAL_MONTH], n_per_month=args.n_val, path=args.data,
                                seed=SEED + 100, return_report=True)
    print(f"train: {len(train):,} filas | val: {len(val):,} filas | carga {time.time()-t0:.0f}s")

    # Reporte de limpieza (sobre las muestras cargadas)
    rep = rep_train.rename(columns={"filas_eliminadas": "elim_train", "pct_del_total": "pct_train"})
    rep = rep.drop(columns="pct_marcadas_regla").merge(
        rep_val.rename(columns={"filas_eliminadas": "elim_val", "pct_del_total": "pct_val"})
        .drop(columns="pct_marcadas_regla"), on="regla")
    rep.to_csv(OUT / "limpieza_reporte.csv", index=False)

    # Deriva temporal del target entre meses
    both = pd.concat([train[["source_month", TARGET]], val[["source_month", TARGET]]])
    drift = both.groupby("source_month")[TARGET].describe(percentiles=[.25, .5, .75, .95]).round(2)
    drift.to_csv(OUT / "deriva_mensual.csv")
    E.plot_monthly_target(both, FIG / "06_deriva_target_por_mes.png")
    del both

    train = add_features(train)
    val = add_features(val)
    X_tr, y_tr = train[ALL_FEATURES], train[TARGET]
    X_va, y_va = val[ALL_FEATURES], val[TARGET]

    models = make_models(len(train), args.rf_max_samples)
    results, preds = {}, {}
    for name, model in models.items():
        t = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t
        p = np.clip(model.predict(X_va), 0, None)
        results[name] = E.regression_metrics(y_va, p) | {"fit_seg": round(fit_s, 1)}
        preds[name] = p
        print(f"  {name:18s} MAE={results[name]['MAE_min']:.3f}  RMSE={results[name]['RMSE_min']:.3f}"
              f"  R2={results[name]['R2']:.4f}  ({fit_s:.0f}s)")

    # Experimento de sensibilidad: sin distancia
    sm = sensitivity_model().fit(X_tr, y_tr)
    p_sd = np.clip(sm.predict(X_va), 0, None)
    sens = E.regression_metrics(y_va, p_sd)
    print(f"  {'hgb_sin_distancia':18s} MAE={sens['MAE_min']:.3f}  (sensibilidad)")

    candidates = {k: v for k, v in results.items() if not k.startswith("baseline")}
    best = min(candidates, key=lambda k: candidates[k]["MAE_min"])
    base_mae = results["baseline_lineal"]["MAE_min"]
    print(f"Mejor modelo en validación: {best} "
          f"(mejora MAE vs baseline lineal: {100*(1-results[best]['MAE_min']/base_mae):.1f}%)")

    # Análisis de errores por segmento (mejor modelo vs baseline lineal)
    seg = E.segment_errors(val, preds[best], baseline_pred=preds["baseline_lineal"])
    seg.to_csv(OUT / "errores_por_segmento_validacion.csv", index=False)

    # Importancia por permutación (sobre variables originales, en submuestra de validación)
    n_imp = min(args.n_importancia, len(val))
    idx = np.random.default_rng(SEED).choice(len(val), n_imp, replace=False)
    pi = permutation_importance(models[best], X_va.iloc[idx], y_va.iloc[idx],
                                scoring="neg_mean_absolute_error", n_repeats=3,
                                random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": ALL_FEATURES, "importancia_media": pi.importances_mean.round(4),
                        "desv": pi.importances_std.round(4)}).sort_values("importancia_media", ascending=False)
    imp.to_csv(OUT / "importancia_variables.csv", index=False)

    # Figuras
    E.plot_model_comparison(results, FIG / "07_comparacion_modelos_validacion.png")
    E.plot_pred_vs_real(y_va, preds[best], FIG / "08_pred_vs_real_validacion.png",
                        f"{best} — validación (marzo)")
    E.plot_segment_mae(seg, "distancia (mi)", FIG / "09_mae_por_distancia.png",
                       "MAE por distancia del viaje (validación)")
    E.plot_segment_mae(seg, "duración real (min)", FIG / "10_mae_por_duracion_real.png",
                       "MAE por duración real (validación)")
    E.plot_residuals_by_hour(val, preds[best], FIG / "11_error_por_hora.png")
    E.plot_importance(imp, FIG / "12_importancia_permutacion.png",
                      f"Importancia por permutación — {best}")

    out = {
        "etapa": "validacion",
        "split": {"train": TRAIN_MONTHS, "validacion": VAL_MONTH},
        "n_train": int(len(train)), "n_val": int(len(val)),
        "muestreo": {"n_por_mes_train": args.n_por_mes, "n_val": args.n_val},
        "resultados": results,
        "sensibilidad_sin_distancia": sens,
        "mejor_modelo": best,
        "mejora_mae_vs_baseline_lineal_pct": round(100 * (1 - results[best]["MAE_min"] / base_mae), 2),
        "features": ALL_FEATURES,
        "seed": SEED,
    }
    (OUT / "metrics_validacion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Etapa validación lista en {time.time()-t0:.0f}s")
    return out


# ----------------------------------------------------------------------------
# Etapa 2: test (abril) — se ejecuta UNA vez con el modelo ya elegido
# ----------------------------------------------------------------------------
def etapa_test(args) -> dict:
    val_path = OUT / "metrics_validacion.json"
    if args.modelo:
        best = args.modelo
    elif val_path.exists():
        best = json.loads(val_path.read_text())["mejor_modelo"]
    else:
        raise SystemExit("Primero corre --etapa validacion (o pasa --modelo).")
    print(f"== ETAPA TEST: train = ene-mar, test = abril | modelo elegido en validación: {best} ==")
    t0 = time.time()

    train = add_features(D.load_clean(TRAIN_MONTHS + [VAL_MONTH], n_per_month=args.n_por_mes,
                                      path=args.data, seed=SEED))
    test, rep_test = D.load_clean([TEST_MONTH], n_per_month=args.n_test, path=args.data,
                                  seed=SEED + 200, return_report=True)
    print(f"train: {len(train):,} filas | test: {len(test):,} filas")

    all_models = make_models(len(train), args.rf_max_samples)
    names = ["baseline_mediana", "baseline_lineal", best]
    results, preds = {}, {}
    final_model = None
    for name in names:
        model = all_models[name].fit(train[ALL_FEATURES], train[TARGET])
        p = predict_in_chunks(model, test)
        results[name] = E.regression_metrics(test[TARGET], p)
        preds[name] = p
        if name == best:
            final_model = model
        print(f"  {name:18s} MAE={results[name]['MAE_min']:.3f}  RMSE={results[name]['RMSE_min']:.3f}"
              f"  R2={results[name]['R2']:.4f}")

    seg = E.segment_errors(add_features(test), preds[best], baseline_pred=preds["baseline_lineal"])
    seg.to_csv(OUT / "errores_por_segmento_test.csv", index=False)
    E.plot_pred_vs_real(test[TARGET], preds[best], FIG / "13_pred_vs_real_test.png",
                        f"{best} — test (abril)")

    try:
        import joblib
        joblib.dump(final_model, OUT / "modelo_final.joblib", compress=3)
    except Exception as exc:  # el modelo es regenerable; no es crítico guardarlo
        print("No se pudo guardar el modelo:", exc)

    base_mae = results["baseline_lineal"]["MAE_min"]
    out = {
        "etapa": "test",
        "split": {"train": TRAIN_MONTHS + [VAL_MONTH], "test": TEST_MONTH},
        "n_train": int(len(train)), "n_test": int(len(test)),
        "modelo_final": best,
        "resultados": results,
        "mejora_mae_vs_baseline_lineal_pct": round(100 * (1 - results[best]["MAE_min"] / base_mae), 2),
        "limpieza_test": rep_test.to_dict(orient="records"),
        "seed": SEED,
    }
    (OUT / "metrics_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Etapa test lista en {time.time()-t0:.0f}s")
    return out


# ----------------------------------------------------------------------------
# Reporte en markdown con las tablas (para pegar/citar en el informe)
# ----------------------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def write_markdown_summary():
    md = ["# Resultados generados automáticamente — Avance 1",
          "", "_Archivo generado por `python src/train.py`. No editar a mano._", ""]
    vp = OUT / "metrics_validacion.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        t = pd.DataFrame(v["resultados"]).T.reset_index(names="modelo")
        md += [f"## Validación (train {', '.join(v['split']['train'])} → validación {v['split']['validacion']})",
               f"n_train = {v['n_train']:,} · n_val = {v['n_val']:,}", "", _md_table(t), "",
               f"**Mejor modelo:** `{v['mejor_modelo']}` — mejora de MAE vs baseline lineal: "
               f"**{v['mejora_mae_vs_baseline_lineal_pct']}%**", "",
               f"**Sensibilidad (HGB sin `trip_distance`):** MAE = {v['sensibilidad_sin_distancia']['MAE_min']} min, "
               f"R² = {v['sensibilidad_sin_distancia']['R2']}", ""]
    if (OUT / "limpieza_reporte.csv").exists():
        md += ["## Limpieza (sobre las muestras cargadas)", "",
               _md_table(pd.read_csv(OUT / "limpieza_reporte.csv")), ""]
    if (OUT / "deriva_mensual.csv").exists():
        md += ["## Distribución del target por mes", "",
               _md_table(pd.read_csv(OUT / "deriva_mensual.csv")), ""]
    if (OUT / "errores_por_segmento_validacion.csv").exists():
        md += ["## Errores por segmento (validación, mejor modelo vs baseline lineal)", "",
               _md_table(pd.read_csv(OUT / "errores_por_segmento_validacion.csv")), ""]
    if (OUT / "importancia_variables.csv").exists():
        md += ["## Importancia por permutación (top 10)", "",
               _md_table(pd.read_csv(OUT / "importancia_variables.csv").head(10)), ""]
    tp = OUT / "metrics_test.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        tt = pd.DataFrame(t["resultados"]).T.reset_index(names="modelo")
        md += [f"## Test (train ene-mar → test {t['split']['test']}) — evaluación única",
               f"n_train = {t['n_train']:,} · n_test = {t['n_test']:,}", "", _md_table(tt), "",
               f"Mejora de MAE vs baseline lineal en test: **{t['mejora_mae_vs_baseline_lineal_pct']}%**", ""]
    (ROOT / "reports" / "resultados_avance1.md").write_text("\n".join(md), encoding="utf-8")
    # metrics.json combinado (estructura recomendada por el curso)
    combined = {}
    for k, p in (("validacion", vp), ("test", tp)):
        if p.exists():
            combined[k] = json.loads(p.read_text())
    (OUT / "metrics.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapa", choices=["validacion", "test", "todo"], default="validacion")
    ap.add_argument("--data", default=str(D.DEFAULT_DATA_PATH))
    ap.add_argument("--n-por-mes", type=int, default=700_000,
                    help="filas muestreadas por mes de entrenamiento (memoria ~4GB)")
    ap.add_argument("--n-val", type=int, default=1_000_000, help="filas muestreadas de marzo")
    ap.add_argument("--n-test", type=int, default=0, help="filas de abril (0 = mes completo)")
    ap.add_argument("--rf-max-samples", type=int, default=300_000, help="filas por árbol del Random Forest")
    ap.add_argument("--n-importancia", type=int, default=30_000)
    ap.add_argument("--modelo", default=None, help="forzar modelo para la etapa test")
    ap.add_argument("--rapido", action="store_true", help="prueba rápida con muestras pequeñas")
    args = ap.parse_args()
    if args.rapido:
        args.n_por_mes, args.n_val, args.n_test = 60_000, 40_000, 40_000
        args.rf_max_samples, args.n_importancia = 30_000, 5_000
    args.n_test = args.n_test or None

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if args.etapa in ("validacion", "todo"):
        etapa_validacion(args)
    if args.etapa in ("test", "todo"):
        etapa_test(args)
    write_markdown_summary()


if __name__ == "__main__":
    main()
