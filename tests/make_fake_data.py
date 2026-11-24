"""
Genera un parquet SINTÉTICO con el mismo esquema que el combinado del TLC,
solo para probar que el pipeline corre de punta a punta (smoke test / CI)
sin descargar los ~300MB reales.

NO usar sus métricas para nada: los datos son inventados.

Uso:
    python tests/make_fake_data.py --out data/fake_yellow.parquet --n-por-mes 120000
    python src/train.py --etapa todo --rapido --data data/fake_yellow.parquet
"""
import argparse

import numpy as np
import pandas as pd


def make_month(month: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    secs = rng.integers(0, int((end - start).total_seconds()), n)
    pickup = start + pd.to_timedelta(secs, unit="s")
    hour = pickup.hour.to_numpy()

    pu = rng.integers(1, 266, n)
    do = np.where(rng.random(n) < 0.1, pu, rng.integers(1, 266, n))
    dist = rng.lognormal(0.5, 0.8, n).round(2)
    dist[np.isin(pu, [132, 138]) | np.isin(do, [132, 138])] += 10
    zone_speed = 8 + (np.arange(266) % 17)  # mph "por zona"
    rush = 1 + 0.5 * np.isin(hour, [7, 8, 9, 16, 17, 18, 19])
    speed = zone_speed[pu] / rush * rng.lognormal(0, 0.25, n)
    dur_min = dist / speed * 60 + rng.exponential(2, n)

    # errores típicos del TLC
    bad = rng.random(n)
    dur_min[bad < 0.012] *= -1
    dur_min[(bad >= 0.012) & (bad < 0.013)] = 2000
    dist[(bad >= 0.013) & (bad < 0.043)] = 0
    pickup_arr = pickup.to_numpy().copy()
    pickup_arr[(bad >= 0.043) & (bad < 0.044)] = np.datetime64("2009-01-01T00:00:00")
    pickup = pd.DatetimeIndex(pickup_arr)
    dropoff = pickup + pd.to_timedelta(dur_min, unit="m")

    vendor = rng.choice([1, 2, 6, 7], n, p=[0.25, 0.7, 0.01, 0.04])
    missing = rng.random(n) < np.where(vendor == 2, 0.35, 0.05)
    ratecode = np.where(rng.random(n) < 0.95, 1.0, rng.choice([2.0, 3.0, 4.0, 5.0, 99.0], n))
    passengers = rng.choice([1.0, 2.0, 3.0, 4.0, 0.0], n, p=[0.75, 0.15, 0.04, 0.03, 0.03])
    ratecode[missing] = np.nan
    passengers[missing] = np.nan

    return pd.DataFrame({
        "VendorID": vendor.astype("int32"),
        "tpep_pickup_datetime": pickup.astype("datetime64[us]"),
        "tpep_dropoff_datetime": dropoff.astype("datetime64[us]"),
        "passenger_count": passengers,
        "trip_distance": dist,
        "RatecodeID": ratecode,
        "PULocationID": pu.astype("int32"),
        "DOLocationID": do.astype("int32"),
        "payment_type": rng.integers(0, 5, n),
        "fare_amount": (dist * 3 + 3).round(2),
        "source_month": month,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_yellow.parquet")
    ap.add_argument("--n-por-mes", type=int, default=120_000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    df = pd.concat([make_month(m, a.n_por_mes, rng) for m in ["2026-01", "2026-02", "2026-03", "2026-04"]])
    df.to_parquet(a.out, index=False)
    print(f"Escrito {a.out}: {len(df):,} filas (SINTÉTICAS)")
