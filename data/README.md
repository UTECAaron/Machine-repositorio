# Datos

Los parquet no se suben al repositorio por su tamaño (~295 MB combinados).

**Fuente:** NYC Taxi & Limousine Commission — Yellow Taxi Trip Records
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
(uso público, sin restricciones de acceso). Copia del equipo: ver `Link a la data.txt`.

Archivos: `yellow_tripdata_2026-01.parquet` … `yellow_tripdata_2026-04.parquet`
(URL directa: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-MM.parquet`).

Combinarlos (agrega la columna `source_month`, que usa el split temporal):

```python
import sys; sys.path.append("src")
from data import combine_monthly_parquets
combine_monthly_parquets(
    {m: f"data/yellow_tripdata_{m}.parquet" for m in ["2026-01", "2026-02", "2026-03", "2026-04"]},
    "data/yellow_2026_jan_apr.parquet",
)
```

Diccionario oficial de variables: *Yellow Trips Data Dictionary* en la misma página del TLC.
Zonas: `taxi_zone_lookup.csv` (1-263 zonas de NYC, 264 = Unknown, 265 = Outside of NYC; 1 = EWR, 132 = JFK, 138 = LGA).
