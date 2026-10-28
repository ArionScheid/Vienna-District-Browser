# Data

| File | In git? | Source | Licence |
|---|---|---|---|
| `districts.geojson` | yes | [Bezirksgrenzen Wien](https://www.data.gv.at/katalog/dataset/?q=Bezirksgrenzen%20Wien) – Stadt Wien | CC BY 4.0 |
| `listings.json` | yes | ~100 rental listings from willhaben.at, reduced to position, rent, size, rooms, type | academic demo sample, see below |
| `raw/gtfs/*.txt` | no (~600 MB) | [Wiener Linien GTFS](https://www.data.gv.at/katalog/dataset/ab4a73b6-1c2d-42e1-b4d9-049e04889cf0) – Wiener Linien | CC BY 4.0 |
| `raw/willhaben/*.json` | no | raw search-API responses copied from the browser | – |

The web app does **not** need any of these: the pipeline's small outputs are committed in
`src/assets/data/`. You only need the raw data to re-run the pipeline.

## Re-creating the raw data

1. **GTFS** – download the "GTFS-Daten" ZIP from the link above and extract `stops.txt`,
   `stop_times.txt`, `trips.txt` and `shapes.txt` into `data/raw/gtfs/`.
   The committed results were built from the 2026 timetable (route IDs contain `j26`).
2. **Listings** (optional – `listings.json` is already committed). Open a Willhaben rental search
   for Vienna, copy the JSON response of the `webapi/iad/search/...` request from the browser's
   network tab into `data/raw/willhaben/<n>.json`, then run

   ```bash
   python scripts/extract_listings.py
   ```

   The script keeps only coordinates, rent, size, rooms and property type (no text, images or
   advertiser data) and drops garages, parking spots and short-term rentals.
   Please do not scrape Willhaben; a few manually copied pages are enough for a demo.

## Known limitations of the listing sample

The committed sample comes from four result pages sorted by **price ascending**, i.e. the
cheapest listings on one day in 2026. Rent figures are therefore only a *relative* signal
between districts, not market prices, and districts with fewer than three listings are reported
as "no data".
