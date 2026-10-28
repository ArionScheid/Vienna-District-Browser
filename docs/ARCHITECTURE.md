# Architecture

The project has two halves that only meet through a handful of JSON files:

```
data/                kg/ (Python, offline)                        src/ (Angular, static site)
─────                ─────────────────────                        ───────────────────────────
GTFS feed ──┐        1 build_kg     RDF graph of asserted facts
listings ───┼──────► 2 reasoning    SPARQL rules + district features ─┐
districts ──┘        3 embeddings   DistMult + k-means zones          ├─► src/assets/data/*.json ─► map + ranking
                     4 gnn          GCN surrogate of the match score  │
                     5 export_web   compact web data ─────────────────┘
```

No server is involved: GitHub Pages serves the built Angular app plus the pre-computed files.

## Knowledge graph

Ontology: [`kg/schema.ttl`](../kg/schema.ttl). Namespace `http://example.org/vienna/`.

| Class | Instances | Notes |
|---|---|---|
| `District` | 23 | number, name, area, aggregated counts, median rent (only if ≥ 3 listings) |
| `Station` | ~1,750 | all platforms sharing a GTFS station ID; position; district |
| `Line` | ~190 | U6, 18, 13A … – all GTFS route variants of a line merged |
| `TransportMode` | 5 | subway, rail (Badner Bahn), tram, bus, night bus |
| `HousingListing` | ~100 | rent, size, rooms, position, district |

Asserted relations: `locatedInDistrict` (Station/Listing → District), `servedByLine`
(Station → Line), `hasMode` (Line → TransportMode).

Inferred by [`kg/reasoning.py`](../kg/reasoning.py) (SPARQL `INSERT`):

| Rule | Body | Head |
|---|---|---|
| R1 | `?s servedByLine ?l . ?s locatedInDistrict ?d` | `?l servesDistrict ?d` |
| R2 | `?l servesDistrict ?a, ?b . ?l hasMode ≠ night bus . ?a ≠ ?b` | `?a connectsTo ?b` |

## District features and the match score

Computed with SPARQL queries over the enriched graph, then written back as data properties:

| Feature | Definition |
|---|---|
| `direct_lines_to_centre` | daytime lines serving both the district and Innere Stadt |
| `accessibility_score` | the above, divided by the maximum over the other 22 districts (Innere Stadt = 1) |
| `transit_density` | stations per km² (official area from the district GeoJSON) |
| `connectivity` | districts reachable without changing (`connectsTo` degree) |
| `median_rent_per_sqm` | median of the listing sample; `null` with fewer than 3 listings |

Score components, all in [0, 1]: affordability = 1 − min-max(median rent/m²),
accessibility = `accessibility_score`, transit = min-max(station density).

**Match score** ([`kg/scoring.py`](../kg/scoring.py), mirrored in
[`src/app/scoring.ts`](../src/app/scoring.ts)):

```
score = Σ wᵢ · componentᵢ / Σ wᵢ          (weights ≥ 0, missing component = 0.5)
```

Because it is a weighted *average* of values in [0, 1], the score can never leave 0–100 %.
Missing rent data counts as average, never as cheapest or most expensive.

## Embeddings and clustering

[`kg/embeddings.py`](../kg/embeddings.py) trains a PyKEEN model on the 1,080
entity-to-entity triples (`servesDistrict`, `connectsTo`, `hasMode`), clusters the 23 district
vectors with k-means (k ∈ 3…5 by silhouette) and names each zone from its average features.

DistMult is used instead of TransE: `servesDistrict` is one-to-many, which TransE
(`head + relation ≈ tail`) cannot represent – its district vectors collapse. Comparison in
[notebook 03](../kg/notebooks/03_embeddings_clustering.ipynb).

## GNN

[`kg/gnn.py`](../kg/gnn.py): two GCN layers with skip connections over the `connectsTo` graph.
Node features: score components, a "rent unknown" flag, normalised counts, and the
embedding. The preference vector is broadcast to every node. Targets are exact match scores
for 400 random (Dirichlet) preference vectors. The skip connections matter because
`connectsTo` is nearly complete – a plain GCN over-smooths all districts to the same value
([notebook 04](../kg/notebooks/04_gnn_evaluation.ipynb)).

## Web data (`src/assets/data/`)

| File | Size | Loaded |
|---|---|---|
| `districts.geojson` | ~45 KB | at start |
| `district_features.json`, `clusters.json`, `meta.json` | ~25 KB | at start |
| `stations.json` | ~190 KB | when the network view opens |
| `listings.json` | ~12 KB | when the network view opens |
| `lines.json` | ~1.6 MB (~0.3 MB gzipped) | when line geometry is first needed |

The first version of the app parsed the raw GTFS files (~600 MB, incl. a 518 MB `stop_times.txt`)
in the browser and created one DOM marker per platform. Now the browser gets one entry per station,
draws markers on a single canvas, and layers them by zoom: U-Bahn/rail stations and hubs
(≥ 5 daytime lines) always, all stations from zoom 14, and only inside the visible area.

## Frontend

Angular 15 + Leaflet (`@asymmetrik/ngx-leaflet`) + Tailwind.

| File | Role |
|---|---|
| `src/app/data.service.ts` | lazy, cached loading of the JSON files (relative URLs → works under any sub-path) |
| `src/app/scoring.ts` | match score (TypeScript twin of `kg/scoring.py`) |
| `src/app/colors.ts` | score ramp, zone colours, official U-Bahn colours |
| `src/app/components/map/` | the map, both views and the side panel |
| `src/app/app.component.html` | page layout, "How it works", sources |

The URL parameter `?view=network` opens the network explorer directly.

## Tests

* `pytest` – parsers, aggregation, inference rules, scoring bounds, GNN shapes and labels
  (synthetic inputs, no raw data needed)
* `npm run test:ci` – scoring bounds and ranking in the frontend
