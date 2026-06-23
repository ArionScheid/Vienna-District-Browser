# Project Structure: Personalized Vienna District Browser

## Overview

This application combines Vienna's public transport (GTFS) and housing data into a knowledge graph, derives district-level insights through rule-based reasoning, KG embeddings, and a GNN — then visualizes the results in an Angular/Leaflet map with an interactive district overlay.

The system has two halves:

1. **Python KG pipeline** (`kg/`) — offline data processing that outputs static JSON files.
2. **Angular frontend** (`src/`) — loads the JSON files and renders an interactive map.

No API server is needed. The pipeline writes JSON to `src/assets/data/`, and Angular serves them as static assets.

---

## Directory Layout

```
kgcourse-project/
│
├── kg/                                  # Python KG pipeline
│   ├── config.py                        # Paths, hyperparameters, namespace URIs, Vienna district areas
│   ├── schema.ttl                       # OWL/RDFS ontology (Turtle)
│   ├── requirements.txt                 # Python dependencies
│   │
│   ├── data/                            # Data ingestion & preprocessing
│   │   ├── parsers.py                   # GTFS + housing JSON → pandas DataFrames
│   │   └── district_mapper.py           # Point-in-polygon district assignment + aggregation
│   │
│   ├── build_kg.py                      # RDF knowledge graph construction
│   ├── reasoning.py                     # Rule-based feature derivation
│   ├── embeddings.py                    # TransE embeddings (PyKEEN) + k-means clustering
│   ├── gnn.py                           # GCN model for preference-based district scoring
│   ├── pipeline.py                      # End-to-end orchestrator
│   │
│   ├── notebooks/                       # Jupyter notebooks for analysis & reporting
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_kg_and_reasoning.ipynb
│   │   ├── 03_embeddings_clustering.ipynb
│   │   └── 04_gnn_evaluation.ipynb
│   │
│   ├── output/                          # Pipeline outputs (git-ignored)
│   │   ├── kg.ttl                       # Serialized RDF knowledge graph
│   │   ├── district_features.json       # Per-district derived metrics
│   │   ├── clusters.json                # District cluster assignments + colors
│   │   ├── embeddings.npy               # District embedding vectors
│   │   └── gnn_scores.json              # Sample GNN predictions
│   │
│   └── tests/
│       └── test_pipeline.py             # pytest suite for all pipeline stages
│
├── src/                                 # Angular 15 frontend
│   ├── app/
│   │   ├── app.module.ts                # Root module (BrowserModule, FormsModule, LeafletModule, HttpClient)
│   │   ├── components/
│   │   │   └── map/
│   │   │       ├── map.component.ts     # Single main component: map + district overlay + sidebar
│   │   │       ├── map.component.html   # Template: controls, map, sidebar, loading status
│   │   │       └── map.component.css    # Minimal component styles
│   │   └── types/
│   │       ├── stop.type.ts             # Stop (GTFS stop record)
│   │       ├── stop-time.type.ts        # StopTime, TripsPerStop
│   │       ├── trip.type.ts             # Trip (trip_id, route_id, shape_id)
│   │       ├── shape.type.ts            # ShapeType, ShapeCsvType
│   │       ├── flat-info.type.ts        # FlatInfo (Willhaben listing)
│   │       └── district.type.ts         # DistrictFeatures, ClusterInfo, ClusterData, UserPreference
│   └── assets/data/
│       ├── wienerlinien/                # GTFS files (git-ignored)
│       ├── flat_info.json               # Housing data (git-ignored)
│       ├── districts.geojson            # Vienna district boundaries
│       ├── district_features.json       # Pipeline output (git-ignored)
│       └── clusters.json               # Pipeline output (git-ignored)
│
├── helpers/
│   └── extractor.py                     # Converts raw Willhaben JSON → flat_info.json
│
├── CLAUDE.md                            # Claude Code instructions
├── project_structure.md                 # This file
└── README.md                            # Setup & usage guide
```

---

## Python Pipeline Components

### `kg/config.py` — Configuration

Central configuration file. All other modules import paths and constants from here.

| Setting | Purpose |
|---------|---------|
| `GTFS_DIR`, `HOUSING_JSON`, `DISTRICTS_GEOJSON` | Input data paths |
| `OUTPUT_DIR`, `KG_OUTPUT`, `FEATURES_OUTPUT`, etc. | Output file paths |
| `FRONTEND_DATA_DIR` | Where pipeline copies results for Angular |
| `BASE_URI`, `SCHEMA_PATH` | RDF namespace and ontology location |
| `EMBEDDING_DIM`, `EMBEDDING_EPOCHS` | TransE hyperparameters (dim=64, epochs=100) |
| `GNN_HIDDEN_DIM`, `GNN_LEARNING_RATE`, `GNN_EPOCHS` | GCN hyperparameters (hidden=32, lr=0.01, epochs=200) |
| `CLUSTER_K_RANGE` | Range of k values for k-means (3–5) |
| `DISTRICT_AREAS_KM2` | Area in km² for each of Vienna's 23 districts |

### `kg/schema.ttl` — Ontology

OWL/RDFS ontology defining the knowledge graph schema:

- **Classes**: `District`, `Stop`, `Route`, `HousingListing`
- **Object properties**: `locatedInDistrict` (Stop/Housing → District), `servedByRoute` (Stop → Route), `connectsTo` (District ↔ District, symmetric)
- **Data properties**: `districtNumber`, `stopCount`, `routeCount`, `avgPrice`, `avgPricePerSqm`, `accessibilityScore`, `latitude`, `longitude`, `price`, `size`

### `kg/data/parsers.py` — Data Ingestion

Parses raw data files into pandas DataFrames.

| Function | Input | Output |
|----------|-------|--------|
| `parse_stops()` | `stops.txt` | DataFrame: `stop_id`, `stop_name`, `stop_lat`, `stop_lon` |
| `parse_stop_times()` | `stop_times.txt` | DataFrame: `stop_id` (truncated to 3 colon segments), `trip_id` |
| `parse_trips()` | `trips.txt` | DataFrame: `trip_id`, `route_id`, `shape_id` |
| `parse_housing()` | `flat_info.json` | DataFrame: `lat`, `lon`, `price`, `size`, `rooms`, `price_per_sqm` |
| `build_lookups()` | stop_times + trips DataFrames | `stop → [trip_ids]` dict, `trip → route_id` dict |

The `_truncate_stop_id()` helper mirrors the frontend's `nthIndex` logic: stop IDs in `stop_times.txt` are longer than in `stops.txt`, so we truncate to the first three colon-delimited segments to match.

### `kg/data/district_mapper.py` — Geographic Assignment

Maps geographic coordinates to Vienna's 23 districts using point-in-polygon operations.

| Function | Purpose |
|----------|---------|
| `load_districts()` | Reads `districts.geojson`, normalizes to EPSG:4326, identifies district number column |
| `map_to_district(lat, lon)` | Returns district number (1–23) for a single point |
| `map_all_stops(stops_df)` | Adds `district` column to stops via spatial join |
| `map_all_flats(housing_df)` | Adds `district` column to housing via spatial join |
| `aggregate_per_district()` | Computes per-district stats: stop count, route count, avg price, avg price/m² |

### `kg/build_kg.py` — Knowledge Graph Construction

Builds the RDF graph by:
1. Parsing all data and mapping to districts
2. Loading the ontology from `schema.ttl`
3. Creating 23 district nodes with aggregated data properties
4. Creating stop nodes linked to districts (`locatedInDistrict`) and routes (`servedByRoute`)
5. Creating route nodes
6. Creating housing listing nodes linked to districts
7. Adding `connectsTo` edges between districts that share at least one transit route

Output: `kg/output/kg.ttl`

### `kg/reasoning.py` — Rule-Based Feature Derivation

Derives interpretable per-district features from the knowledge graph:

| Feature | Method |
|---------|--------|
| **Accessibility score** | Number of shared routes with district 1 (Innere Stadt), normalized to [0, 1] |
| **Transit density** | Stops per km² using `DISTRICT_AREAS_KM2`; classified as high/medium/low by terciles |
| **Affordability** | Districts classified as affordable/moderate/expensive by avg price/m² terciles |
| **Connectivity score** | Count of distinct districts reachable via direct routes (`connectsTo` degree) |

`compute_all_features(graph)` combines all of the above into a single dict per district.

Output: `kg/output/district_features.json`

### `kg/embeddings.py` — KG Embeddings & Clustering

1. **Triple export** — extracts district-level and route-level triples from the KG (filters out individual stops/housing to keep embeddings focused)
2. **TransE training** — trains TransE via PyKEEN (dim=64, 100 epochs) on an 80/20 train/test split
3. **K-means clustering** — clusters district embeddings with k in [3, 5]; selects best k by silhouette score
4. **Export** — saves embeddings to `.npy` and cluster assignments (with labels + colors) to JSON

Output: `kg/output/embeddings.npy`, `kg/output/clusters.json`

### `kg/gnn.py` — GNN District Scoring

A 2-layer GCN that scores districts based on user preference vectors.

| Component | Description |
|-----------|-------------|
| `build_dataset()` | PyG `Data` object: 23 nodes, `connectsTo` edges, features = normalized metrics + embedding dims |
| `generate_labels()` | Synthetic training data: random preference vectors → weighted-sum scores (200 samples) |
| `DistrictGCN` | 2-layer GCN; concatenates node features with broadcast preference vector; outputs scalar score per node |
| `train()` | MSE loss, Adam optimizer, train/val split (80/20), early stopping (patience=20) |
| `predict(pref)` | Scores and ranks all 23 districts for a given `[affordability, accessibility, transit]` vector |
| `export_scores()` | Writes predictions for 4 sample profiles (budget_commuter, central_living, transit_lover, balanced) |

The preference-to-score mapping: affordability uses inverted normalized price/m² (cheaper = higher), accessibility uses the accessibility score, transit uses transit density.

Output: `kg/output/gnn_scores.json`

### `kg/pipeline.py` — Orchestrator

Runs all stages sequentially:

```
Build KG → Rule-Based Reasoning → Embeddings & Clustering → GNN Training
```

Then copies `district_features.json` and `clusters.json` to `src/assets/data/` for the Angular frontend.

---

## Angular Frontend Components

### `MapComponent` (`src/app/components/map/`)

The single meaningful component. It has two view modes:

**Stops & Housing mode** (default):
- Loads GTFS files (`stops.txt`, `stop_times.txt`, `trips.txt`, `shapes.txt`) via `HttpClient` and parses them with PapaParse
- Places stop markers (small dots) and housing markers (price-labeled icons) on the Leaflet map
- Clicking a stop resolves `stop_id → trip_ids → shape_ids → polylines` and draws transit routes

**District overlay mode** (toggle button):
- Renders `districts.geojson` as colored polygons
- Colors by preference-based suitability score (red–green gradient) or by cluster zone
- Three preference sliders (affordability, accessibility, transit) recompute scores and recolor in real time
- Clicking a district opens a sidebar showing: name, cluster badge, suitability percentage, transit metrics (stops, routes, density, connectivity, accessibility), housing metrics (avg price, price/m², affordability label)

All data loading is done directly in the component with RxJS pipelines — there are no services or state management.

### Types (`src/app/types/`)

| Type | File | Fields |
|------|------|--------|
| `Stop` | `stop.type.ts` | GTFS stop record |
| `StopTime`, `TripsPerStop` | `stop-time.type.ts` | Stop-trip associations |
| `Trip` | `trip.type.ts` | trip_id, route_id, shape_id |
| `ShapeType`, `ShapeCsvType` | `shape.type.ts` | Route geometry points |
| `FlatInfo` | `flat-info.type.ts` | Willhaben listing fields |
| `DistrictFeatures` | `district.type.ts` | All derived district metrics |
| `ClusterInfo`, `ClusterData` | `district.type.ts` | Cluster id, label, color |
| `UserPreference` | `district.type.ts` | Affordability/accessibility/transit weights |

---

## Notebooks

Located in `kg/notebooks/`. Run from the `kg/notebooks/` directory (they prepend `..` to `sys.path`).

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration` | Raw GTFS/housing stats, distributions, scatter plots, district aggregation, data quality notes |
| `02_kg_and_reasoning` | KG construction, SPARQL queries (entity counts, connectivity, top districts), feature choropleth maps |
| `03_embeddings_clustering` | TransE training, t-SNE visualization, silhouette analysis for k selection, geographic cluster map |
| `04_gnn_evaluation` | GNN vs weighted-sum baseline comparison, Spearman rank correlation, score maps for 4 user profiles |

---

## Tests

**Python** (`kg/tests/test_pipeline.py`):
- Parser tests: stop_id truncation, DataFrame shapes, housing filtering, lookup construction
- District mapper: aggregation produces all 23 districts
- KG schema: ontology loads, all 4 classes defined
- GNN: forward pass output shape, label generation shape

**Angular** (`src/app/components/map/map.component.spec.ts`):
- Component creation test

---

## How to Run

### Prerequisites

- Node.js + Angular CLI 15 (`npm install -g @angular/cli`)
- Python 3.10+

### Data Setup (one-time)

1. **GTFS data** — download "GTFS-Daten" from data.gv.at → extract into `src/assets/data/wienerlinien/`
   - Required files: `stops.txt`, `stop_times.txt`, `trips.txt`, `shapes.txt`

2. **Housing data** — copy Willhaben JSON into `helpers/wh/`, then:
   ```bash
   python helpers/extractor.py
   ```
   Copy `flat_info.json` into `src/assets/data/`.

3. **District boundaries** — download Vienna district GeoJSON from data.gv.at → save as `src/assets/data/districts.geojson`

### Install Dependencies

```bash
npm install                    # Angular + frontend deps
pip install -r kg/requirements.txt   # Python pipeline deps
```

### Run the Full Pipeline

```bash
python -m kg.pipeline
```

This runs all stages and copies output JSON to `src/assets/data/`.

### Run Individual Pipeline Stages

Each stage is independently runnable (but depends on the previous stage's output):

```bash
python -m kg.build_kg          # Stage 1: build KG → kg/output/kg.ttl
python -m kg.reasoning         # Stage 2: derive features → kg/output/district_features.json
python -m kg.embeddings        # Stage 3: train embeddings + cluster → kg/output/clusters.json
python -m kg.gnn               # Stage 4: train GCN + score → kg/output/gnn_scores.json
```

### Run the Frontend

```bash
npm start                      # dev server at http://localhost:4200
```

### Run Tests

```bash
npm test                       # Angular unit tests (Karma/Jasmine)
cd kg && pytest                # Python pipeline tests
```

### Run Notebooks

```bash
cd kg/notebooks
jupyter notebook
```

---

## Technology Stack

| Concern | Technology |
|---------|------------|
| KG storage & querying | RDF/Turtle via `rdflib`, SPARQL |
| KG embeddings | PyKEEN (TransE) |
| Graph neural network | PyTorch Geometric (GCN) |
| District geometry | GeoJSON + `geopandas` (point-in-polygon via `shapely`) |
| Clustering | scikit-learn (k-means + silhouette) |
| Frontend framework | Angular 15 |
| Map rendering | Leaflet via `@asymmetrik/ngx-leaflet` |
| CSS | Tailwind CSS |
| CSV parsing (frontend) | PapaParse |
| Base map tiles | `maps.wien.gv.at` (Vienna city basemap) |
