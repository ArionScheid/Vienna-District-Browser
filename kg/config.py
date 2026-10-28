"""Central configuration: paths, namespaces, hyperparameters and Vienna constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KG_ROOT = Path(__file__).parent

# --- Inputs ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
GTFS_DIR = DATA_DIR / "raw" / "gtfs"                # Wiener Linien GTFS (git-ignored, ~600 MB)
HOUSING_JSON = DATA_DIR / "listings.json"           # slim Willhaben sample (see scripts/extract_listings.py)
DISTRICTS_GEOJSON = DATA_DIR / "districts.geojson"  # official district boundaries (Stadt Wien)

# --- Pipeline outputs (git-ignored) ------------------------------------------
OUTPUT_DIR = KG_ROOT / "output"
KG_OUTPUT = OUTPUT_DIR / "kg.ttl"
FEATURES_OUTPUT = OUTPUT_DIR / "district_features.json"
CLUSTERS_OUTPUT = OUTPUT_DIR / "clusters.json"
EMBEDDINGS_OUTPUT = OUTPUT_DIR / "embeddings.npy"
GNN_SCORES_OUTPUT = OUTPUT_DIR / "gnn_scores.json"
STATIONS_OUTPUT = OUTPUT_DIR / "stations.json"
LINES_OUTPUT = OUTPUT_DIR / "lines.json"

# --- Frontend data (committed; small files the web app loads) ----------------
FRONTEND_DATA_DIR = PROJECT_ROOT / "src" / "assets" / "data"

# --- RDF ---------------------------------------------------------------------
BASE_URI = "http://example.org/vienna/"
SCHEMA_PATH = KG_ROOT / "schema.ttl"

# --- Vienna constants --------------------------------------------------------
NUM_DISTRICTS = 23
INNERE_STADT_DISTRICT = 1

DISTRICT_NAMES: dict[int, str] = {
    1: "Innere Stadt", 2: "Leopoldstadt", 3: "Landstraße", 4: "Wieden",
    5: "Margareten", 6: "Mariahilf", 7: "Neubau", 8: "Josefstadt",
    9: "Alsergrund", 10: "Favoriten", 11: "Simmering", 12: "Meidling",
    13: "Hietzing", 14: "Penzing", 15: "Rudolfsheim-Fünfhaus",
    16: "Ottakring", 17: "Hernals", 18: "Währing", 19: "Döbling",
    20: "Brigittenau", 21: "Floridsdorf", 22: "Donaustadt", 23: "Liesing",
}

# --- Housing data cleaning ---------------------------------------------------
MIN_LISTING_SIZE_M2 = 10            # smaller "flats" are storage rooms / data errors
PRICE_PER_SQM_RANGE = (3.0, 60.0)   # €/m² outside this range is implausible for a monthly rent
MIN_LISTINGS_PER_DISTRICT = 3       # fewer listings → price statistics are reported as missing

# --- Embeddings & clustering -------------------------------------------------
# DistMult instead of TransE: servesDistrict is one-to-many (a line serves many
# districts), which TransE's "head + relation ≈ tail" cannot represent – its
# district vectors collapse (silhouette ≈ 0.06, no geographic signal).  See
# notebook 03 for the comparison.
EMBEDDING_MODEL = "DistMult"
EMBEDDING_DIM = 32
EMBEDDING_EPOCHS = 200
CLUSTER_K_RANGE = range(3, 6)
RANDOM_SEED = 42

# --- GNN ---------------------------------------------------------------------
GNN_HIDDEN_DIM = 32
GNN_LEARNING_RATE = 0.01
GNN_EPOCHS = 200
