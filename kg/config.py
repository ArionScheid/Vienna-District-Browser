from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KG_ROOT = Path(__file__).parent

# Data paths
GTFS_DIR = PROJECT_ROOT / "src" / "assets" / "data" / "wienerlinien"
HOUSING_JSON = PROJECT_ROOT / "src" / "assets" / "data" / "flat_info.json"
DISTRICTS_GEOJSON = PROJECT_ROOT / "src" / "assets" / "data" / "districts.geojson"

# Output paths
OUTPUT_DIR = KG_ROOT / "output"
KG_OUTPUT = OUTPUT_DIR / "kg.ttl"
FEATURES_OUTPUT = OUTPUT_DIR / "district_features.json"
CLUSTERS_OUTPUT = OUTPUT_DIR / "clusters.json"
EMBEDDINGS_OUTPUT = OUTPUT_DIR / "embeddings.npy"
GNN_SCORES_OUTPUT = OUTPUT_DIR / "gnn_scores.json"

# Frontend static data (pipeline copies results here for Angular to load)
FRONTEND_DATA_DIR = PROJECT_ROOT / "src" / "assets" / "data"

# RDF namespaces
BASE_URI = "http://example.org/vienna/"
SCHEMA_PATH = KG_ROOT / "schema.ttl"

# Vienna constants
NUM_DISTRICTS = 23
INNERE_STADT_DISTRICT = 1

# Embedding hyperparameters
EMBEDDING_DIM = 64
EMBEDDING_EPOCHS = 100

# GNN hyperparameters
GNN_HIDDEN_DIM = 32
GNN_LEARNING_RATE = 0.01
GNN_EPOCHS = 200

# Clustering
CLUSTER_K_RANGE = range(3, 6)

# Vienna district areas in km² (source: Statistik Austria)
DISTRICT_AREAS_KM2: dict[int, float] = {
    1: 2.87, 2: 19.27, 3: 7.41, 4: 1.78, 5: 2.01,
    6: 1.48, 7: 1.61, 8: 1.09, 9: 2.99, 10: 31.82,
    11: 23.23, 12: 8.21, 13: 37.71, 14: 33.76, 15: 3.92,
    16: 8.67, 17: 11.35, 18: 6.28, 19: 24.95, 20: 5.67,
    21: 44.46, 22: 102.29, 23: 32.06,
}
