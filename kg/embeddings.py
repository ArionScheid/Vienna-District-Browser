"""Stage 3 – knowledge-graph embeddings (PyKEEN) and district clustering.

The embedding model sees the *structural* part of the graph – which lines
serve which districts, which districts are connected, and each line's mode:

    (line, servesDistrict, district)   (district, connectsTo, district)
    (line, hasMode, mode)

Districts served by similar sets of lines end up close in embedding space.
K-means on the 23 district vectors then yields "transit zones"; the zone names
shown in the app are generated afterwards from the zones' average features.

Model choice: the project started with TransE, but ``servesDistrict`` is a
one-to-many relation.  TransE models it as ``line + r ≈ district`` for *every*
district of a line, which pulls those districts onto one point; the resulting
district vectors carry almost no structure.  DistMult (a bilinear model) has no
such constraint and recovers geographically coherent zones.  Both are compared
in ``notebooks/03_embeddings_clustering.ipynb``.
"""

import json

import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from rdflib import Graph, Namespace
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kg.config import (
    BASE_URI, OUTPUT_DIR, EMBEDDINGS_OUTPUT, CLUSTERS_OUTPUT, NUM_DISTRICTS,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_EPOCHS, CLUSTER_K_RANGE, RANDOM_SEED,
)

VN = Namespace(BASE_URI)
EMBEDDED_RELATIONS = (VN.servesDistrict, VN.connectsTo, VN.hasMode)


def export_triples(g: Graph) -> list[list[str]]:
    """Entity-to-entity triples used for training (literals and stations excluded)."""
    return [[str(s), str(p), str(o)] for p in EMBEDDED_RELATIONS for s, o in g.subject_objects(p)]


def train_embeddings(
    triples: list[list[str]], model: str = EMBEDDING_MODEL
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Train a KG embedding model; returns entity → vector and link-prediction metrics on a 10 % hold-out."""
    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
    # PyKEEN's split keeps every entity in the training set
    training, testing = tf.split([0.9, 0.1], random_state=RANDOM_SEED)

    result = pipeline(
        training=training,
        testing=testing,
        model=model,
        model_kwargs={"embedding_dim": EMBEDDING_DIM},
        training_kwargs={"num_epochs": EMBEDDING_EPOCHS, "use_tqdm": False},
        evaluation_kwargs={"use_tqdm": False},
        random_seed=RANDOM_SEED,
        device="cpu",
    )

    vectors = result.model.entity_representations[0]().detach().cpu().numpy()
    embeddings = {label: vectors[idx] for label, idx in training.entity_to_id.items()}
    metrics = {
        "mrr": float(result.get_metric("both.realistic.inverse_harmonic_mean_rank")),
        "hits_at_10": float(result.get_metric("both.realistic.hits_at_10")),
        "test_triples": int(testing.num_triples),
    }
    return embeddings, metrics


def district_matrix(embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """23 × dim matrix, row i = district i+1."""
    return np.array([embeddings[f"{BASE_URI}district/{d}"] for d in range(1, NUM_DISTRICTS + 1)])


def cluster_districts(X: np.ndarray) -> tuple[dict[int, int], int, dict[int, float]]:
    """K-means on district embeddings; k chosen by silhouette score."""
    silhouettes = {}
    for k in CLUSTER_K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
        silhouettes[k] = float(silhouette_score(X, labels))
    best_k = max(silhouettes, key=silhouettes.get)

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    assignments = {d: int(labels[d - 1]) for d in range(1, NUM_DISTRICTS + 1)}
    return assignments, best_k, silhouettes


def _relative(
    value: float | None, values: list[float | None], words: tuple[str, str, str], min_spread: float = 0.15
) -> str | None:
    """Describe a zone's value relative to the other zones (lowest / middle / highest third).

    Returns None when the zones barely differ (spread below ``min_spread`` of the mean), so
    that e.g. 9.4 vs 9.5 €/m² is not presented as "lower" vs "higher" rents.
    """
    known = sorted(v for v in values if v is not None)
    if value is None or len(known) < 2 or (known[-1] - known[0]) < min_spread * abs(np.mean(known)):
        return None
    position = known.index(value) / (len(known) - 1)
    return words[0] if position < 1 / 3 else words[1] if position <= 2 / 3 else words[2]


def describe_clusters(assignments: dict[int, int], features: dict[int, dict]) -> list[dict]:
    """Summarise each cluster; clusters are re-numbered from most to least central."""
    raw_ids = sorted(set(assignments.values()))
    summary = []
    for cid in raw_ids:
        members = sorted(d for d, c in assignments.items() if c == cid)
        comp = [features[d]["components"] for d in members]
        rents = [features[d]["median_rent_per_sqm"] for d in members if features[d]["median_rent_per_sqm"]]
        summary.append({
            "raw_id": cid,
            "members": members,
            "mean_accessibility": round(float(np.mean([c["accessibility"] for c in comp])), 3),
            "mean_transit": round(float(np.mean([c["transit"] for c in comp])), 3),
            "median_rent_per_sqm": round(float(np.median(rents)), 2) if rents else None,
        })
    summary.sort(key=lambda s: -(s["mean_accessibility"] + s["mean_transit"]))
    for new_id, s in enumerate(summary):
        s["cluster_id"] = new_id
        s["label"] = f"Zone {chr(65 + new_id)}"
        parts = [
            _relative(s["mean_transit"], [x["mean_transit"] for x in summary], ("sparse", "medium", "dense")),
            _relative(s["mean_accessibility"], [x["mean_accessibility"] for x in summary], ("few", "some", "many")),
            _relative(s["median_rent_per_sqm"], [x["median_rent_per_sqm"] for x in summary], ("lower", "mid", "higher")),
        ]
        phrases = [
            f"{parts[0]} network" if parts[0] else None,
            f"{parts[1]} direct lines to the centre" if parts[1] else None,
            f"{parts[2]} rents" if parts[2] else None,
        ]
        text = ", ".join(p for p in phrases if p) or "average on all features"
        s["description"] = text[0].upper() + text[1:]
    return summary


def export_results(
    X: np.ndarray,
    assignments: dict[int, int],
    silhouettes: dict[int, float],
    metrics: dict[str, float],
    features: dict[int, dict],
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_OUTPUT), X)

    summary = describe_clusters(assignments, features)
    by_raw = {s["raw_id"]: s for s in summary}
    clusters = {
        "k": len(summary),
        "silhouette": {str(k): round(v, 4) for k, v in silhouettes.items()},
        "embedding_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "clusters": [{k: v for k, v in s.items() if k != "raw_id"} for s in summary],
        "districts": {str(d): by_raw[c]["cluster_id"] for d, c in assignments.items()},
    }
    with open(CLUSTERS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)
    return clusters


def run(g: Graph, features: dict[int, dict]) -> dict:
    triples = export_triples(g)
    print(f"  {len(triples):,} triples for embedding")
    embeddings, metrics = train_embeddings(triples)
    print(f"  {EMBEDDING_MODEL} hold-out: MRR={metrics['mrr']:.3f}  Hits@10={metrics['hits_at_10']:.3f}")
    X = district_matrix(embeddings)
    assignments, best_k, silhouettes = cluster_districts(X)
    print(f"  Silhouette by k: {', '.join(f'{k}={s:.3f}' for k, s in silhouettes.items())} -> k={best_k}")
    clusters = export_results(X, assignments, silhouettes, metrics, features)
    for c in clusters["clusters"]:
        print(f"  {c['label']} ({c['description']}): districts {c['members']}")
    return clusters


if __name__ == "__main__":
    from kg.build_kg import load_kg
    from kg.config import FEATURES_OUTPUT

    with open(FEATURES_OUTPUT, encoding="utf-8") as fh:
        feats = {int(k): v for k, v in json.load(fh).items()}
    run(load_kg(), feats)
    print(f"Saved to {EMBEDDINGS_OUTPUT} and {CLUSTERS_OUTPUT}")
