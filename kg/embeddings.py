"""Train KG embeddings (TransE via PyKEEN) and cluster districts."""

import json

import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kg.config import (
    BASE_URI, KG_OUTPUT, OUTPUT_DIR, EMBEDDINGS_OUTPUT, CLUSTERS_OUTPUT,
    NUM_DISTRICTS, EMBEDDING_DIM, EMBEDDING_EPOCHS, CLUSTER_K_RANGE,
)


def export_triples() -> list[list[str]]:
    """Extract triples from the KG, filtered to district/route-level entities."""
    from rdflib import Graph, RDF

    g = Graph()
    g.parse(str(KG_OUTPUT), format="turtle")

    # Keep triples involving districts and routes (skip individual stops/housing
    # to keep the embedding space focused on district-level structure)
    district_uris = {f"{BASE_URI}district/{d}" for d in range(1, NUM_DISTRICTS + 1)}
    route_uris = {str(s) for s in g.subjects(RDF.type, None)
                  if str(s).startswith(f"{BASE_URI}route/")}
    keep_entities = district_uris | route_uris

    triples = []
    for s, p, o in g:
        s_str, p_str, o_str = str(s), str(p), str(o)
        if s_str in keep_entities and o_str in keep_entities:
            triples.append([s_str, p_str, o_str])

    return triples


def train_embeddings(triples: list[list[str]]) -> tuple[dict[str, np.ndarray], TriplesFactory]:
    """Train TransE on the filtered triples. Returns entity->embedding dict."""
    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
    training, testing = tf.split([0.8, 0.2], random_state=42)

    result = pipeline(
        training=training,
        testing=testing,
        model="TransE",
        model_kwargs={"embedding_dim": EMBEDDING_DIM},
        training_kwargs={"num_epochs": EMBEDDING_EPOCHS},
        random_seed=42,
        device="cpu",
    )

    model = result.model
    entity_embs = model.entity_representations[0]().detach().cpu().numpy()
    id_to_label = training.entity_id_to_label

    embeddings = {}
    for idx, label in id_to_label.items():
        embeddings[label] = entity_embs[idx]

    return embeddings, training


def cluster_districts(
    embeddings: dict[str, np.ndarray],
) -> tuple[dict[int, int], int]:
    """K-means on district embeddings; pick best k by silhouette score."""
    district_ids = list(range(1, NUM_DISTRICTS + 1))
    X = np.array([embeddings[f"{BASE_URI}district/{d}"] for d in district_ids])

    best_k, best_score = CLUSTER_K_RANGE.start, -1.0
    for k in CLUSTER_K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    assignments = {d: int(labels[i]) for i, d in enumerate(district_ids)}
    return assignments, best_k


def export_results(
    embeddings: dict[str, np.ndarray],
    assignments: dict[int, int],
    best_k: int,
) -> None:
    """Save embeddings (.npy) and cluster assignments (.json)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save district embeddings only
    district_embs = {
        d: embeddings[f"{BASE_URI}district/{d}"]
        for d in range(1, NUM_DISTRICTS + 1)
    }
    np.save(str(EMBEDDINGS_OUTPUT), district_embs)

    # Cluster colors for frontend choropleth
    CLUSTER_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    CLUSTER_LABELS = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]

    clusters = {
        "k": best_k,
        "districts": {
            str(d): {
                "cluster_id": cid,
                "label": CLUSTER_LABELS[cid],
                "color": CLUSTER_COLORS[cid],
            }
            for d, cid in assignments.items()
        },
    }
    with open(CLUSTERS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)


if __name__ == "__main__":
    print("Exporting triples...")
    triples = export_triples()
    print(f"  {len(triples)} triples for embedding")

    print("Training TransE embeddings...")
    embeddings, tf = train_embeddings(triples)
    print(f"  {len(embeddings)} entity embeddings (dim={EMBEDDING_DIM})")

    print("Clustering districts...")
    assignments, best_k = cluster_districts(embeddings)
    print(f"  Best k={best_k}")

    export_results(embeddings, assignments, best_k)
    print(f"Saved to {EMBEDDINGS_OUTPUT} and {CLUSTERS_OUTPUT}")

    for d in sorted(assignments):
        print(f"  District {d:2d} -> Cluster {assignments[d]}")
