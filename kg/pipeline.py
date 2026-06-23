"""Run the full KG pipeline end-to-end."""

import shutil

from kg.config import FEATURES_OUTPUT, CLUSTERS_OUTPUT, FRONTEND_DATA_DIR, OUTPUT_DIR
from kg.build_kg import build_kg, save_kg
from kg.reasoning import compute_all_features, export_features
from kg.embeddings import export_triples, train_embeddings, cluster_districts, export_results
from kg.gnn import build_dataset, _load_features, train, export_scores


def run_pipeline() -> None:
    """Run all pipeline stages in order: parse -> build KG -> reason -> embed -> train GNN."""
    print("=" * 60)
    print("STAGE 1: Building Knowledge Graph")
    print("=" * 60)
    graph = build_kg()
    save_kg(graph)
    print(f"  Triples: {len(graph)}")
    print(f"  Unique subjects: {len(set(graph.subjects()))}")

    print("\n" + "=" * 60)
    print("STAGE 2: Rule-Based Reasoning")
    print("=" * 60)
    features = compute_all_features(graph)
    export_features(features)
    print(f"  Exported features for {len(features)} districts")

    print("\n" + "=" * 60)
    print("STAGE 3: KG Embeddings & Clustering")
    print("=" * 60)
    triples = export_triples()
    print(f"  {len(triples)} triples for embedding")
    embeddings, tf = train_embeddings(triples)
    print(f"  {len(embeddings)} entity embeddings")
    assignments, best_k = cluster_districts(embeddings)
    print(f"  Best k={best_k}")
    export_results(embeddings, assignments, best_k)

    print("\n" + "=" * 60)
    print("STAGE 4: GNN Training & Scoring")
    print("=" * 60)
    data = build_dataset()
    print(f"  Nodes: {data.num_nodes}, Features: {data.x.size(1)}, Edges: {data.edge_index.size(1)}")
    feat_dict = _load_features()
    model = train(data, feat_dict)
    export_scores(model, data)

    print("\n" + "=" * 60)
    print("STAGE 5: Copying outputs to frontend")
    print("=" * 60)
    _copy_to_frontend()
    print("\nPipeline complete!")


def _copy_to_frontend() -> None:
    """Copy pipeline outputs to the Angular assets directory."""
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src_file in [FEATURES_OUTPUT, CLUSTERS_OUTPUT]:
        dst = FRONTEND_DATA_DIR / src_file.name
        shutil.copy2(src_file, dst)
        print(f"  Copied {src_file.name} -> {dst}")


if __name__ == "__main__":
    run_pipeline()
