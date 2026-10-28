"""Run the full pipeline end-to-end:  python -m kg.pipeline

    raw data ─► 1 build KG ─► 2 reasoning ─► 3 embeddings + clusters ─► 4 GNN ─► 5 web export
"""

import datetime as dt

from kg.build_kg import load_inputs, build_kg, save_kg
from kg.config import KG_OUTPUT, EMBEDDING_MODEL
from kg import reasoning, embeddings, gnn, export_web


def _stage(n: int, title: str) -> None:
    print(f"\n{'=' * 64}\nSTAGE {n}: {title}\n{'=' * 64}")


def run_pipeline() -> None:
    _stage(1, "Parse data & build knowledge graph")
    inputs = load_inputs()
    graph = build_kg(inputs)
    asserted = len(graph)
    print(f"  stations in Vienna: {inputs.stations['district'].notna().sum():,}   "
          f"lines: {len(inputs.line_modes)}   listings: {inputs.housing['district'].notna().sum()}")
    print(f"  asserted triples: {asserted:,}")

    _stage(2, "Rule-based reasoning")
    inferred = reasoning.materialize(graph)
    print(f"  inferred triples (servesDistrict, connectsTo): {inferred:,}")
    features = reasoning.compute_all_features(graph)
    reasoning.write_back(graph, features)
    reasoning.export_features(features)
    save_kg(graph)
    reasoning.print_features(features)
    print(f"  saved {KG_OUTPUT.name} ({len(graph):,} triples)")

    _stage(3, f"KG embeddings ({EMBEDDING_MODEL}) & clustering")
    clusters = embeddings.run(graph, features)

    _stage(4, "GNN district scoring")
    gnn_metrics = gnn.run(graph, features)

    _stage(5, "Export web data")
    meta = {
        "generated": dt.date.today().isoformat(),
        "graph": {
            "asserted_triples": asserted,
            "inferred_triples": inferred,
            "total_triples": len(graph),
            "stations": int(inputs.stations["district"].notna().sum()),
            "lines": len(inputs.line_modes),
            "listings": int(inputs.housing["district"].notna().sum()),
        },
        "embeddings": clusters["embedding_metrics"],
        "clustering": {"k": clusters["k"], "silhouette": clusters["silhouette"][str(clusters["k"])]},
        "gnn": gnn_metrics,
    }
    export_web.export_all(inputs, meta)
    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
