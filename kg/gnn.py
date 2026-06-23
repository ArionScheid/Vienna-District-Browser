"""GCN model that scores districts based on user preference vectors."""

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from kg.config import (
    BASE_URI, KG_OUTPUT, FEATURES_OUTPUT, EMBEDDINGS_OUTPUT,
    OUTPUT_DIR, GNN_SCORES_OUTPUT,
    NUM_DISTRICTS, GNN_HIDDEN_DIM, GNN_LEARNING_RATE, GNN_EPOCHS,
)

NUMERIC_FEATURE_KEYS = [
    "stop_count", "route_count", "avg_price", "avg_price_per_sqm",
    "accessibility_score", "transit_density", "connectivity_score",
]
NUM_PREFS = 3  # affordability, accessibility, transit


def build_dataset() -> Data:
    """Build a PyG Data object: 23 nodes, connectsTo edges, derived + embedding features."""
    features = _load_features()
    embeddings = np.load(str(EMBEDDINGS_OUTPUT), allow_pickle=True).item()
    edge_index = _load_edges()

    # Node features: normalized derived metrics + embedding dims
    raw = np.array([
        [features[str(d)][k] for k in NUMERIC_FEATURE_KEYS]
        for d in range(1, NUM_DISTRICTS + 1)
    ])
    # Min-max normalize each column
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (raw - mins) / ranges

    emb_matrix = np.array([embeddings[d] for d in range(1, NUM_DISTRICTS + 1)])

    x = np.hstack([normed, emb_matrix])

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        num_nodes=NUM_DISTRICTS,
    )


def _load_features() -> dict:
    with open(FEATURES_OUTPUT, encoding="utf-8") as f:
        return json.load(f)


def _load_edges() -> torch.Tensor:
    from rdflib import Graph, Namespace
    g = Graph()
    g.parse(str(KG_OUTPUT), format="turtle")
    VN = Namespace(BASE_URI)

    src, dst = [], []
    for s, _, o in g.triples((None, VN.connectsTo, None)):
        # District numbers are 1-based; convert to 0-based indices
        src.append(int(str(s).rsplit("/", 1)[1]) - 1)
        dst.append(int(str(o).rsplit("/", 1)[1]) - 1)

    return torch.tensor([src, dst], dtype=torch.long)


def generate_labels(
    features: dict, n_samples: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic training labels: weighted sums of normalized features.

    Each sample is a (preference_vector, scores_per_district) pair where scores
    are the weighted combination of affordability, accessibility, transit features.
    """
    rng = np.random.RandomState(42)

    raw = np.array([
        [features[str(d)][k] for k in NUMERIC_FEATURE_KEYS]
        for d in range(1, NUM_DISTRICTS + 1)
    ])
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (raw - mins) / ranges

    # Map preference dims to feature columns:
    #   affordability -> avg_price_per_sqm (inverted: lower price = higher score)
    #   accessibility -> accessibility_score
    #   transit       -> transit_density
    afford_col = NUMERIC_FEATURE_KEYS.index("avg_price_per_sqm")
    access_col = NUMERIC_FEATURE_KEYS.index("accessibility_score")
    transit_col = NUMERIC_FEATURE_KEYS.index("transit_density")

    pref_vectors = []
    score_matrices = []

    for _ in range(n_samples):
        weights = rng.dirichlet([1, 1, 1])  # 3 weights summing to 1
        scores = (
            weights[0] * (1.0 - normed[:, afford_col])  # invert: cheap = good
            + weights[1] * normed[:, access_col]
            + weights[2] * normed[:, transit_col]
        )
        pref_vectors.append(weights)
        score_matrices.append(scores)

    return (
        torch.tensor(np.array(pref_vectors), dtype=torch.float),
        torch.tensor(np.array(score_matrices), dtype=torch.float),
    )


class DistrictGCN(torch.nn.Module):
    """2-layer GCN: node features + broadcast preference -> scalar score per district."""

    def __init__(self, in_dim: int, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_dim + NUM_PREFS, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        # Broadcast preference vector to all nodes: [N, in_dim+NUM_PREFS]
        pref_broadcast = pref.unsqueeze(0).expand(x.size(0), -1)
        h = torch.cat([x, pref_broadcast], dim=1)
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return h.squeeze(-1)  # [N]


def train(data: Data, features: dict) -> DistrictGCN:
    """Train the GCN on synthetic preference->score pairs."""
    prefs, targets = generate_labels(features)
    n_train = int(len(prefs) * 0.8)

    model = DistrictGCN(in_dim=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)

    best_val_loss = float("inf")
    patience, patience_counter = 20, 0

    for epoch in range(1, GNN_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for i in range(n_train):
            pred = model(data.x, data.edge_index, prefs[i])
            loss = F.mse_loss(pred, targets[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(n_train, len(prefs)):
                pred = model(data.x, data.edge_index, prefs[i])
                val_loss += F.mse_loss(pred, targets[i]).item()

        val_loss /= max(len(prefs) - n_train, 1)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}  train={train_loss/n_train:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return model


def predict(model: DistrictGCN, data: Data, pref: list[float]) -> list[tuple[int, float]]:
    """Score and rank districts for a given preference vector."""
    model.eval()
    with torch.no_grad():
        pref_t = torch.tensor(pref, dtype=torch.float)
        scores = model(data.x, data.edge_index, pref_t).numpy()
    ranked = sorted(
        [(d + 1, float(scores[d])) for d in range(NUM_DISTRICTS)],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def export_scores(model: DistrictGCN, data: Data) -> None:
    """Write sample predictions for a few preference profiles."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = {
        "budget_commuter": [0.6, 0.3, 0.1],
        "central_living": [0.1, 0.7, 0.2],
        "transit_lover": [0.1, 0.2, 0.7],
        "balanced": [0.33, 0.34, 0.33],
    }

    results = {}
    for name, pref in profiles.items():
        ranked = predict(model, data, pref)
        results[name] = {
            "preferences": {"affordability": pref[0], "accessibility": pref[1], "transit": pref[2]},
            "ranking": [{"district": d, "score": round(s, 4)} for d, s in ranked],
        }

    with open(GNN_SCORES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    print("Building dataset...")
    data = build_dataset()
    print(f"  Nodes: {data.num_nodes}, Features: {data.x.size(1)}, Edges: {data.edge_index.size(1)}")

    features = _load_features()

    print("Training GCN...")
    model = train(data, features)

    print("\nSample predictions:")
    for name, pref in [
        ("Budget commuter", [0.6, 0.3, 0.1]),
        ("Central living", [0.1, 0.7, 0.2]),
        ("Transit lover", [0.1, 0.2, 0.7]),
    ]:
        ranked = predict(model, data, pref)
        top3 = ", ".join(f"D{d}({s:.3f})" for d, s in ranked[:3])
        print(f"  {name:20s} -> Top 3: {top3}")

    export_scores(model, data)
    print(f"\nSaved to {GNN_SCORES_OUTPUT}")
