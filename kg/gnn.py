"""Stage 4 – a graph neural network that scores districts for a preference vector.

Setup
-----
* Graph: the 23 districts, edges = inferred ``connectsTo`` relation.
* Node features: the three score components (+ a "rent unknown" flag), other
  normalised district metrics, and the TransE embedding from stage 3.
* Input preference: weights for (affordability, accessibility, transit),
  broadcast to every node.
* Target: the transparent match score from :mod:`kg.scoring` for random
  preference vectors (Dirichlet samples, i.e. weights summing to 1).

The GNN therefore learns a *surrogate* of the scoring function from graph
structure and features – an exercise in applying message passing to the KG.
The web app uses the exact formula; notebook 04 compares both.

Model note: ``connectsTo`` is almost a complete graph (most districts are
linked by at least one line), so plain GCN layers average every node towards
the same value ("over-smoothing").  Each layer therefore has a skip
connection that keeps the node's own representation.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from rdflib import Graph, Namespace
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from kg.config import (
    BASE_URI, OUTPUT_DIR, GNN_SCORES_OUTPUT, EMBEDDINGS_OUTPUT, NUM_DISTRICTS,
    GNN_HIDDEN_DIM, GNN_LEARNING_RATE, GNN_EPOCHS, RANDOM_SEED,
)
from kg.scoring import COMPONENTS, NEUTRAL, match_score

VN = Namespace(BASE_URI)
EXTRA_FEATURE_KEYS = ["station_count", "line_count", "connectivity", "direct_lines_to_centre"]
NUM_PREFS = len(COMPONENTS)

PROFILES = {
    "budget_commuter": [0.6, 0.3, 0.1],
    "central_living": [0.1, 0.7, 0.2],
    "transit_lover": [0.1, 0.2, 0.7],
    "balanced": [1 / 3, 1 / 3, 1 / 3],
}


def _components(features: dict[int, dict], d: int) -> dict[str, float | None]:
    return features[d]["components"]


def node_features(features: dict[int, dict], embeddings: np.ndarray) -> np.ndarray:
    rows = []
    for d in range(1, NUM_DISTRICTS + 1):
        comp = _components(features, d)
        missing_rent = comp["affordability"] is None
        rows.append(
            [NEUTRAL if comp[c] is None else comp[c] for c in COMPONENTS]
            + [float(missing_rent)]
            + [features[d][k] for k in EXTRA_FEATURE_KEYS]
        )
    raw = np.array(rows, dtype=float)
    # min-max normalise the extra metrics (components are already in [0, 1])
    extra = raw[:, NUM_PREFS + 1:]
    span = extra.max(axis=0) - extra.min(axis=0)
    span[span == 0] = 1.0
    raw[:, NUM_PREFS + 1:] = (extra - extra.min(axis=0)) / span
    return np.hstack([raw, embeddings])


def edge_index(g: Graph) -> torch.Tensor:
    src, dst = [], []
    for s, o in g.subject_objects(VN.connectsTo):
        src.append(int(str(s).rsplit("/", 1)[1]) - 1)
        dst.append(int(str(o).rsplit("/", 1)[1]) - 1)
    return torch.tensor([src, dst], dtype=torch.long)


def build_dataset(g: Graph, features: dict[int, dict]) -> Data:
    embeddings = np.load(str(EMBEDDINGS_OUTPUT))
    x = node_features(features, embeddings)
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index(g), num_nodes=NUM_DISTRICTS)


def target_scores(features: dict[int, dict], pref: list[float]) -> np.ndarray:
    weights = dict(zip(COMPONENTS, pref))
    return np.array([match_score(_components(features, d), weights) for d in range(1, NUM_DISTRICTS + 1)])


def generate_labels(features: dict[int, dict], n_samples: int = 400, seed: int = RANDOM_SEED):
    """Random preference vectors (summing to 1) and the exact scores they imply."""
    rng = np.random.default_rng(seed)
    prefs = rng.dirichlet([1.0] * NUM_PREFS, size=n_samples)
    targets = np.array([target_scores(features, p.tolist()) for p in prefs])
    return torch.tensor(prefs, dtype=torch.float), torch.tensor(targets, dtype=torch.float)


class DistrictGCN(torch.nn.Module):
    """Two GCN layers with skip connections → one score per district."""

    def __init__(self, in_dim: int, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_dim + NUM_PREFS, hidden_dim)
        self.skip1 = torch.nn.Linear(in_dim + NUM_PREFS, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.skip2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, pref.unsqueeze(0).expand(x.size(0), -1)], dim=1)
        h = F.relu(self.conv1(h, edge_index) + self.skip1(h))
        h = F.relu(self.conv2(h, edge_index) + self.skip2(h))
        return torch.sigmoid(self.head(h)).squeeze(-1)  # scores live in [0, 1] like the target


def train(data: Data, features: dict[int, dict], verbose: bool = True) -> DistrictGCN:
    torch.manual_seed(RANDOM_SEED)
    prefs, targets = generate_labels(features)
    n_train = int(len(prefs) * 0.8)

    model = DistrictGCN(in_dim=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)
    best_val, best_state, patience = float("inf"), None, 0

    for epoch in range(1, GNN_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for i in torch.randperm(n_train).tolist():
            loss = F.mse_loss(model(data.x, data.edge_index, prefs[i]), targets[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val = np.mean([
                F.mse_loss(model(data.x, data.edge_index, prefs[i]), targets[i]).item()
                for i in range(n_train, len(prefs))
            ])
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train MSE={train_loss / n_train:.5f}  val MSE={val:.5f}")
        if val < best_val - 1e-6:
            best_val, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val MSE={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    return model


def predict(model: DistrictGCN, data: Data, pref: list[float]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index, torch.tensor(pref, dtype=torch.float)).numpy()


def evaluate(model: DistrictGCN, data: Data, features: dict[int, dict], n: int = 100) -> dict[str, float]:
    """Compare GNN and exact scores on unseen preference vectors."""
    prefs, targets = generate_labels(features, n_samples=n, seed=RANDOM_SEED + 1)
    rhos, maes, top3 = [], [], []
    for p, t in zip(prefs.numpy(), targets.numpy()):
        pred = predict(model, data, p.tolist())
        rhos.append(spearmanr(pred, t).statistic)
        maes.append(np.abs(pred - t).mean())
        top3.append(len(set(np.argsort(-pred)[:3]) & set(np.argsort(-t)[:3])) / 3)
    return {
        "spearman_mean": round(float(np.mean(rhos)), 4),
        "mae_mean": round(float(np.mean(maes)), 4),
        "top3_overlap_mean": round(float(np.mean(top3)), 4),
    }


def export_scores(model: DistrictGCN, data: Data, features: dict[int, dict], metrics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"evaluation": metrics, "profiles": {}}
    for name, pref in PROFILES.items():
        gnn = predict(model, data, pref)
        exact = target_scores(features, pref)
        ranking = sorted(range(NUM_DISTRICTS), key=lambda i: -gnn[i])
        results["profiles"][name] = {
            "preferences": dict(zip(COMPONENTS, [round(w, 3) for w in pref])),
            "ranking": [
                {"district": i + 1, "gnn_score": round(float(gnn[i]), 4), "exact_score": round(float(exact[i]), 4)}
                for i in ranking
            ],
        }
    with open(GNN_SCORES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run(g: Graph, features: dict[int, dict]) -> dict:
    data = build_dataset(g, features)
    print(f"  nodes={data.num_nodes}  node features={data.x.size(1)}  edges={data.edge_index.size(1)}")
    model = train(data, features)
    metrics = evaluate(model, data, features)
    print(f"  unseen preferences: Spearman={metrics['spearman_mean']:.3f}  "
          f"MAE={metrics['mae_mean']:.4f}  top-3 overlap={metrics['top3_overlap_mean']:.2f}")
    export_scores(model, data, features, metrics)
    return metrics


if __name__ == "__main__":
    from kg.build_kg import load_kg
    from kg.config import FEATURES_OUTPUT

    with open(FEATURES_OUTPUT, encoding="utf-8") as fh:
        feats = {int(k): v for k, v in json.load(fh).items()}
    run(load_kg(), feats)
    print(f"Saved to {GNN_SCORES_OUTPUT}")
