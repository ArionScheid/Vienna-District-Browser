"""Rule-based reasoning: derive interpretable district features from the KG."""

import json

from rdflib import Graph, Namespace

from kg.config import (
    BASE_URI, KG_OUTPUT, FEATURES_OUTPUT, OUTPUT_DIR,
    NUM_DISTRICTS, INNERE_STADT_DISTRICT, DISTRICT_AREAS_KM2,
)

VN = Namespace(BASE_URI)


def _load_kg() -> Graph:
    g = Graph()
    g.parse(str(KG_OUTPUT), format="turtle")
    return g


def _get_district_data(g: Graph) -> dict[int, dict]:
    """Extract per-district data properties from the KG."""
    districts = {}
    for d_num in range(1, NUM_DISTRICTS + 1):
        d_uri = VN[f"district/{d_num}"]
        districts[d_num] = {
            "stop_count": int(g.value(d_uri, VN.stopCount) or 0),
            "route_count": int(g.value(d_uri, VN.routeCount) or 0),
            "avg_price": float(g.value(d_uri, VN.avgPrice) or 0),
            "avg_price_per_sqm": float(g.value(d_uri, VN.avgPricePerSqm) or 0),
        }
    return districts


def _get_adjacency(g: Graph) -> dict[int, set[int]]:
    """Build district adjacency from connectsTo triples."""
    adj: dict[int, set[int]] = {d: set() for d in range(1, NUM_DISTRICTS + 1)}
    for s, _, o in g.triples((None, VN.connectsTo, None)):
        s_num = int(str(s).rsplit("/", 1)[1])
        o_num = int(str(o).rsplit("/", 1)[1])
        adj[s_num].add(o_num)
    return adj


def _get_shared_route_counts(g: Graph) -> dict[tuple[int, int], int]:
    """Count routes shared between each pair of districts."""
    # Collect route -> set of districts
    route_districts: dict[str, set[int]] = {}
    for s, _, o in g.triples((None, VN.servedByRoute, None)):
        stop_uri = str(s)
        route_uri = str(o)
        district_uri = g.value(s, VN.locatedInDistrict)
        if district_uri:
            d_num = int(str(district_uri).rsplit("/", 1)[1])
            route_districts.setdefault(route_uri, set()).add(d_num)

    counts: dict[tuple[int, int], int] = {}
    for districts in route_districts.values():
        dlist = sorted(districts)
        for i, d1 in enumerate(dlist):
            for d2 in dlist[i + 1:]:
                key = (d1, d2)
                counts[key] = counts.get(key, 0) + 1
    return counts


def accessibility_scores(g: Graph) -> dict[int, float]:
    """Score based on number of shared routes with district 1, normalized to [0, 1]."""
    shared = _get_shared_route_counts(g)
    raw = {}
    for d_num in range(1, NUM_DISTRICTS + 1):
        if d_num == INNERE_STADT_DISTRICT:
            continue
        key = (min(d_num, INNERE_STADT_DISTRICT), max(d_num, INNERE_STADT_DISTRICT))
        raw[d_num] = shared.get(key, 0)

    max_val = max(raw.values()) if raw else 1
    scores = {INNERE_STADT_DISTRICT: 1.0}
    for d_num, count in raw.items():
        scores[d_num] = round(count / max_val, 4) if max_val else 0.0
    return scores


def transit_density(districts: dict[int, dict]) -> dict[int, tuple[float, str]]:
    """Stops per km² per district; classify as high/medium/low."""
    densities = {}
    for d_num, data in districts.items():
        area = DISTRICT_AREAS_KM2[d_num]
        density = data["stop_count"] / area
        densities[d_num] = density

    values = sorted(densities.values())
    p33, p66 = values[len(values) // 3], values[2 * len(values) // 3]

    result = {}
    for d_num, density in densities.items():
        label = "high" if density >= p66 else ("medium" if density >= p33 else "low")
        result[d_num] = (round(density, 2), label)
    return result


def affordability(districts: dict[int, dict]) -> dict[int, str]:
    """Classify districts by avg price/sqm into affordable/moderate/expensive."""
    prices = {d: data["avg_price_per_sqm"] for d, data in districts.items() if data["avg_price_per_sqm"] > 0}
    if not prices:
        return {d: "unknown" for d in range(1, NUM_DISTRICTS + 1)}

    values = sorted(prices.values())
    p33, p66 = values[len(values) // 3], values[2 * len(values) // 3]

    result = {}
    for d_num in range(1, NUM_DISTRICTS + 1):
        p = districts[d_num]["avg_price_per_sqm"]
        if p == 0:
            result[d_num] = "no_data"
        elif p <= p33:
            result[d_num] = "affordable"
        elif p <= p66:
            result[d_num] = "moderate"
        else:
            result[d_num] = "expensive"
    return result


def connectivity_scores(adj: dict[int, set[int]]) -> dict[int, int]:
    """Number of distinct districts reachable via direct routes."""
    return {d: len(neighbors) for d, neighbors in adj.items()}


def compute_all_features(g: Graph) -> dict[int, dict]:
    """Derive all features for every district. Returns {district_num: {...}}."""
    districts = _get_district_data(g)
    adj = _get_adjacency(g)

    acc = accessibility_scores(g)
    density = transit_density(districts)
    afford = affordability(districts)
    conn = connectivity_scores(adj)

    features = {}
    for d_num in range(1, NUM_DISTRICTS + 1):
        features[d_num] = {
            **districts[d_num],
            "accessibility_score": round(acc[d_num], 4),
            "transit_density": density[d_num][0],
            "transit_density_label": density[d_num][1],
            "affordability_label": afford[d_num],
            "connectivity_score": conn[d_num],
        }
    return features


def export_features(features: dict[int, dict]) -> None:
    """Write features to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in features.items()}
    with open(FEATURES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    print("Loading KG and computing features...")
    graph = _load_kg()
    features = compute_all_features(graph)
    export_features(features)

    for d_num in sorted(features):
        f = features[d_num]
        print(
            f"  District {d_num:2d}: "
            f"access={f['accessibility_score']:.2f}  "
            f"density={f['transit_density']:6.1f} ({f['transit_density_label']:6s})  "
            f"afford={f['affordability_label']:10s}  "
            f"connect={f['connectivity_score']:2d}"
        )
    print(f"\nSaved to {FEATURES_OUTPUT}")
