"""Stage 2 – rule-based reasoning over the knowledge graph.

1. **Materialise rules** (SPARQL ``INSERT``):

   * R1  ``?station servedByLine ?line . ?station locatedInDistrict ?d``
         ⇒ ``?line servesDistrict ?d``
   * R2  ``?line servesDistrict ?a, ?b . ?line hasMode ≠ night bus . ?a ≠ ?b``
         ⇒ ``?a connectsTo ?b``

2. **Derive district features** with SPARQL queries over the enriched graph
   and write the key ones back as data properties.
"""

import json

import numpy as np
from rdflib import Graph, Literal, Namespace, XSD

from kg.config import (
    BASE_URI, FEATURES_OUTPUT, OUTPUT_DIR, NUM_DISTRICTS, INNERE_STADT_DISTRICT,
)
from kg.scoring import min_max

VN = Namespace(BASE_URI)
PREFIX = f"PREFIX vn: <{BASE_URI}>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"

RULE_SERVES_DISTRICT = PREFIX + """
INSERT { ?line vn:servesDistrict ?d }
WHERE  { ?station vn:servedByLine ?line ; vn:locatedInDistrict ?d . }
"""

NIGHT_BUS = f"<{BASE_URI}mode/night_bus>"
CENTRE = f"<{BASE_URI}district/{INNERE_STADT_DISTRICT}>"

RULE_CONNECTS_TO = PREFIX + f"""
INSERT {{ ?a vn:connectsTo ?b }}
WHERE  {{
  ?line vn:servesDistrict ?a, ?b ; vn:hasMode ?mode .
  FILTER (?a != ?b && ?mode != {NIGHT_BUS})
}}
"""


def materialize(g: Graph) -> int:
    """Apply the inference rules in place; returns the number of new triples."""
    before = len(g)
    g.update(RULE_SERVES_DISTRICT)
    g.update(RULE_CONNECTS_TO)
    return len(g) - before


def _district_num(uri) -> int:
    return int(str(uri).rsplit("/", 1)[1])


def _district_data(g: Graph) -> dict[int, dict]:
    q = PREFIX + """
    SELECT ?num ?name ?area ?stations ?lines ?listings ?rent ?rentSqm WHERE {
      ?d a vn:District ; vn:districtNumber ?num ; rdfs:label ?name ; vn:areaKm2 ?area ;
         vn:stationCount ?stations ; vn:lineCount ?lines ; vn:listingCount ?listings .
      OPTIONAL { ?d vn:medianRent ?rent ; vn:medianRentPerSqm ?rentSqm . }
    }"""
    out = {}
    for r in g.query(q):
        out[int(r.num)] = {
            "name": str(r.name),
            "area_km2": round(float(r.area), 2),
            "station_count": int(r.stations),
            "line_count": int(r.lines),
            "listing_count": int(r.listings),
            "median_rent": None if r.rent is None else round(float(r.rent), 2),
            "median_rent_per_sqm": None if r.rentSqm is None else round(float(r.rentSqm), 2),
        }
    return out


def _lines_per_district(g: Graph) -> dict[int, dict[str, list[str]]]:
    """district → {mode: [line names]} via the inferred servesDistrict relation."""
    q = PREFIX + """
    SELECT ?d ?lineName ?modeName WHERE {
      ?line vn:servesDistrict ?d ; rdfs:label ?lineName ; vn:hasMode ?mode .
      ?mode rdfs:label ?modeName .
    }"""
    out: dict[int, dict[str, list[str]]] = {d: {} for d in range(1, NUM_DISTRICTS + 1)}
    for r in g.query(q):
        out[_district_num(r.d)].setdefault(str(r.modeName), []).append(str(r.lineName))
    for modes in out.values():
        for mode in modes:
            modes[mode].sort(key=_line_sort_key)
    return out


def _line_sort_key(name: str):
    # U1 < U6, 2 < 18 < 71, 13A < 13B; lettered lines (D, O) last
    digits = "".join(ch for ch in name if ch.isdigit())
    return (not digits, int(digits or 0), name)


def direct_lines_to_centre(g: Graph) -> dict[int, int]:
    """Number of daytime lines that serve both the district and Innere Stadt."""
    q = PREFIX + f"""
    SELECT ?d (COUNT(DISTINCT ?line) AS ?n) WHERE {{
      ?line vn:servesDistrict ?d, {CENTRE} ; vn:hasMode ?mode .
      FILTER (?mode != {NIGHT_BUS})
    }} GROUP BY ?d"""
    counts = {d: 0 for d in range(1, NUM_DISTRICTS + 1)}
    for r in g.query(q):
        counts[_district_num(r.d)] = int(r.n)
    return counts


def connectivity(g: Graph) -> dict[int, int]:
    """Number of other districts reachable without changing (connectsTo degree)."""
    q = PREFIX + """
    SELECT ?a (COUNT(DISTINCT ?b) AS ?n) WHERE { ?a vn:connectsTo ?b } GROUP BY ?a"""
    counts = {d: 0 for d in range(1, NUM_DISTRICTS + 1)}
    for r in g.query(q):
        counts[_district_num(r.a)] = int(r.n)
    return counts


def accessibility_scores(centre_lines: dict[int, int]) -> dict[int, float]:
    """Direct lines to the centre, scaled by the best-connected *other* district.

    Innere Stadt itself is defined as 1.0 (you are already in the centre).
    """
    others = {d: n for d, n in centre_lines.items() if d != INNERE_STADT_DISTRICT}
    best = max(others.values()) or 1
    scores = {d: n / best for d, n in others.items()}
    scores[INNERE_STADT_DISTRICT] = 1.0
    return scores


def tercile_labels(values: dict[int, float | None], labels: tuple[str, str, str]) -> dict[int, str]:
    """Assign low/mid/high labels by terciles of the known values."""
    known = np.array([v for v in values.values() if v is not None])
    p33, p66 = np.quantile(known, [1 / 3, 2 / 3])
    out = {}
    for d, v in values.items():
        if v is None:
            out[d] = "no data"
        else:
            out[d] = labels[0] if v < p33 else labels[1] if v < p66 else labels[2]
    return out


def compute_all_features(g: Graph) -> dict[int, dict]:
    """Derive all features for every district (expects a materialised graph)."""
    districts = _district_data(g)
    lines = _lines_per_district(g)
    centre = direct_lines_to_centre(g)
    access = accessibility_scores(centre)
    conn = connectivity(g)

    density = {d: districts[d]["station_count"] / districts[d]["area_km2"] for d in districts}
    rent = {d: districts[d]["median_rent_per_sqm"] for d in districts}

    density_label = tercile_labels(density, ("low", "medium", "high"))
    afford_label = tercile_labels(rent, ("affordable", "moderate", "expensive"))
    afford_component = min_max(rent, invert=True)
    transit_component = min_max(density)

    features = {}
    for d in range(1, NUM_DISTRICTS + 1):
        features[d] = {
            **districts[d],
            "lines": lines[d],
            "direct_lines_to_centre": centre[d],
            "accessibility_score": round(access[d], 4),
            "transit_density": round(density[d], 2),
            "transit_density_label": density_label[d],
            "connectivity": conn[d],
            "affordability_label": afford_label[d],
            "components": {
                "affordability": None if afford_component[d] is None else round(afford_component[d], 4),
                "accessibility": round(access[d], 4),
                "transit": round(transit_component[d], 4),
            },
        }
    return features


def write_back(g: Graph, features: dict[int, dict]) -> None:
    """Store the derived metrics as data properties on the district nodes."""
    for d, f in features.items():
        node = VN[f"district/{d}"]
        g.add((node, VN.directLinesToCentre, Literal(f["direct_lines_to_centre"], datatype=XSD.integer)))
        g.add((node, VN.accessibilityScore, Literal(f["accessibility_score"], datatype=XSD.float)))
        g.add((node, VN.transitDensity, Literal(f["transit_density"], datatype=XSD.float)))
        g.add((node, VN.connectivity, Literal(f["connectivity"], datatype=XSD.integer)))


def export_features(features: dict[int, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in features.items()}, f, indent=2, ensure_ascii=False)


def print_features(features: dict[int, dict]) -> None:
    for d in sorted(features):
        f = features[d]
        rent = f["median_rent_per_sqm"]
        print(
            f"  {d:2d} {f['name']:22s} lines->centre={f['direct_lines_to_centre']:3d}  "
            f"density={f['transit_density']:5.1f}/km² ({f['transit_density_label']:6s})  "
            f"rent={'   n/a' if rent is None else f'{rent:6.2f}'} €/m² ({f['affordability_label']:10s})  "
            f"connects={f['connectivity']:2d}"
        )


if __name__ == "__main__":
    from kg.build_kg import load_kg, save_kg

    graph = load_kg()
    print(f"Materialised {materialize(graph):,} triples")
    feats = compute_all_features(graph)
    write_back(graph, feats)
    save_kg(graph)
    export_features(feats)
    print_features(feats)
    print(f"\nSaved to {FEATURES_OUTPUT}")
