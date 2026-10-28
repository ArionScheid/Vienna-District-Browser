"""Tests for the pipeline stages.  They use tiny synthetic inputs, so the raw GTFS feed is not needed.

Run from the repository root:  pytest kg/tests
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rdflib import Graph, Namespace, RDF

from kg.config import BASE_URI, SCHEMA_PATH, NUM_DISTRICTS

VN = Namespace(BASE_URI)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class TestParsers:
    def test_truncate_stop_id(self):
        from kg.data.parsers import _truncate_stop_id
        assert _truncate_stop_id("at:43:123:0:1") == "at:43:123"
        assert _truncate_stop_id("at:43:456") == "at:43:456"
        assert _truncate_stop_id("short") == "short"

    @pytest.mark.parametrize("route_id, line, mode", [
        ("21-U6-j26-31", "U6", "subway"),
        ("22-18-j26-6", "18", "tram"),
        ("23-13A-j26-1", "13A", "bus"),
        ("24-N25-j26-2", "N25", "night bus"),
        ("11-WLB-j26-6", "WLB", "rail"),
    ])
    def test_route_to_line_and_mode(self, route_id, line, mode):
        from kg.data.parsers import route_to_line, route_to_mode
        assert route_to_line(route_id) == line
        assert route_to_mode(route_id) == mode

    def test_parse_stations_merges_platforms(self, tmp_path):
        from kg.data.parsers import parse_stops, parse_stations
        (tmp_path / "stops.txt").write_text(
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "at:49:1:0:1,Karlsplatz,48.200,16.370\n"
            "at:49:1:0:2,Karlsplatz,48.202,16.372\n"
            "at:49:2:0:1,Oper,48.203,16.369\n"
        )
        with patch("kg.data.parsers.GTFS_DIR", tmp_path):
            stations = parse_stations(parse_stops())
        assert len(stations) == 2
        karlsplatz = stations.set_index("station_id").loc["at:49:1"]
        assert karlsplatz["platforms"] == 2
        assert karlsplatz["lat"] == pytest.approx(48.201)

    def test_station_lines_and_modes(self):
        from kg.data.parsers import station_lines, line_modes
        stop_times = pd.DataFrame({"stop_id": ["S1", "S1", "S2"], "trip_id": ["T1", "T2", "T3"]})
        trips = pd.DataFrame({
            "trip_id": ["T1", "T2", "T3"],
            "line": ["U1", "U1", "41A"],
            "mode": ["subway", "subway", "night bus"],
        })
        pairs = station_lines(stop_times, trips)
        assert sorted(map(tuple, pairs.values)) == [("S1", "U1"), ("S2", "41A")]
        # a line seen as night bus *and* bus is a regular bus
        trips_both = pd.concat([trips, pd.DataFrame({"trip_id": ["T4"], "line": ["41A"], "mode": ["bus"]})])
        assert line_modes(trips_both)["41A"] == "bus"

    def test_parse_housing_filters_implausible_listings(self, tmp_path):
        from kg.data.parsers import parse_housing
        data = [
            {"lat": 48.2, "lon": 16.3, "price": 1000, "size": 50, "rooms": 2},   # kept: 20 €/m²
            {"lat": 48.3, "lon": 16.4, "price": 0, "size": 30, "rooms": 1},      # no price
            {"lat": 48.3, "lon": 16.4, "price": 300, "size": 5, "rooms": 1},     # too small
            {"lat": 48.3, "lon": 16.4, "price": 4000, "size": 10, "rooms": 1},   # 400 €/m²
        ]
        path = tmp_path / "listings.json"
        path.write_text(json.dumps(data))
        with patch("kg.data.parsers.HOUSING_JSON", path):
            df = parse_housing()
        assert len(df) == 1
        assert df.iloc[0]["price_per_sqm"] == 20.0


# ---------------------------------------------------------------------------
# District aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def _districts(self):
        return pd.DataFrame({"district": range(1, NUM_DISTRICTS + 1), "area_km2": [2.0] * NUM_DISTRICTS})

    def test_all_districts_present_and_missing_rent_is_nan(self):
        from kg.data.district_mapper import aggregate_per_district
        stations = pd.DataFrame({"station_id": ["S1", "S2"], "district": [1, 2]})
        pairs = pd.DataFrame({"station_id": ["S1", "S1", "S2"], "line": ["U1", "2", "U1"]})
        housing = pd.DataFrame({
            "price": [800.0, 900.0, 1000.0, 500.0],
            "price_per_sqm": [10.0, 12.0, 14.0, 9.0],
            "district": [1, 1, 1, 2],
        })
        result = aggregate_per_district(stations, pairs, housing, self._districts())
        assert len(result) == NUM_DISTRICTS
        assert result.loc[1, "line_count"] == 2
        assert result.loc[1, "median_rent_per_sqm"] == 12.0
        # one listing is not enough → NaN, never 0 (0 would look "cheapest")
        assert np.isnan(result.loc[2, "median_rent_per_sqm"])
        assert result.loc[3, "station_count"] == 0


# ---------------------------------------------------------------------------
# Knowledge graph & reasoning
# ---------------------------------------------------------------------------

def _toy_graph() -> Graph:
    """District 1 and 2 share line U1; district 3 is only reached by a night bus."""
    g = Graph()
    g.parse(str(SCHEMA_PATH), format="turtle")
    for line, mode in [("U1", "subway"), ("N25", "night_bus")]:
        g.add((VN[f"line/{line}"], RDF.type, VN.Line))
        g.add((VN[f"line/{line}"], VN.hasMode, VN[f"mode/{mode}"]))
    for station, district, lines in [("a", 1, ["U1"]), ("b", 2, ["U1", "N25"]), ("c", 3, ["N25"])]:
        s = VN[f"station/{station}"]
        g.add((s, RDF.type, VN.Station))
        g.add((s, VN.locatedInDistrict, VN[f"district/{district}"]))
        for line in lines:
            g.add((s, VN.servedByLine, VN[f"line/{line}"]))
    return g


class TestKnowledgeGraph:
    def test_schema_defines_classes(self):
        from rdflib import OWL
        g = Graph()
        g.parse(str(SCHEMA_PATH), format="turtle")
        classes = {str(s).replace(BASE_URI, "") for s in g.subjects(RDF.type, OWL.Class)}
        assert {"District", "Station", "Line", "TransportMode", "HousingListing"} <= classes

    def test_rules_infer_serves_and_connects(self):
        from kg.reasoning import materialize
        g = _toy_graph()
        materialize(g)
        assert (VN["line/U1"], VN.servesDistrict, VN["district/2"]) in g
        assert (VN["district/1"], VN.connectsTo, VN["district/2"]) in g
        assert (VN["district/2"], VN.connectsTo, VN["district/1"]) in g
        # night buses do not create connectsTo edges
        assert (VN["district/2"], VN.connectsTo, VN["district/3"]) not in g

    def test_direct_lines_to_centre(self):
        from kg.reasoning import materialize, direct_lines_to_centre, accessibility_scores
        g = _toy_graph()
        materialize(g)
        counts = direct_lines_to_centre(g)
        assert counts[2] == 1 and counts[3] == 0
        scores = accessibility_scores(counts)
        assert scores[1] == 1.0 and scores[2] == 1.0 and scores[3] == 0.0

    def test_tercile_labels_keep_missing_values(self):
        from kg.reasoning import tercile_labels
        labels = tercile_labels({1: 1.0, 2: 2.0, 3: 3.0, 4: None}, ("low", "mid", "high"))
        assert labels == {1: "low", 2: "mid", 3: "high", 4: "no data"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_is_bounded_even_with_large_weights(self):
        from kg.scoring import match_score
        best = {"affordability": 1.0, "accessibility": 1.0, "transit": 1.0}
        assert match_score(best, {"affordability": 10, "accessibility": 10, "transit": 10}) == pytest.approx(1.0)

    def test_random_scores_within_unit_interval(self):
        from kg.scoring import match_score
        rng = np.random.default_rng(0)
        for _ in range(500):
            comp = {"affordability": None if rng.random() < 0.2 else rng.random(),
                    "accessibility": rng.random(), "transit": rng.random()}
            w = dict(zip(comp, rng.random(3) * 10))
            assert 0.0 <= match_score(comp, w) <= 1.0

    def test_missing_component_is_neutral(self):
        from kg.scoring import match_score, NEUTRAL
        comp = {"affordability": None, "accessibility": 1.0, "transit": 1.0}
        assert match_score(comp, {"affordability": 1, "accessibility": 0, "transit": 0}) == NEUTRAL

    def test_min_max_invert(self):
        from kg.scoring import min_max
        assert min_max({1: 10.0, 2: 20.0, 3: None}, invert=True) == {1: 1.0, 2: 0.0, 3: None}


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------

def _toy_features():
    return {d: {
        "station_count": d * 10, "line_count": d, "connectivity": d % 5, "direct_lines_to_centre": 23 - d,
        "components": {"affordability": None if d % 7 == 0 else d / 23, "accessibility": 1 - d / 23, "transit": 0.5},
    } for d in range(1, NUM_DISTRICTS + 1)}


class TestGNN:
    def test_forward_shape_and_range(self):
        import torch
        from kg.gnn import DistrictGCN
        x = torch.randn(NUM_DISTRICTS, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out = DistrictGCN(in_dim=10, hidden_dim=16)(x, edge_index, torch.tensor([0.4, 0.3, 0.3]))
        assert out.shape == (NUM_DISTRICTS,)
        assert torch.all((out >= 0) & (out <= 1))

    def test_labels_match_scoring_function(self):
        from kg.gnn import generate_labels
        from kg.scoring import COMPONENTS, match_score
        features = _toy_features()
        prefs, targets = generate_labels(features, n_samples=5)
        assert prefs.shape == (5, 3) and targets.shape == (5, NUM_DISTRICTS)
        assert torch_allclose(prefs.sum(dim=1), 1.0)
        w = dict(zip(COMPONENTS, prefs[0].tolist()))
        assert targets[0, 6].item() == pytest.approx(match_score(features[7]["components"], w), abs=1e-6)

    def test_node_features_flag_missing_rent(self):
        from kg.gnn import node_features
        x = node_features(_toy_features(), np.zeros((NUM_DISTRICTS, 4)))
        assert x.shape == (NUM_DISTRICTS, 3 + 1 + 4 + 4)
        assert x[6, 3] == 1.0 and x[0, 3] == 0.0   # district 7 has no rent data


def torch_allclose(t, value):
    import torch
    return torch.allclose(t, torch.full_like(t, value), atol=1e-5)
