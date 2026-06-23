"""Tests for the KG pipeline stages."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Phase 1: Parsers
# ---------------------------------------------------------------------------

class TestParsers:
    def test_truncate_stop_id(self):
        from kg.data.parsers import _truncate_stop_id
        assert _truncate_stop_id("at:43:123:0:1") == "at:43:123"
        assert _truncate_stop_id("at:43:456") == "at:43:456"
        assert _truncate_stop_id("short") == "short"

    def test_truncate_stop_id_preserves_three_segments(self):
        from kg.data.parsers import _truncate_stop_id
        assert _truncate_stop_id("a:b:c:d:e:f") == "a:b:c"

    def test_parse_stops(self, tmp_path):
        from kg.data.parsers import parse_stops
        stops_txt = tmp_path / "stops.txt"
        stops_txt.write_text(
            "stop_id,stop_name,stop_lat,stop_lon,other_col\n"
            "at:43:1,Central,48.210,16.363,x\n"
            "at:43:2,West,48.197,16.339,y\n"
        )
        with patch("kg.data.parsers.GTFS_DIR", tmp_path):
            df = parse_stops()
        assert list(df.columns) == ["stop_id", "stop_name", "stop_lat", "stop_lon"]
        assert len(df) == 2
        assert df.iloc[0]["stop_lat"] == pytest.approx(48.210)

    def test_parse_stop_times_truncates(self, tmp_path):
        from kg.data.parsers import parse_stop_times
        st_txt = tmp_path / "stop_times.txt"
        st_txt.write_text(
            "stop_id,trip_id,arrival_time,departure_time\n"
            "at:43:100:0:1,T1,08:00:00,08:01:00\n"
            "at:43:200:0:2,T2,09:00:00,09:01:00\n"
        )
        with patch("kg.data.parsers.GTFS_DIR", tmp_path):
            df = parse_stop_times()
        assert df.iloc[0]["stop_id"] == "at:43:100"
        assert df.iloc[1]["stop_id"] == "at:43:200"

    def test_parse_housing(self, tmp_path):
        from kg.data.parsers import parse_housing
        data = [
            {"LATITUDE": "48.2", "LONGITUDE": "16.3", "PRICE": "1000",
             "ESTATE_SIZE/LIVING_AREA": "50", "NUMBER_OF_ROOMS": "2"},
            {"LATITUDE": "48.3", "LONGITUDE": "16.4", "PRICE": "0",
             "ESTATE_SIZE/LIVING_AREA": "30", "NUMBER_OF_ROOMS": "1"},
        ]
        housing_json = tmp_path / "flat_info.json"
        housing_json.write_text(json.dumps(data))
        with patch("kg.data.parsers.HOUSING_JSON", housing_json):
            df = parse_housing()
        assert len(df) == 1  # second entry filtered (price=0)
        assert df.iloc[0]["price_per_sqm"] == 20.0

    def test_build_lookups(self):
        from kg.data.parsers import build_lookups
        stop_times_df = pd.DataFrame({
            "stop_id": ["S1", "S1", "S2"],
            "trip_id": ["T1", "T2", "T3"],
        })
        trips_df = pd.DataFrame({
            "trip_id": ["T1", "T2", "T3"],
            "route_id": ["R1", "R1", "R2"],
        })
        stop_to_trips, trip_to_route = build_lookups(stop_times_df, trips_df)
        assert set(stop_to_trips["S1"]) == {"T1", "T2"}
        assert trip_to_route["T1"] == "R1"


# ---------------------------------------------------------------------------
# Phase 1: District Mapper
# ---------------------------------------------------------------------------

class TestDistrictMapper:
    def test_aggregate_includes_all_23(self):
        from kg.data.district_mapper import aggregate_per_district
        stops = pd.DataFrame({
            "stop_id": ["S1", "S2"],
            "district": [1, 2],
        })
        housing = pd.DataFrame({
            "price": [1000.0, 2000.0],
            "price_per_sqm": [20.0, 25.0],
            "district": [1, 2],
        })
        stop_times = pd.DataFrame({
            "stop_id": ["S1", "S2"],
            "trip_id": ["T1", "T2"],
        })
        trips = pd.DataFrame({
            "trip_id": ["T1", "T2"],
            "route_id": ["R1", "R2"],
        })
        result = aggregate_per_district(stops, housing, stop_times, trips)
        assert len(result) == 23
        assert result.loc[1, "stop_count"] == 1
        assert result.loc[3, "stop_count"] == 0


# ---------------------------------------------------------------------------
# Phase 2: Knowledge Graph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    def test_schema_loads(self):
        from rdflib import Graph
        from kg.config import SCHEMA_PATH
        g = Graph()
        g.parse(str(SCHEMA_PATH), format="turtle")
        assert len(g) > 0

    def test_schema_has_classes(self):
        from rdflib import Graph, RDF, OWL
        from kg.config import SCHEMA_PATH, BASE_URI
        g = Graph()
        g.parse(str(SCHEMA_PATH), format="turtle")
        classes = {str(s).replace(BASE_URI, "") for s, _, _ in g.triples((None, RDF.type, OWL.Class))}
        assert "District" in classes
        assert "Stop" in classes
        assert "Route" in classes
        assert "HousingListing" in classes


# ---------------------------------------------------------------------------
# Phase 5: GNN
# ---------------------------------------------------------------------------

class TestGNN:
    def test_district_gcn_forward_shape(self):
        import torch
        from torch_geometric.data import Data
        from kg.gnn import DistrictGCN

        n_nodes = 23
        in_dim = 10
        x = torch.randn(n_nodes, in_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)

        model = DistrictGCN(in_dim=in_dim, hidden_dim=16)
        pref = torch.tensor([0.4, 0.3, 0.3])
        out = model(data.x, data.edge_index, pref)

        assert out.shape == (n_nodes,)

    def test_generate_labels_shape(self):
        from kg.gnn import generate_labels
        features = {str(d): {
            "stop_count": d * 10,
            "route_count": d * 2,
            "avg_price": 800 + d * 50,
            "avg_price_per_sqm": 10 + d,
            "accessibility_score": 1.0 / d,
            "transit_density": d * 3.5,
            "connectivity_score": d,
        } for d in range(1, 24)}

        prefs, targets = generate_labels(features, n_samples=10)
        assert prefs.shape == (10, 3)
        assert targets.shape == (10, 23)
