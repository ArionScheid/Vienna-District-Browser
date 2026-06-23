"""Build an RDF knowledge graph from parsed GTFS + housing data."""

from collections import defaultdict

from rdflib import Graph, Literal, Namespace, RDF, XSD

from kg.config import BASE_URI, SCHEMA_PATH, KG_OUTPUT, OUTPUT_DIR
from kg.data.parsers import (
    parse_stops, parse_stop_times, parse_trips, parse_housing,
    build_lookups, _truncate_stop_id,
)
from kg.data.district_mapper import (
    load_districts, map_all_stops, map_all_flats, aggregate_per_district,
)

VN = Namespace(BASE_URI)


def build_kg() -> Graph:
    """Build the full knowledge graph and return it."""
    # --- Parse data ---
    stops_df = parse_stops()
    stop_times_df = parse_stop_times()
    trips_df = parse_trips()
    housing_df = parse_housing()
    stop_to_trips, trip_to_route = build_lookups(stop_times_df, trips_df)

    districts_gdf = load_districts()
    stops_df = map_all_stops(stops_df, districts_gdf)
    housing_df = map_all_flats(housing_df, districts_gdf)
    agg = aggregate_per_district(stops_df, housing_df, stop_times_df, trips_df)

    # --- Build graph ---
    g = Graph()
    g.parse(str(SCHEMA_PATH), format="turtle")
    g.bind("vn", VN)

    _add_districts(g, agg)
    _add_stops(g, stops_df, stop_to_trips, trip_to_route)
    _add_routes(g, trips_df)
    _add_housing(g, housing_df)
    _add_connections(g, stops_df, stop_to_trips, trip_to_route)

    return g


def _add_districts(g: Graph, agg):
    """Add district nodes with aggregated data properties."""
    for district_num, row in agg.iterrows():
        d = VN[f"district/{district_num}"]
        g.add((d, RDF.type, VN.District))
        g.add((d, VN.districtNumber, Literal(int(district_num), datatype=XSD.integer)))
        g.add((d, VN.stopCount, Literal(int(row.stop_count), datatype=XSD.integer)))
        g.add((d, VN.routeCount, Literal(int(row.route_count), datatype=XSD.integer)))
        g.add((d, VN.avgPrice, Literal(float(row.avg_price), datatype=XSD.float)))
        g.add((d, VN.avgPricePerSqm, Literal(float(row.avg_price_per_sqm), datatype=XSD.float)))


def _add_stops(g: Graph, stops_df, stop_to_trips, trip_to_route):
    """Add stop nodes linked to districts and routes."""
    for _, row in stops_df.iterrows():
        sid = str(row.stop_id)
        s = VN[f"stop/{sid}"]
        g.add((s, RDF.type, VN.Stop))
        g.add((s, VN.latitude, Literal(float(row.stop_lat), datatype=XSD.float)))
        g.add((s, VN.longitude, Literal(float(row.stop_lon), datatype=XSD.float)))

        if not _is_na(row.get("district")):
            g.add((s, VN.locatedInDistrict, VN[f"district/{int(row.district)}"]))

        # Link stop to routes via stop_id -> trips -> routes
        truncated = _truncate_stop_id(sid)
        routes_seen = set()
        for trip_id in stop_to_trips.get(truncated, []):
            route_id = trip_to_route.get(trip_id)
            if route_id and route_id not in routes_seen:
                routes_seen.add(route_id)
                g.add((s, VN.servedByRoute, VN[f"route/{route_id}"]))


def _add_routes(g: Graph, trips_df):
    """Add route nodes (unique route_ids from trips)."""
    for route_id in trips_df["route_id"].unique():
        g.add((VN[f"route/{route_id}"], RDF.type, VN.Route))


def _add_housing(g: Graph, housing_df):
    """Add housing listing nodes linked to districts."""
    for idx, row in housing_df.iterrows():
        h = VN[f"housing/{idx}"]
        g.add((h, RDF.type, VN.HousingListing))
        g.add((h, VN.latitude, Literal(row.lat, datatype=XSD.float)))
        g.add((h, VN.longitude, Literal(row.lon, datatype=XSD.float)))
        g.add((h, VN.price, Literal(row.price, datatype=XSD.float)))
        g.add((h, VN.size, Literal(row["size"], datatype=XSD.float)))

        if not _is_na(row.get("district")):
            g.add((h, VN.locatedInDistrict, VN[f"district/{int(row.district)}"]))


def _add_connections(g: Graph, stops_df, stop_to_trips, trip_to_route):
    """Add connectsTo edges between districts that share at least one route."""
    # Build route -> set of districts
    route_districts: dict[str, set[int]] = defaultdict(set)

    for _, row in stops_df.dropna(subset=["district"]).iterrows():
        truncated = _truncate_stop_id(str(row.stop_id))
        district = int(row.district)
        for trip_id in stop_to_trips.get(truncated, []):
            route_id = trip_to_route.get(trip_id)
            if route_id:
                route_districts[route_id].add(district)

    # For each route touching 2+ districts, add pairwise connectsTo
    connected_pairs: set[tuple[int, int]] = set()
    for districts in route_districts.values():
        dlist = sorted(districts)
        for i, d1 in enumerate(dlist):
            for d2 in dlist[i + 1:]:
                connected_pairs.add((d1, d2))

    for d1, d2 in connected_pairs:
        g.add((VN[f"district/{d1}"], VN.connectsTo, VN[f"district/{d2}"]))
        g.add((VN[f"district/{d2}"], VN.connectsTo, VN[f"district/{d1}"]))


def _is_na(val) -> bool:
    """Check if a value is NaN/None/NA (works for pandas nullable ints)."""
    import pandas as pd
    return pd.isna(val)


def save_kg(g: Graph) -> None:
    """Serialize the graph to Turtle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g.serialize(str(KG_OUTPUT), format="turtle")


if __name__ == "__main__":
    print("Building knowledge graph...")
    graph = build_kg()
    save_kg(graph)

    print(f"Triples: {len(graph)}")
    print(f"Unique subjects: {len(set(graph.subjects()))}")
    print(f"Unique predicates: {len(set(graph.predicates()))}")
    print(f"Unique objects: {len(set(graph.objects()))}")
    print(f"Saved to {KG_OUTPUT}")
