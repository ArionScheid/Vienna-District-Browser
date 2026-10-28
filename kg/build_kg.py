"""Stage 1 – build the RDF knowledge graph from GTFS + housing data.

Only *asserted* facts are added here (what the data says directly).  Derived
relations such as ``servesDistrict`` and ``connectsTo`` are inferred by the
rules in :mod:`kg.reasoning`.
"""

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD

from kg.config import BASE_URI, SCHEMA_PATH, KG_OUTPUT, OUTPUT_DIR, DISTRICT_NAMES
from kg.data.parsers import (
    parse_stops, parse_stations, parse_stop_times, parse_trips, parse_housing,
    station_lines, line_modes,
)
from kg.data.district_mapper import load_districts, assign_districts, aggregate_per_district

VN = Namespace(BASE_URI)


@dataclass
class Inputs:
    """All parsed inputs; shared by the KG build and the web export."""
    districts: gpd.GeoDataFrame      # district, area_km2, geometry
    stations: pd.DataFrame           # station_id, name, lat, lon, platforms, district
    station_lines: pd.DataFrame      # station_id, line
    line_modes: dict[str, str]       # line -> mode
    trips: pd.DataFrame              # trip_id, route_id, shape_id, line, mode
    housing: pd.DataFrame            # lat, lon, price, size, rooms, type, price_per_sqm, district
    aggregates: pd.DataFrame         # per-district statistics (see aggregate_per_district)


def load_inputs() -> Inputs:
    """Parse all raw data and map stations and listings to districts."""
    districts = load_districts()
    trips = parse_trips()
    stations = assign_districts(parse_stations(parse_stops()), districts)
    pairs = station_lines(parse_stop_times(), trips)
    housing = assign_districts(parse_housing(), districts)
    return Inputs(
        districts=districts,
        stations=stations,
        station_lines=pairs,
        line_modes=line_modes(trips),
        trips=trips,
        housing=housing,
        aggregates=aggregate_per_district(stations, pairs, housing, districts),
    )


def mode_uri(mode: str):
    return VN[f"mode/{mode.replace(' ', '_')}"]


def build_kg(inputs: Inputs) -> Graph:
    """Create the graph: ontology + districts, stations, lines and listings."""
    g = Graph()
    g.parse(str(SCHEMA_PATH), format="turtle")
    g.bind("vn", VN)

    _add_districts(g, inputs.aggregates)
    _add_lines(g, inputs.line_modes)
    _add_stations(g, inputs.stations, inputs.station_lines)
    _add_housing(g, inputs.housing)
    return g


def _add_districts(g: Graph, agg: pd.DataFrame) -> None:
    for d_num, row in agg.iterrows():
        d = VN[f"district/{d_num}"]
        g.add((d, RDF.type, VN.District))
        g.add((d, RDFS.label, Literal(DISTRICT_NAMES[d_num])))
        g.add((d, VN.districtNumber, Literal(int(d_num), datatype=XSD.integer)))
        g.add((d, VN.areaKm2, Literal(round(float(row.area_km2), 3), datatype=XSD.float)))
        g.add((d, VN.stationCount, Literal(int(row.station_count), datatype=XSD.integer)))
        g.add((d, VN.lineCount, Literal(int(row.line_count), datatype=XSD.integer)))
        g.add((d, VN.listingCount, Literal(int(row.listing_count), datatype=XSD.integer)))
        # Missing rent data is simply not asserted (open-world), never stored as 0
        if pd.notna(row.median_rent):
            g.add((d, VN.medianRent, Literal(float(row.median_rent), datatype=XSD.float)))
            g.add((d, VN.medianRentPerSqm, Literal(float(row.median_rent_per_sqm), datatype=XSD.float)))


def _add_lines(g: Graph, modes: dict[str, str]) -> None:
    for line, mode in modes.items():
        node = VN[f"line/{line}"]
        g.add((node, RDF.type, VN.Line))
        g.add((node, RDFS.label, Literal(line)))
        g.add((node, VN.hasMode, mode_uri(mode)))


def _add_stations(g: Graph, stations: pd.DataFrame, pairs: pd.DataFrame) -> None:
    vienna = stations.dropna(subset=["district"])
    for row in vienna.itertuples(index=False):
        s = VN[f"station/{row.station_id}"]
        g.add((s, RDF.type, VN.Station))
        g.add((s, RDFS.label, Literal(row.name)))
        g.add((s, VN.latitude, Literal(round(row.lat, 6), datatype=XSD.float)))
        g.add((s, VN.longitude, Literal(round(row.lon, 6), datatype=XSD.float)))
        g.add((s, VN.locatedInDistrict, VN[f"district/{int(row.district)}"]))

    in_vienna = set(vienna["station_id"])
    for station_id, line in pairs.itertuples(index=False):
        if station_id in in_vienna:
            g.add((VN[f"station/{station_id}"], VN.servedByLine, VN[f"line/{line}"]))


def _add_housing(g: Graph, housing: pd.DataFrame) -> None:
    for idx, row in housing.dropna(subset=["district"]).iterrows():
        h = VN[f"listing/{idx}"]
        g.add((h, RDF.type, VN.HousingListing))
        g.add((h, VN.latitude, Literal(row.lat, datatype=XSD.float)))
        g.add((h, VN.longitude, Literal(row.lon, datatype=XSD.float)))
        g.add((h, VN.rent, Literal(row.price, datatype=XSD.float)))
        g.add((h, VN.size, Literal(row["size"], datatype=XSD.float)))
        g.add((h, VN.rooms, Literal(row.rooms, datatype=XSD.float)))
        g.add((h, VN.locatedInDistrict, VN[f"district/{int(row.district)}"]))


def save_kg(g: Graph) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g.serialize(str(KG_OUTPUT), format="turtle")


def load_kg() -> Graph:
    g = Graph()
    g.parse(str(KG_OUTPUT), format="turtle")
    return g


if __name__ == "__main__":
    print("Building knowledge graph (asserted facts only; run kg.pipeline for the full graph)...")
    graph = build_kg(load_inputs())
    save_kg(graph)
    print(f"Triples: {len(graph):,}")
    print(f"Saved to {KG_OUTPUT}")
