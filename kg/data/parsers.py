"""Parse GTFS text files and housing JSON into pandas DataFrames."""

import json
from collections import defaultdict

import pandas as pd

from kg.config import GTFS_DIR, HOUSING_JSON


def _truncate_stop_id(stop_id: str) -> str:
    """Truncate stop_id to first 3 colon-delimited segments.

    Mirrors the frontend's nthIndex logic: stop IDs in stop_times.txt are
    longer than those in stops.txt, so we strip to match.
    """
    parts = stop_id.split(":")
    return ":".join(parts[:3])


def parse_stops() -> pd.DataFrame:
    """Parse stops.txt → DataFrame with stop_id, stop_name, stop_lat, stop_lon."""
    df = pd.read_csv(GTFS_DIR / "stops.txt")
    df = df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    return df


def parse_stop_times() -> pd.DataFrame:
    """Parse stop_times.txt → DataFrame with truncated stop_id and trip_id."""
    df = pd.read_csv(GTFS_DIR / "stop_times.txt", usecols=["stop_id", "trip_id"])
    df["stop_id"] = df["stop_id"].astype(str).apply(_truncate_stop_id)
    return df


def parse_trips() -> pd.DataFrame:
    """Parse trips.txt → DataFrame with trip_id, route_id, shape_id."""
    df = pd.read_csv(GTFS_DIR / "trips.txt")
    return df[["trip_id", "route_id", "shape_id"]].copy()


def parse_housing() -> pd.DataFrame:
    """Parse flat_info.json → DataFrame with lat, lon, price, size, rooms, price_per_sqm."""
    with open(HOUSING_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        try:
            lat = float(entry["LATITUDE"])
            lon = float(entry["LONGITUDE"])
            price = float(entry.get("PRICE", 0))
            size = float(entry.get("ESTATE_SIZE/LIVING_AREA", 0) or entry.get("ESTATE_SIZE", 0))
            rooms = float(entry.get("NUMBER_OF_ROOMS", 0))
        except (ValueError, TypeError):
            continue
        if price <= 0 or size <= 0:
            continue
        records.append({
            "lat": lat,
            "lon": lon,
            "price": price,
            "size": size,
            "rooms": rooms,
            "price_per_sqm": round(price / size, 2),
        })

    return pd.DataFrame(records)


def build_lookups(
    stop_times_df: pd.DataFrame, trips_df: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build stop→[trip_ids] and trip→route_id lookup dicts."""
    stop_to_trips: dict[str, list[str]] = defaultdict(list)
    for stop_id, trip_id in zip(stop_times_df["stop_id"], stop_times_df["trip_id"]):
        stop_to_trips[stop_id].append(trip_id)

    trip_to_route: dict[str, str] = dict(
        zip(trips_df["trip_id"].astype(str), trips_df["route_id"].astype(str))
    )

    return dict(stop_to_trips), trip_to_route
