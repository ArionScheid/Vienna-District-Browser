"""Parse the raw GTFS feed and the housing sample into pandas DataFrames.

Terminology used throughout the project:

* **platform** – a GTFS stop as listed in ``stops.txt`` (``at:49:1234:0:3``)
* **station**  – all platforms that share the first three ID segments
  (``at:49:1234``); this is what a passenger perceives as "one stop"
* **route**    – a GTFS ``route_id`` (``22-18-j26-6``); Wiener Linien publishes
  several route variants per line
* **line**     – the public line name derived from the route ID (``18``, ``U6``,
  ``13A``); lines are what the knowledge graph models
"""

import json

import pandas as pd

from kg.config import (
    GTFS_DIR, HOUSING_JSON, MIN_LISTING_SIZE_M2, PRICE_PER_SQM_RANGE,
)

# First segment of a Wiener Linien route_id → transport mode
MODE_BY_ROUTE_PREFIX = {
    "21": "subway",
    "11": "rail",       # Badner Bahn (WLB)
    "22": "tram",
    "23": "bus",
    "24": "night bus",
    "25": "night bus",
}
# When a line name appears under several prefixes, the most "regular" mode wins
MODE_PRIORITY = ["subway", "rail", "tram", "bus", "night bus"]
# Temporary rail-replacement services are not part of the regular network
EXCLUDED_LINES = {"SEV"}


def _truncate_stop_id(stop_id: str) -> str:
    """Platform ID → station ID (first three colon-delimited segments)."""
    return ":".join(stop_id.split(":")[:3])


def route_to_line(route_id: str) -> str:
    """``22-18-j26-6`` → ``18``; ``21-U6-j26-31`` → ``U6``."""
    parts = route_id.split("-")
    return parts[1] if len(parts) > 1 else route_id


def route_to_mode(route_id: str) -> str:
    return MODE_BY_ROUTE_PREFIX.get(route_id.split("-")[0], "bus")


def parse_stops() -> pd.DataFrame:
    """Platforms from ``stops.txt`` with stop_id, stop_name, stop_lat, stop_lon."""
    df = pd.read_csv(GTFS_DIR / "stops.txt", dtype={"stop_id": str})
    df = df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    return df


def parse_stations(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Merge platforms into stations (mean position, most common name)."""
    df = stops_df.assign(station_id=stops_df["stop_id"].map(_truncate_stop_id))
    return (
        df.groupby("station_id")
        .agg(
            name=("stop_name", lambda s: s.mode().iloc[0]),
            lat=("stop_lat", "mean"),
            lon=("stop_lon", "mean"),
            platforms=("stop_id", "nunique"),
        )
        .reset_index()
    )


def parse_stop_times() -> pd.DataFrame:
    """``stop_times.txt`` reduced to (station_id, trip_id).

    The file has ~6 million rows, so IDs are read as strings and the
    truncation is computed once per unique platform ID.
    """
    df = pd.read_csv(GTFS_DIR / "stop_times.txt", usecols=["stop_id", "trip_id"], dtype=str)
    unique_ids = df["stop_id"].unique()
    mapping = {sid: _truncate_stop_id(sid) for sid in unique_ids}
    df["stop_id"] = df["stop_id"].map(mapping)
    return df


def parse_trips() -> pd.DataFrame:
    """``trips.txt`` with trip_id, route_id, shape_id plus derived line and mode."""
    df = pd.read_csv(GTFS_DIR / "trips.txt", usecols=["trip_id", "route_id", "shape_id"], dtype=str)
    df["line"] = df["route_id"].map(route_to_line)
    df["mode"] = df["route_id"].map(route_to_mode)
    return df[~df["line"].isin(EXCLUDED_LINES)].reset_index(drop=True)


def station_lines(stop_times_df: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    """Unique (station_id, line) pairs: which lines stop at which station."""
    trip_line = trips_df[["trip_id", "line"]]
    pairs = stop_times_df.merge(trip_line, on="trip_id")[["stop_id", "line"]]
    return pairs.drop_duplicates().rename(columns={"stop_id": "station_id"}).reset_index(drop=True)


def line_modes(trips_df: pd.DataFrame) -> dict[str, str]:
    """line → transport mode (resolving lines that appear under several prefixes)."""
    rank = {m: i for i, m in enumerate(MODE_PRIORITY)}
    modes = trips_df[["line", "mode"]].drop_duplicates()
    modes = modes.assign(rank=modes["mode"].map(rank)).sort_values("rank")
    return dict(modes.drop_duplicates("line")[["line", "mode"]].itertuples(index=False))


def parse_housing() -> pd.DataFrame:
    """Housing sample → DataFrame with lat, lon, price, size, rooms, type, price_per_sqm.

    Listings without a positive price, smaller than ``MIN_LISTING_SIZE_M2`` or
    with an implausible €/m² (outside ``PRICE_PER_SQM_RANGE``) are dropped.
    """
    with open(HOUSING_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    lo, hi = PRICE_PER_SQM_RANGE
    records = []
    for entry in raw:
        try:
            lat, lon = float(entry["lat"]), float(entry["lon"])
            price = float(entry.get("price") or 0)
            size = float(entry.get("size") or 0)
            rooms = float(entry.get("rooms") or 0)
        except (KeyError, ValueError, TypeError):
            continue
        if price <= 0 or size < MIN_LISTING_SIZE_M2:
            continue
        per_sqm = price / size
        if not lo <= per_sqm <= hi:
            continue
        records.append({
            "lat": lat, "lon": lon, "price": price, "size": size, "rooms": rooms,
            "type": entry.get("type", ""), "price_per_sqm": round(per_sqm, 2),
        })

    return pd.DataFrame(records, columns=["lat", "lon", "price", "size", "rooms", "type", "price_per_sqm"])
