"""Stage 5 – export compact, pre-aggregated data for the web app.

The browser never sees the raw GTFS feed (~600 MB).  Instead it loads:

* ``districts.geojson``      simplified district polygons             (~100 KB)
* ``district_features.json`` derived features + score components      (stage 2)
* ``clusters.json``          embedding clusters                       (stage 3)
* ``stations.json``          1 entry per station: name, position, lines
* ``lines.json``             simplified geometry per line (lazy-loaded)
* ``listings.json``          cleaned housing sample
* ``meta.json``              pipeline statistics shown in the "How it works" panel
"""

import json
import shutil

import pandas as pd
import shapely
from shapely.geometry import LineString, mapping

from kg.build_kg import Inputs
from kg.config import (
    GTFS_DIR, FRONTEND_DATA_DIR, FEATURES_OUTPUT, CLUSTERS_OUTPUT, DISTRICT_NAMES,
)

COORD_DECIMALS = 5          # ~1 m precision
LINE_SIMPLIFY_DEG = 0.00004  # ~3–4 m; invisible at city zoom, shrinks files a lot
DISTRICT_SIMPLIFY_DEG = 0.0001
MAJOR_MODES = {"subway", "rail"}
MAJOR_MIN_LINES = 5          # stations with this many daytime lines count as hubs


def _write(name: str, obj) -> int:
    path = FRONTEND_DATA_DIR / name
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _round_coords(coords) -> list:
    return [[round(lat, COORD_DECIMALS), round(lon, COORD_DECIMALS)] for lon, lat in coords]


def export_districts(inputs: Inputs) -> int:
    features = []
    for row in inputs.districts.itertuples(index=False):
        geom = shapely.set_precision(
            row.geometry.simplify(DISTRICT_SIMPLIFY_DEG, preserve_topology=True), 10 ** -COORD_DECIMALS
        )
        features.append({
            "type": "Feature",
            "properties": {"district": int(row.district), "name": DISTRICT_NAMES[int(row.district)]},
            "geometry": mapping(geom),
        })
    return _write("districts.geojson", {"type": "FeatureCollection", "features": features})


def export_stations(inputs: Inputs) -> int:
    vienna = inputs.stations.dropna(subset=["district"])
    lines_at = inputs.station_lines.groupby("station_id")["line"].apply(sorted).to_dict()
    out = []
    for row in vienna.itertuples(index=False):
        lines = lines_at.get(row.station_id, [])
        if not lines:
            continue  # stations without scheduled service in this feed
        modes = {inputs.line_modes[line] for line in lines}
        daytime = [line for line in lines if inputs.line_modes[line] != "night bus"]
        out.append({
            "name": row.name,
            "lat": round(row.lat, COORD_DECIMALS),
            "lon": round(row.lon, COORD_DECIMALS),
            "district": int(row.district),
            "lines": lines,
            "major": bool(modes & MAJOR_MODES) or len(daytime) >= MAJOR_MIN_LINES,
        })
    return _write("stations.json", out)


def export_lines(inputs: Inputs) -> int:
    """Merge all shape variants of a line into one simplified multi-path."""
    shapes = pd.read_csv(
        GTFS_DIR / "shapes.txt",
        usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
        dtype={"shape_id": str},
    ).sort_values(["shape_id", "shape_pt_sequence"])
    geoms = {
        sid: LineString(zip(g["shape_pt_lon"], g["shape_pt_lat"]))
        for sid, g in shapes.groupby("shape_id", sort=False)
        if len(g) >= 2
    }
    shapes_per_line = inputs.trips.groupby("line")["shape_id"].unique()

    out = {}
    for line, mode in inputs.line_modes.items():
        parts = [geoms[s] for s in shapes_per_line.get(line, []) if s in geoms]
        if not parts:
            continue
        merged = shapely.line_merge(shapely.unary_union(parts).simplify(LINE_SIMPLIFY_DEG))
        paths = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        out[line] = {"mode": mode, "paths": [_round_coords(p.coords) for p in paths if len(p.coords) >= 2]}
    return _write("lines.json", out)


def export_listings(inputs: Inputs) -> int:
    housing = inputs.housing.dropna(subset=["district"])
    out = [
        {
            "lat": round(r.lat, COORD_DECIMALS), "lon": round(r.lon, COORD_DECIMALS),
            "rent": r.price, "size": r.size, "rooms": r.rooms, "type": r.type,
            "rent_per_sqm": r.price_per_sqm, "district": int(r.district),
        }
        for r in housing.itertuples(index=False)
    ]
    return _write("listings.json", out)


def export_all(inputs: Inputs, meta: dict) -> None:
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sizes = {
        "districts.geojson": export_districts(inputs),
        "stations.json": export_stations(inputs),
        "lines.json": export_lines(inputs),
        "listings.json": export_listings(inputs),
        "meta.json": _write("meta.json", meta),
    }
    for src in (FEATURES_OUTPUT, CLUSTERS_OUTPUT):
        shutil.copy2(src, FRONTEND_DATA_DIR / src.name)
        sizes[src.name] = src.stat().st_size
    for name, size in sizes.items():
        print(f"  {name:24s} {size / 1024:8.1f} KB")
