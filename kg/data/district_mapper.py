"""Assign points to Vienna's 23 districts and aggregate per-district statistics."""

import geopandas as gpd
import pandas as pd

from kg.config import DISTRICTS_GEOJSON, NUM_DISTRICTS, MIN_LISTINGS_PER_DISTRICT


def load_districts() -> gpd.GeoDataFrame:
    """District polygons with columns ``district`` (1–23), ``area_km2`` and ``geometry``."""
    gdf = gpd.read_file(DISTRICTS_GEOJSON)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf["district"] = gdf["BEZNR"].astype(int)
    # FLAECHE is the official area in m²
    gdf["area_km2"] = gdf["FLAECHE"].astype(float) / 1e6
    return gdf[["district", "area_km2", "geometry"]].sort_values("district").reset_index(drop=True)


def assign_districts(
    df: pd.DataFrame, districts_gdf: gpd.GeoDataFrame, lat: str = "lat", lon: str = "lon"
) -> pd.DataFrame:
    """Return a copy of ``df`` with a nullable ``district`` column (point-in-polygon).

    Points outside Vienna (the GTFS feed contains a few stops in Lower Austria)
    get ``<NA>``.
    """
    points = gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=gpd.points_from_xy(df[lon], df[lat]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, districts_gdf[["district", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # points exactly on a border
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["district"] = out["district"].astype("Int64")
    return out


def aggregate_per_district(
    stations_df: pd.DataFrame,
    station_lines_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    districts_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Per-district statistics, indexed by district number 1–23.

    Columns: area_km2, station_count, line_count, listing_count,
    median_rent, median_rent_per_sqm.  Rent statistics are NaN when a district
    has fewer than ``MIN_LISTINGS_PER_DISTRICT`` listings – missing data must
    never be confused with "cheap".
    """
    stations = stations_df.dropna(subset=["district"])
    station_count = stations.groupby("district")["station_id"].nunique().rename("station_count")

    lines = station_lines_df.merge(stations[["station_id", "district"]], on="station_id")
    line_count = lines.groupby("district")["line"].nunique().rename("line_count")

    housing = housing_df.dropna(subset=["district"])
    housing_stats = housing.groupby("district").agg(
        listing_count=("price", "size"),
        median_rent=("price", "median"),
        median_rent_per_sqm=("price_per_sqm", "median"),
    )
    too_few = housing_stats["listing_count"] < MIN_LISTINGS_PER_DISTRICT
    housing_stats.loc[too_few, ["median_rent", "median_rent_per_sqm"]] = float("nan")

    result = (
        districts_gdf.set_index("district")[["area_km2"]]
        .reindex(range(1, NUM_DISTRICTS + 1))
        .join(station_count).join(line_count).join(housing_stats)
    )
    count_cols = ["station_count", "line_count", "listing_count"]
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    result.index = result.index.astype(int)
    result.index.name = "district"
    return result
