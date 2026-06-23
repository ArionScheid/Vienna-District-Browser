"""Map geographic coordinates to Vienna's 23 districts using point-in-polygon."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from kg.config import DISTRICTS_GEOJSON, NUM_DISTRICTS


def load_districts() -> gpd.GeoDataFrame:
    """Load Vienna district boundaries from GeoJSON."""
    gdf = gpd.read_file(DISTRICTS_GEOJSON)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Identify the district number column — different GeoJSON sources use
    # different names.  We look for common ones and normalise to "district".
    number_col = None
    for candidate in ("BEZNR", "BEZ", "DISTRICT_CODE", "district", "bez", "OBJECTID"):
        if candidate in gdf.columns:
            number_col = candidate
            break

    if number_col is None:
        # Fallback: try the first integer-typed column
        for col in gdf.columns:
            if pd.api.types.is_integer_dtype(gdf[col]):
                number_col = col
                break

    if number_col is None:
        raise ValueError(
            f"Cannot find district number column in GeoJSON. Columns: {list(gdf.columns)}"
        )

    gdf["district"] = gdf[number_col].astype(int)
    return gdf[["district", "geometry"]].copy()


def map_to_district(lat: float, lon: float, districts_gdf: gpd.GeoDataFrame) -> int | None:
    """Return the district number (1–23) for a lat/lon, or None if outside Vienna."""
    pt = Point(lon, lat)  # GeoJSON is (lon, lat)
    for _, row in districts_gdf.iterrows():
        if row["geometry"].contains(pt):
            return int(row["district"])
    return None


def map_all_stops(stops_df: pd.DataFrame, districts_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add a 'district' column to stops_df via point-in-polygon."""
    geometry = [Point(lon, lat) for lon, lat in zip(stops_df["stop_lon"], stops_df["stop_lat"])]
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(stops_gdf, districts_gdf, how="left", predicate="within")
    joined = joined.rename(columns={"district_right": "district"}).drop(columns=["index_right"], errors="ignore")
    if "district_left" in joined.columns:
        joined = joined.drop(columns=["district_left"])
    joined["district"] = joined["district"].astype("Int64")
    return pd.DataFrame(joined.drop(columns=["geometry"]))


def map_all_flats(housing_df: pd.DataFrame, districts_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add a 'district' column to housing_df via point-in-polygon."""
    geometry = [Point(lon, lat) for lon, lat in zip(housing_df["lon"], housing_df["lat"])]
    housing_gdf = gpd.GeoDataFrame(housing_df, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(housing_gdf, districts_gdf, how="left", predicate="within")
    joined = joined.rename(columns={"district_right": "district"}).drop(columns=["index_right"], errors="ignore")
    if "district_left" in joined.columns:
        joined = joined.drop(columns=["district_left"])
    joined["district"] = joined["district"].astype("Int64")
    return pd.DataFrame(joined.drop(columns=["geometry"]))


def aggregate_per_district(
    stops_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    trips_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-district stats: stop count, route count, avg price, avg price/sqm.

    Both stops_df and housing_df must already have a 'district' column.
    """
    from kg.data.parsers import _truncate_stop_id

    # Stop count per district
    stop_counts = (
        stops_df.dropna(subset=["district"])
        .groupby("district")["stop_id"]
        .nunique()
        .rename("stop_count")
    )

    # Route count per district: stop → trips → routes
    # Truncate stops' IDs to match stop_times' truncated format
    stops_truncated = stops_df[["stop_id", "district"]].dropna().copy()
    stops_truncated["stop_id"] = stops_truncated["stop_id"].apply(_truncate_stop_id)
    stop_trips = stop_times_df.merge(stops_truncated, on="stop_id")
    stop_routes = stop_trips.merge(trips_df[["trip_id", "route_id"]], on="trip_id")
    route_counts = (
        stop_routes.dropna(subset=["district"])
        .groupby("district")["route_id"]
        .nunique()
        .rename("route_count")
    )

    # Housing stats per district
    housing_stats = (
        housing_df.dropna(subset=["district"])
        .groupby("district")
        .agg(avg_price=("price", "mean"), avg_price_per_sqm=("price_per_sqm", "mean"))
    )

    # Combine everything, ensure all 23 districts present
    all_districts = pd.DataFrame({"district": range(1, NUM_DISTRICTS + 1)})
    result = all_districts.set_index("district")
    result = result.join(stop_counts).join(route_counts).join(housing_stats)
    result = result.fillna(0)
    result.index = result.index.astype(int)
    return result
