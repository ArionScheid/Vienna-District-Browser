/** Shapes of the JSON files written by the pipeline (kg/export_web.py) into src/assets/data/. */

export type Component = 'affordability' | 'accessibility' | 'transit';
export type Weights = Record<Component, number>;
export type TransportMode = 'subway' | 'rail' | 'tram' | 'bus' | 'night bus';

export interface DistrictFeatures {
  name: string;
  area_km2: number;
  station_count: number;
  line_count: number;
  listing_count: number;
  median_rent: number | null;
  median_rent_per_sqm: number | null;
  lines: Partial<Record<TransportMode, string[]>>;
  direct_lines_to_centre: number;
  accessibility_score: number;
  transit_density: number;
  transit_density_label: string;
  connectivity: number;
  affordability_label: string;
  /** Score components in [0, 1]; affordability is null when there is too little rent data. */
  components: Record<Component, number | null>;
}

export type DistrictFeatureMap = Record<string, DistrictFeatures>;

export interface ClusterSummary {
  cluster_id: number;
  label: string;
  description: string;
  members: number[];
  mean_accessibility: number;
  mean_transit: number;
  median_rent_per_sqm: number | null;
}

export interface ClusterData {
  k: number;
  silhouette: Record<string, number>;
  clusters: ClusterSummary[];
  /** district number → cluster_id */
  districts: Record<string, number>;
}

export interface Station {
  name: string;
  lat: number;
  lon: number;
  district: number;
  lines: string[];
  /** Subway/rail stations and hubs with many lines – shown at every zoom level. */
  major: boolean;
}

export interface LineGeometry {
  mode: TransportMode;
  /** Each path is a list of [lat, lon] pairs. */
  paths: [number, number][][];
}

export type LineMap = Record<string, LineGeometry>;

export interface Listing {
  lat: number;
  lon: number;
  rent: number;
  size: number;
  rooms: number;
  type: string;
  rent_per_sqm: number;
  district: number;
}

export interface PipelineMeta {
  generated: string;
  graph: {
    asserted_triples: number;
    inferred_triples: number;
    total_triples: number;
    stations: number;
    lines: number;
    listings: number;
  };
  embeddings: { mrr: number; hits_at_10: number; test_triples: number };
  clustering: { k: number; silhouette: number };
  gnn: { spearman_mean: number; mae_mean: number; top3_overlap_mean: number };
}
