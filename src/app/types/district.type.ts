export interface DistrictFeatures {
  stop_count: number;
  route_count: number;
  avg_price: number;
  avg_price_per_sqm: number;
  accessibility_score: number;
  transit_density: number;
  transit_density_label: string;
  affordability_label: string;
  connectivity_score: number;
}

export interface ClusterInfo {
  cluster_id: number;
  label: string;
  color: string;
}

export interface ClusterData {
  k: number;
  districts: { [district: string]: ClusterInfo };
}

export interface UserPreference {
  affordability: number;
  accessibility: number;
  transit: number;
}
