import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable, shareReplay} from 'rxjs';
import {ClusterData, DistrictFeatureMap, LineMap, Listing, PipelineMeta, Station} from './types/data.types';

/**
 * Loads the pre-computed pipeline outputs. Every file is fetched at most once and only
 * when first requested, so the district view never pays for the network layers.
 * URLs are relative so the app also works under a sub-path (GitHub Pages).
 */
@Injectable({providedIn: 'root'})
export class DataService {
  private cache = new Map<string, Observable<unknown>>();

  constructor(private http: HttpClient) {
  }

  private load<T>(file: string): Observable<T> {
    if (!this.cache.has(file)) {
      this.cache.set(file, this.http.get<T>(`assets/data/${file}`).pipe(shareReplay(1)));
    }
    return this.cache.get(file) as Observable<T>;
  }

  districts = () => this.load<GeoJSON.FeatureCollection>('districts.geojson');
  features = () => this.load<DistrictFeatureMap>('district_features.json');
  clusters = () => this.load<ClusterData>('clusters.json');
  meta = () => this.load<PipelineMeta>('meta.json');
  stations = () => this.load<Station[]>('stations.json');
  lines = () => this.load<LineMap>('lines.json');
  listings = () => this.load<Listing[]>('listings.json');
}
