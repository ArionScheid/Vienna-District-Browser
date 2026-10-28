import {Component, NgZone, OnDestroy} from '@angular/core';
import {
  canvas, circleMarker, geoJSON, GeoJSON as LeafletGeoJSON, GeoJSONOptions, PathOptions, latLng, latLngBounds, layerGroup, Map as LeafletMap,
  MapOptions, Path, polyline, svg, tileLayer,
} from 'leaflet';
import {combineLatest, Subscription} from 'rxjs';
import {DataService} from '../../data.service';
import {COMPONENTS, matchScore, normalizeWeights} from '../../scoring';
import {CLUSTER_COLORS, LISTING_COLOR, lineColor, MODE_LABELS, MODE_ORDER, rampColor, SCORE_RAMP} from '../../colors';
import {
  ClusterData, Component as ScoreComponent, DistrictFeatureMap, DistrictFeatures, LineMap, Listing, Station,
  TransportMode, Weights,
} from '../../types/data.types';

type View = 'districts' | 'network';

/** Below this zoom level only major stations (U-Bahn, rail, hubs) are drawn. */
export const ALL_STATIONS_ZOOM = 14;

interface RankedDistrict {
  district: number;
  name: string;
  score: number;
}

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css'],
})
export class MapComponent implements OnDestroy {
  readonly components = COMPONENTS;
  readonly componentLabels: Record<ScoreComponent, string> = {
    affordability: 'Affordable rent',
    accessibility: 'Direct lines to the centre',
    transit: 'Dense station network',
  };
  readonly modeOrder = MODE_ORDER;
  readonly modeLabels = MODE_LABELS;
  readonly scoreRamp = SCORE_RAMP;
  readonly clusterColors = CLUSTER_COLORS;
  readonly listingColor = LISTING_COLOR;
  readonly allStationsZoom = ALL_STATIONS_ZOOM;

  view: View = 'districts';
  colorMode: 'score' | 'cluster' = 'score';
  /** Raw slider positions (0–10); only their shares matter. */
  weights: Weights = {affordability: 5, accessibility: 5, transit: 5};

  features: DistrictFeatureMap | null = null;
  clusters: ClusterData | null = null;
  ranking: RankedDistrict[] = [];
  scores: Record<number, number> = {};
  selectedDistrict: number | null = null;

  // Network view layers (toggled in the panel)
  showStations = true;
  showSubway = true;
  showTram = false;
  showListings = true;
  networkLoading = false;
  stationCount = 0;
  visibleStationCount = 0;
  zoom = 12;
  selectedStation: Station | null = null;
  lineModes: Record<string, TransportMode> = {};

  options: MapOptions = {
    layers: [
      tileLayer('https://maps.wien.gv.at/basemap/bmapgrau/normal/google3857/{z}/{y}/{x}.png', {
        maxZoom: 19,
        attribution: 'Basemap: <a href="https://basemap.at">basemap.at</a>',
      }),
    ],
    zoom: 12,
    minZoom: 10,
    center: latLng(48.2082, 16.3738),
  };

  private map: LeafletMap | null = null;
  /** Stations, lines and listings share one canvas in their own pane (hidden in the district view). */
  private renderer = canvas({padding: 0.5, pane: 'network'});
  private districtLayer: LeafletGeoJSON | null = null;
  private stationLayer = layerGroup();
  private baseLineLayer = layerGroup();
  private highlightLayer = layerGroup();
  private listingLayer = layerGroup();
  private stations: Station[] = [];
  private lines: LineMap | null = null;
  private subscriptions = new Subscription();

  constructor(private data: DataService, private zone: NgZone) {
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
  }

  // ---------------------------------------------------------------- setup

  onMapReady(map: LeafletMap): void {
    this.map = map;
    const pane = map.createPane('network');
    pane.style.zIndex = '450';
    pane.style.display = 'none';
    // ngx-leaflet runs Leaflet outside Angular's zone; re-enter it for anything the template shows
    map.on('moveend zoomend', () => this.zone.run(() => {
      this.zoom = map.getZoom();
      this.refreshStations();
    }));

    this.subscriptions.add(
      combineLatest([this.data.districts(), this.data.features(), this.data.clusters()])
        .subscribe(([geojson, features, clusters]) => this.zone.run(() => {
          this.features = features;
          this.clusters = clusters;
          this.computeScores();
          this.districtLayer = geoJSON(geojson, <GeoJSONOptions & PathOptions>{
            renderer: svg(),  // passed through to every polygon
            style: f => this.districtStyle(f?.properties.district),
            onEachFeature: (feature, layer) => {
              const d: number = feature.properties.district;
              layer.bindTooltip(`${d}. ${feature.properties.name}`, {sticky: true});
              layer.on('click', () => this.zone.run(() => this.selectDistrict(d)));
            },
          }).addTo(map);
        })),
    );

    // Deep link: ?view=network opens the network explorer directly
    if (new URLSearchParams(window.location.search).get('view') === 'network') {
      this.zone.run(() => this.setView('network'));
    }
  }

  // ---------------------------------------------------------------- views

  setView(view: View): void {
    if (view === this.view) return;
    this.view = view;
    this.selectedStation = null;
    this.highlightLayer.clearLayers();
    this.restyleDistricts();
    if (!this.map) return;

    const networkLayers = [this.baseLineLayer, this.stationLayer, this.highlightLayer];
    if (this.showListings) networkLayers.push(this.listingLayer);
    this.map.getPane('network')!.style.display = view === 'network' ? '' : 'none';
    if (view === 'network') {
      networkLayers.forEach(l => l.addTo(this.map!));
      this.loadNetwork();
    } else {
      networkLayers.forEach(l => l.remove());
      this.listingLayer.remove();
    }
  }

  private loadNetwork(): void {
    if (this.stations.length) return;
    this.networkLoading = true;
    this.subscriptions.add(
      combineLatest([this.data.stations(), this.data.listings()]).subscribe(([stations, listings]) => {
        this.stations = stations;
        this.stationCount = stations.length;
        this.renderListings(listings);
        this.networkLoading = false;
        this.refreshStations();
        if (this.showSubway || this.showTram) this.refreshBaseLines();
      }),
    );
  }

  // ---------------------------------------------------------------- district view

  computeScores(): void {
    if (!this.features) return;
    this.scores = {};
    for (const [d, f] of Object.entries(this.features)) {
      this.scores[+d] = matchScore(f.components, this.weights);
    }
    this.ranking = Object.entries(this.scores)
      .map(([d, score]) => ({district: +d, name: this.features![d].name, score}))
      .sort((a, b) => b.score - a.score);
  }

  onWeightsChange(): void {
    this.computeScores();
    this.restyleDistricts();
  }

  share(c: ScoreComponent): number {
    return normalizeWeights(this.weights)[c];
  }

  get scoreRange(): [number, number] {
    const values = this.ranking.map(r => r.score);
    return values.length ? [Math.min(...values), Math.max(...values)] : [0, 1];
  }

  rankOf(d: number): number {
    return this.ranking.findIndex(r => r.district === d) + 1;
  }

  districtColor(d: number): string {
    if (this.colorMode === 'cluster' && this.clusters) {
      return CLUSTER_COLORS[this.clusters.districts[String(d)]] ?? '#999';
    }
    // Colour is relative to the current best/worst district so differences stay visible
    const [lo, hi] = this.scoreRange;
    return rampColor(((this.scores[d] ?? lo) - lo) / (hi - lo || 1));
  }

  private districtStyle(d: number) {
    const selected = d === this.selectedDistrict;
    if (this.view === 'network') {
      return {fillOpacity: 0, color: '#52514e', weight: 1, opacity: 0.5};
    }
    return {
      fillColor: this.districtColor(d),
      fillOpacity: 0.72,
      color: selected ? '#0b0b0b' : '#ffffff',
      weight: selected ? 3 : 1.5,
      opacity: 1,
    };
  }

  restyleDistricts(): void {
    this.districtLayer?.eachLayer(layer => {
      const d = (layer as any).feature.properties.district;
      const style = this.districtStyle(d);
      (layer as Path).setStyle(style);
      if (d === this.selectedDistrict && this.view === 'districts') (layer as Path).bringToFront();
    });
  }

  selectDistrict(d: number | null, zoomTo = false): void {
    this.selectedDistrict = d;
    this.restyleDistricts();
    if (zoomTo && d && this.map && this.districtLayer) {
      this.districtLayer.eachLayer(layer => {
        if ((layer as any).feature.properties.district === d) {
          this.map!.fitBounds((layer as LeafletGeoJSON).getBounds(), {maxZoom: 14, padding: [40, 40]});
        }
      });
    }
  }

  get selected(): DistrictFeatures | null {
    return this.selectedDistrict && this.features ? this.features[String(this.selectedDistrict)] : null;
  }

  clusterOf(d: number) {
    const id = this.clusters?.districts[String(d)];
    return id === undefined ? null : this.clusters!.clusters[id];
  }

  modesOf(f: DistrictFeatures): TransportMode[] {
    return MODE_ORDER.filter(m => f.lines[m]?.length);
  }

  // ---------------------------------------------------------------- network view

  /** Draw stations in layers: hubs at every zoom, all stations only when zoomed in, only inside the viewport. */
  refreshStations(): void {
    this.stationLayer.clearLayers();
    this.visibleStationCount = 0;
    if (!this.map || this.view !== 'network' || !this.showStations) return;

    const showAll = this.map.getZoom() >= ALL_STATIONS_ZOOM;
    const bounds = this.map.getBounds().pad(0.15);
    for (const s of this.stations) {
      if (!showAll && !s.major) continue;
      if (!bounds.contains([s.lat, s.lon])) continue;
      const marker = circleMarker([s.lat, s.lon], {
        renderer: this.renderer,
        radius: s.major ? 5 : 3.5,
        color: '#ffffff',
        weight: 1.5,
        fillColor: s.major ? '#0d366b' : '#3987e5',
        fillOpacity: 1,
      });
      marker.bindTooltip(s.name, {direction: 'top', offset: [0, -4]});
      marker.on('click', () => this.zone.run(() => this.selectStation(s)));
      marker.addTo(this.stationLayer);
      this.visibleStationCount++;
    }
  }

  /** Base network: U-Bahn (and optionally tram) lines – needs lines.json, fetched on first use. */
  refreshBaseLines(): void {
    this.baseLineLayer.clearLayers();
    if (!this.showSubway && !this.showTram) return;
    this.withLines(lines => {
      this.baseLineLayer.clearLayers();
      for (const [name, line] of Object.entries(lines)) {
        const wanted = (line.mode === 'subway' && this.showSubway) || (line.mode === 'tram' && this.showTram);
        if (!wanted) continue;
        polyline(line.paths, {
          renderer: this.renderer,
          color: lineColor(name, line.mode),
          weight: line.mode === 'subway' ? 4 : 2,
          opacity: line.mode === 'subway' ? 0.9 : 0.6,
          interactive: false,
        }).addTo(this.baseLineLayer);
      }
    });
  }

  selectStation(s: Station | null): void {
    this.selectedStation = s;
    this.highlightLayer.clearLayers();
    if (!s) return;
    this.withLines(lines => {
      if (this.selectedStation !== s) return;
      for (const name of s.lines) {
        const line = lines[name];
        if (!line) continue;
        polyline(line.paths, {
          renderer: this.renderer, color: lineColor(name, line.mode), weight: 4, opacity: 0.85, interactive: false,
        }).addTo(this.highlightLayer);
      }
      circleMarker([s.lat, s.lon], {
        renderer: this.renderer, radius: 8, color: '#0b0b0b', weight: 3, fillColor: '#ffffff', fillOpacity: 1,
        interactive: false,
      }).addTo(this.highlightLayer);
    });
  }

  stationLinesByMode(s: Station): { mode: TransportMode; lines: string[] }[] {
    return MODE_ORDER
      .map(mode => ({mode, lines: s.lines.filter(l => this.lineModes[l] === mode)}))
      .filter(group => group.lines.length > 0);
  }

  lineChipColor(line: string): string {
    const mode = this.lineModes[line];
    return mode ? lineColor(line, mode) : '#52514e';
  }

  toggleListings(): void {
    if (!this.map || this.view !== 'network') return;
    if (this.showListings) this.listingLayer.addTo(this.map);
    else this.listingLayer.remove();
  }

  private withLines(fn: (lines: LineMap) => void): void {
    if (this.lines) {
      fn(this.lines);
      return;
    }
    this.subscriptions.add(this.data.lines().subscribe(lines => this.zone.run(() => {
      this.lines = lines;
      this.lineModes = Object.fromEntries(Object.entries(lines).map(([name, l]) => [name, l.mode]));
      fn(lines);
    })));
  }

  private renderListings(listings: Listing[]): void {
    const eur = new Intl.NumberFormat('de-AT', {style: 'currency', currency: 'EUR', maximumFractionDigits: 0});
    for (const l of listings) {
      circleMarker([l.lat, l.lon], {
        renderer: this.renderer, radius: 5, color: '#ffffff', weight: 1.5, fillColor: LISTING_COLOR, fillOpacity: 1,
      })
        .bindTooltip(`${eur.format(l.rent)} / month · ${l.size} m² · ${l.rent_per_sqm.toFixed(2)} €/m²<br>${l.type}`)
        .addTo(this.listingLayer);
    }
  }

  fitVienna(): void {
    this.map?.fitBounds(latLngBounds([48.118, 16.183], [48.323, 16.578]));
  }
}
