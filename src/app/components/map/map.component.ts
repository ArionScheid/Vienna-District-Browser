import {Component, Input, OnDestroy, OnInit} from '@angular/core';
import {
  divIcon,
  geoJSON,
  latLng,
  layerGroup,
  LayerGroup,
  GeoJSON as LeafletGeoJSON,
  LeafletEvent,
  Map,
  marker,
  point,
  polyline,
  tileLayer,
} from "leaflet";
import {HttpClient} from "@angular/common/http";
// @ts-ignore
import * as Papa from "papaparse";
import {
  concatMap,
  from,
  groupBy,
  map,
  mergeMap,
  Observable,
  of,
  Subscription,
  toArray,
  zip
} from "rxjs";
import {FlatInfo} from "../../types/flat-info.type";
import {Stop} from "../../types/stop.type";
import {ShapeCsvType, ShapeType} from "../../types/shape.type";
import {StopTime, TripsPerStop} from "../../types/stop-time.type";
import {Trip} from "../../types/trip.type";
import {ClusterData, DistrictFeatures, UserPreference} from "../../types/district.type";

const DISTRICT_NAMES: { [key: number]: string } = {
  1: 'Innere Stadt', 2: 'Leopoldstadt', 3: 'Landstraße', 4: 'Wieden',
  5: 'Margareten', 6: 'Mariahilf', 7: 'Neubau', 8: 'Josefstadt',
  9: 'Alsergrund', 10: 'Favoriten', 11: 'Simmering', 12: 'Meidling',
  13: 'Hietzing', 14: 'Penzing', 15: 'Rudolfsheim-Fünfhaus',
  16: 'Ottakring', 17: 'Hernals', 18: 'Währing', 19: 'Döbling',
  20: 'Brigittenau', 21: 'Floridsdorf', 22: 'Donaustadt', 23: 'Liesing',
};

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css']
})
export class MapComponent implements OnInit, OnDestroy {

  @Input() shoudLoadData: boolean = true;

  layersControl = {
    baseLayers: {},
    overlays: {}
  }

  options = {
    layers: [
      tileLayer('https://maps.wien.gv.at/basemap/bmaphidpi/normal/google3857/{z}/{y}/{x}.jpeg', {maxZoom: 19}),
    ],
    zoom: 13,
    center: latLng(48.210033, 16.363449)
  };

  tripsPerStop$!: Observable<{ [key: string]: string[] }>;
  shapesPerTrip$!: Observable<{ [key: string]: string }>;
  shapes$!: Observable<{ [key: string]: ShapeType[] }>;

  tripsPerStop: { [key: string]: string[] } = {};
  shapesPerTrip: { [key: string]: string } = {};
  shapes: { [key: string]: ShapeType[]; } = {};

  tripsPerStopSubscription!: Subscription;
  shapesPerTripSubscription!: Subscription;
  shapesSubscription!: Subscription;
  stopsSubscription: Subscription | undefined;
  flatSubscription: Subscription | undefined;

  stopsLayerGroup: LayerGroup = layerGroup();
  housingLayerGroup: LayerGroup = layerGroup();
  pLayerGroup: LayerGroup = layerGroup();

  stopsLoaded = false;
  stopTimesLoaded = false;
  tripsLoaded = false;
  shapesLoaded = false;
  flatsLoaded = false;

  // District overlay
  showDistrictOverlay = false;
  districtLayer: LeafletGeoJSON | null = null;
  districtGeoJson: any = null;
  districtFeatures: { [key: string]: DistrictFeatures } = {};
  clusterData: ClusterData | null = null;
  selectedDistrict: number | null = null;
  districtScores: { [key: number]: number } = {};
  districtDataLoaded = false;
  colorMode: 'cluster' | 'score' = 'score';

  preferences: UserPreference = {
    affordability: 0.33,
    accessibility: 0.34,
    transit: 0.33,
  };

  private mapInstance: Map | null = null;

  constructor(private http: HttpClient) {
  }

  ngOnInit(): void {
    if (this.shoudLoadData) {
      const stopTimes$ = this.http
        .get('/assets/data/wienerlinien/stop_times.txt', {responseType: 'text'})
        .pipe(
          map<string, StopTime[]>(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data)
        );

      this.tripsPerStop$ = stopTimes$
        .pipe(
          concatMap((arr) => from(arr)),
          map<StopTime, TripsPerStop>((e) => ({
            meta_stop_id: e.stop_id.substring(0, this.nthIndex(e.stop_id, ':', 3)),
            trip_id: e.trip_id,
          })),
          groupBy(data => data.meta_stop_id),
          mergeMap((group) => zip(of(group.key), group.pipe(map(el => el.trip_id), toArray()))),
          toArray(),
          map(data => data.reduce((obj: { [key: string]: string[] }, item) => {
            obj[item[0]] = item[1];
            return obj
          }, {})),
        );

      this.tripsPerStop = {};
      this.tripsPerStopSubscription = this.tripsPerStop$.subscribe((data) => {
        this.tripsPerStop = data;
        this.stopTimesLoaded = true;
      });

      this.shapesPerTrip = {};

      const trips$ = this.http
        .get('/assets/data/wienerlinien/trips.txt', {responseType: 'text'})
        .pipe(
          map<string, Trip[]>(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data)
        );

      this.shapesPerTrip$ = trips$.pipe(
        map(data => data.reduce((obj: { [key: string]: string }, item) => {
          obj[item.trip_id] = item.shape_id;
          return obj
        }, {}))
      )

      this.shapesPerTripSubscription = this.shapesPerTrip$.subscribe((data) => {
        this.shapesPerTrip = data;
        this.tripsLoaded = true;
      })

      this.shapes = {};
      this.shapes$ = this.http
        .get('/assets/data/wienerlinien/shapes.txt', {responseType: 'text'})
        .pipe(
          map<string, ShapeCsvType[]>(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data),
          concatMap((arr) => from(arr)),
          map<ShapeCsvType, ShapeType>((data) => ({
            shape_id: data.shape_id,
            shape_pt_lat: parseFloat(data.shape_pt_lat),
            shape_pt_lon: parseFloat(data.shape_pt_lon),
            shape_pt_sequence: parseFloat(data.shape_pt_sequence)
          })),
          groupBy((data) => data.shape_id),
          mergeMap((group) => zip(of(group.key), group.pipe(toArray()))),
          toArray(),
          map(data => data.reduce((obj: { [key: string]: ShapeType[] }, item) => {
            obj[item[0]] = item[1].sort((a, b) => a.shape_pt_sequence - b.shape_pt_sequence);
            return obj
          }, {})),
        )

      this.shapesSubscription = this.shapes$.subscribe((data) => {
        this.shapes = data;
        this.shapesLoaded = true;
      })
    }
  }

  ngOnDestroy() {
    if (this.stopsSubscription != undefined) {
      this.stopsSubscription.unsubscribe();
    }
    if (this.flatSubscription != undefined) {
      this.flatSubscription.unsubscribe();
    }

    this.shapesSubscription.unsubscribe();
    this.tripsPerStopSubscription.unsubscribe();
    this.shapesPerTripSubscription.unsubscribe();
  }

  nthIndex(str: string, pat: string, n: number) {
    let L = str.length, i = -1;
    while (n-- && i++ < L) {
      i = str.indexOf(pat, i);
      if (i < 0) break;
    }
    return i;
  }


  onMapReady(mapElement: Map) {
    this.mapInstance = mapElement;

    if (this.shoudLoadData) {
      this.stopsLayerGroup.addTo(mapElement);
      this.housingLayerGroup.addTo(mapElement);
      this.pLayerGroup.addTo(mapElement);

      const iconType1 = (label: string) => divIcon({
        iconSize: point(8, 8),
        className: "hover:!z-[1000] icon-stationen group relative flext items-center justify-center",
        html: `<div class="hidden group-hover:block absolute  whitespace-nowrap mt-4 -left-1/2 -ml-[50%] font-bold [text-shadow:0_4px_8px_rgba(0,0,0,0.12)]">${label}</div>`
      });

      const iconType2 = (label: string) => divIcon({
        iconSize: point(16, 16),
        className: "hover:!z-[1000] icon-unterkunft group relative flext items-center justify-center",
        html: `<div class="hidden group-hover:block absolute  whitespace-nowrap mt-4 -left-1/2 -ml-[50%] font-bold [text-shadow:0_4px_8px_rgba(0,0,0,0.12)]">${label}</div>`
      });

      const stops$: Observable<Stop[]> = this.http
        .get('/assets/data/wienerlinien/stops.txt', {responseType: 'text'})
        .pipe(
          map(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data)
        );

      this.stopsSubscription = stops$.subscribe((data) => {
        data.forEach((entry) => {
          const location = latLng(parseFloat(entry.stop_lat), parseFloat(entry.stop_lon));
          marker(location, {
            title: entry.stop_id,
            icon: iconType1(entry.stop_name)
          }).addTo(this.stopsLayerGroup).on('click', (e) => this.handleMarkerClick(e))
        })
        this.stopsLoaded = true;
      })

      const flats$ = this.http
        .get<FlatInfo[]>('/assets/data/flat_info.json', {responseType: 'json'});

      const formatter = new Intl.NumberFormat('de-DE', {style: 'currency', currency: 'EUR'});

      this.flatSubscription = flats$.subscribe((data) => {
        data.forEach((entry) => {
          const location = latLng(parseFloat(entry.LATITUDE), parseFloat(entry.LONGITUDE));
          marker(location, {
            icon: iconType2(formatter.format(parseFloat(entry.PRICE)))
          }).addTo(this.housingLayerGroup).on('click', (e) => {
            console.log(entry)
          })
        })
        this.flatsLoaded = true;
      })

      this.loadDistrictData();
    }
  }

  handleMarkerClick(e: LeafletEvent) {
    this.pLayerGroup.clearLayers();

    const title = e.target.options.title;
    const shortTitle = title.substring(0, this.nthIndex(title, ':', 3));

    const tripNames = this.tripsPerStop[shortTitle];

    if (tripNames == null) {
      console.log("Trips not found, have you loaded only a subset of the data?")
      return;
    }

    const shapeNames = tripNames.map((tripName: string) => this.shapesPerTrip[tripName]).filter((thing, i: number, arr) => {
      const thingName = arr.find((t) => t == thing);
      return thingName != undefined && arr.indexOf(thingName) === i;
    });

    const lines = shapeNames.map((shapeName: string) => this.shapes[shapeName]);

    lines.forEach((line) => {
      const pointList = line.map((linePoints: { shape_pt_lat: number; shape_pt_lon: number; }) => latLng(linePoints.shape_pt_lat, linePoints.shape_pt_lon));
      const el = polyline(pointList, {color: 'red'});
      this.pLayerGroup.addLayer(el);
    });
  }

  // --- District overlay ---

  private loadDistrictData(): void {
    this.http.get('/assets/data/districts.geojson').subscribe({
      next: (geojson) => {
        this.districtGeoJson = geojson;
        this.loadDistrictFeatures();
      },
      error: () => console.log('districts.geojson not found — district overlay disabled')
    });
  }

  private loadDistrictFeatures(): void {
    this.http.get<{ [key: string]: DistrictFeatures }>('/assets/data/district_features.json').subscribe({
      next: (features) => {
        this.districtFeatures = features;
        this.loadClusterData();
      },
      error: () => {
        this.districtDataLoaded = true;
      }
    });
  }

  private loadClusterData(): void {
    this.http.get<ClusterData>('/assets/data/clusters.json').subscribe({
      next: (data) => {
        this.clusterData = data;
        this.districtDataLoaded = true;
        this.computeDistrictScores();
      },
      error: () => {
        this.districtDataLoaded = true;
        this.computeDistrictScores();
      }
    });
  }

  toggleOverlay(): void {
    if (!this.mapInstance || !this.districtGeoJson) return;

    this.showDistrictOverlay = !this.showDistrictOverlay;
    this.selectedDistrict = null;

    if (this.showDistrictOverlay) {
      this.mapInstance.removeLayer(this.stopsLayerGroup);
      this.mapInstance.removeLayer(this.housingLayerGroup);
      this.mapInstance.removeLayer(this.pLayerGroup);
      this.renderDistrictOverlay();
    } else {
      if (this.districtLayer) {
        this.districtLayer.remove();
        this.districtLayer = null;
      }
      this.stopsLayerGroup.addTo(this.mapInstance);
      this.housingLayerGroup.addTo(this.mapInstance);
    }
  }

  private renderDistrictOverlay(): void {
    if (!this.districtGeoJson || !this.mapInstance) return;

    if (this.districtLayer) {
      this.districtLayer.remove();
    }

    this.districtLayer = geoJSON(this.districtGeoJson, {
      style: (feature) => {
        const dNum = this.getDistrictNumber(feature);
        return {
          fillColor: this.getDistrictColor(dNum),
          weight: 2,
          opacity: 1,
          color: '#ffffff',
          fillOpacity: 0.65,
        };
      },
      onEachFeature: (feature, layer) => {
        const dNum = this.getDistrictNumber(feature);
        const name = DISTRICT_NAMES[dNum] || `District ${dNum}`;
        layer.bindTooltip(`${dNum}. ${name}`, {sticky: true});
        layer.on('click', () => this.onDistrictClick(dNum));
      }
    });

    this.districtLayer.addTo(this.mapInstance);
  }

  private getDistrictNumber(feature: any): number {
    const props = feature?.properties || {};
    for (const key of ['BEZNR', 'BEZ', 'DISTRICT_CODE', 'district', 'bez', 'OBJECTID']) {
      if (props[key] !== undefined && props[key] !== null) {
        const n = parseInt(props[key], 10);
        if (!isNaN(n) && n >= 1 && n <= 23) return n;
      }
    }
    for (const val of Object.values(props)) {
      const n = parseInt(val as string, 10);
      if (!isNaN(n) && n >= 1 && n <= 23) return n;
    }
    return 0;
  }

  getDistrictColor(districtNum: number): string {
    if (this.colorMode === 'cluster' && this.clusterData) {
      const info = this.clusterData.districts[String(districtNum)];
      return info?.color || '#999999';
    }

    const score = this.districtScores[districtNum] ?? 0.5;
    const scores = Object.values(this.districtScores);
    if (scores.length === 0) return '#999999';

    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min || 1;
    const norm = (score - min) / range;

    const hue = norm * 120;
    return `hsl(${hue}, 70%, 45%)`;
  }

  computeDistrictScores(): void {
    const keys = Object.keys(this.districtFeatures);
    if (keys.length === 0) return;

    const prices = keys.map(d => this.districtFeatures[d].avg_price_per_sqm);
    const access = keys.map(d => this.districtFeatures[d].accessibility_score);
    const transit = keys.map(d => this.districtFeatures[d].transit_density);

    const normalize = (vals: number[]): number[] => {
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const range = max - min || 1;
      return vals.map(v => (v - min) / range);
    };

    const normPrices = normalize(prices);
    const normAccess = normalize(access);
    const normTransit = normalize(transit);

    keys.forEach((d, i) => {
      this.districtScores[parseInt(d)] =
        this.preferences.affordability * (1 - normPrices[i]) +
        this.preferences.accessibility * normAccess[i] +
        this.preferences.transit * normTransit[i];
    });
  }

  onPreferenceChange(): void {
    this.computeDistrictScores();
    if (this.showDistrictOverlay && this.colorMode === 'score') {
      this.renderDistrictOverlay();
    }
  }

  onColorModeChange(): void {
    if (this.showDistrictOverlay) {
      this.renderDistrictOverlay();
    }
  }

  onDistrictClick(districtNum: number): void {
    this.selectedDistrict = districtNum;
  }

  closeDistrictSidebar(): void {
    this.selectedDistrict = null;
  }

  getDistrictName(d: number): string {
    return DISTRICT_NAMES[d] || `District ${d}`;
  }

  getSelectedFeatures(): DistrictFeatures | null {
    if (!this.selectedDistrict) return null;
    return this.districtFeatures[String(this.selectedDistrict)] || null;
  }

  getSelectedClusterColor(): string {
    if (!this.selectedDistrict || !this.clusterData) return '#999';
    return this.clusterData.districts[String(this.selectedDistrict)]?.color || '#999';
  }

  getSelectedClusterLabel(): string {
    if (!this.selectedDistrict || !this.clusterData) return '—';
    return this.clusterData.districts[String(this.selectedDistrict)]?.label || '—';
  }

  getSelectedScore(): number {
    if (!this.selectedDistrict) return 0;
    return Math.round((this.districtScores[this.selectedDistrict] ?? 0) * 100);
  }

  formatCurrency(value: number): string {
    return new Intl.NumberFormat('de-DE', {style: 'currency', currency: 'EUR'}).format(value);
  }
}
