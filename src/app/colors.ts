import {TransportMode} from './types/data.types';

/** Sequential single-hue ramp (light → dark blue) for the match-score choropleth. */
export const SCORE_RAMP = ['#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'];

/** Categorical slots for transit zones, in fixed order (first three are colour-blind safe as a set). */
export const CLUSTER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'];

export const LISTING_COLOR = '#eb6834';

/** Official U-Bahn line colours; other lines are coloured by mode. */
const SUBWAY_COLORS: Record<string, string> = {
  U1: '#e3000f', U2: '#a862a4', U3: '#ef7c00', U4: '#00963f', U5: '#008f95', U6: '#9d6930',
};

export const MODE_COLORS: Record<TransportMode, string> = {
  'subway': '#52514e',
  'rail': '#0073a8',
  'tram': '#c4161c',
  'bus': '#1c5cab',
  'night bus': '#3f3d9b',
};

export const MODE_LABELS: Record<TransportMode, string> = {
  'subway': 'U-Bahn',
  'rail': 'Badner Bahn',
  'tram': 'Tram',
  'bus': 'Bus',
  'night bus': 'Night bus',
};

export const MODE_ORDER: TransportMode[] = ['subway', 'rail', 'tram', 'bus', 'night bus'];

export function lineColor(line: string, mode: TransportMode): string {
  return SUBWAY_COLORS[line] ?? MODE_COLORS[mode];
}

/** Interpolate along SCORE_RAMP for t in [0, 1]. */
export function rampColor(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (SCORE_RAMP.length - 1);
  const i = Math.min(Math.floor(x), SCORE_RAMP.length - 2);
  const f = x - i;
  const a = hexToRgb(SCORE_RAMP[i]);
  const b = hexToRgb(SCORE_RAMP[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`;
}

function hexToRgb(hex: string): number[] {
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
}
