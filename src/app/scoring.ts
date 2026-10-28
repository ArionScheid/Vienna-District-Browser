import {Component, DistrictFeatures, Weights} from './types/data.types';

/**
 * TypeScript twin of kg/scoring.py – keep both in sync.
 *
 * The match score is a weighted *average* of the three components (each in [0, 1]):
 * weights are normalised to sum to 1, so the score always lies in [0, 1].
 * A missing component (no rent data) is imputed as NEUTRAL ("average").
 */
export const COMPONENTS: Component[] = ['affordability', 'accessibility', 'transit'];
export const NEUTRAL = 0.5;

/** Weights as shares that sum to 1 (all-zero sliders count as equal weights). */
export function normalizeWeights(weights: Weights): Weights {
  const clipped = COMPONENTS.map(c => Math.max(0, weights[c] || 0));
  const total = clipped.reduce((a, b) => a + b, 0);
  const shares = total === 0 ? clipped.map(() => 1 / COMPONENTS.length) : clipped.map(w => w / total);
  return {affordability: shares[0], accessibility: shares[1], transit: shares[2]};
}

export function matchScore(components: DistrictFeatures['components'], weights: Weights): number {
  const w = normalizeWeights(weights);
  return COMPONENTS.reduce((sum, c) => sum + w[c] * (components[c] ?? NEUTRAL), 0);
}
