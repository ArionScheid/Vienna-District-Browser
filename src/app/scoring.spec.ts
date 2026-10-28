import {matchScore, normalizeWeights} from './scoring';

describe('scoring', () => {
  const best = {affordability: 1, accessibility: 1, transit: 1};

  it('never exceeds 100 % even when every slider is at its maximum', () => {
    expect(matchScore(best, {affordability: 10, accessibility: 10, transit: 10})).toBeCloseTo(1);
  });

  it('normalises weights to shares that sum to 1', () => {
    const w = normalizeWeights({affordability: 2, accessibility: 1, transit: 1});
    expect(w.affordability).toBeCloseTo(0.5);
    expect(w.affordability + w.accessibility + w.transit).toBeCloseTo(1);
  });

  it('treats all-zero weights as equal weights', () => {
    const score = matchScore({affordability: 0, accessibility: 0.6, transit: 0.9}, {affordability: 0, accessibility: 0, transit: 0});
    expect(score).toBeCloseTo(0.5);
  });

  it('imputes missing rent data as average instead of best or worst', () => {
    const score = matchScore({affordability: null, accessibility: 1, transit: 1}, {affordability: 1, accessibility: 0, transit: 0});
    expect(score).toBeCloseTo(0.5);
  });

  it('stays within [0, 1] for random inputs', () => {
    for (let i = 0; i < 200; i++) {
      const s = matchScore(
        {affordability: Math.random() < 0.2 ? null : Math.random(), accessibility: Math.random(), transit: Math.random()},
        {affordability: Math.random() * 10, accessibility: Math.random() * 10, transit: Math.random() * 10},
      );
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});
