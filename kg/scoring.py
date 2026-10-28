"""The district match score – one definition shared by the GNN labels and the web app.

Each district has three components in [0, 1] (higher = better):

* ``affordability`` – 1 − min-max-scaled median rent per m² (``None`` if unknown)
* ``accessibility`` – direct daytime lines to the city centre, scaled to [0, 1]
* ``transit``       – min-max-scaled station density

The score is a weighted *average* of the components: weights are normalised to
sum to 1, so the score always lies in [0, 1] (shown as 0–100 %).  A missing
component is imputed as ``NEUTRAL`` (0.5, "average") – it must neither reward
nor punish a district just because the housing sample has no listings there.

The TypeScript twin of this function lives in ``src/app/scoring.ts``.
"""

COMPONENTS = ("affordability", "accessibility", "transit")
NEUTRAL = 0.5


def match_score(components: dict[str, float | None], weights: dict[str, float]) -> float:
    w = {c: max(0.0, float(weights.get(c, 0.0))) for c in COMPONENTS}
    total = sum(w.values())
    if total == 0:  # all sliders at zero → treat as equal weights
        w, total = {c: 1.0 for c in COMPONENTS}, float(len(COMPONENTS))
    value = {c: NEUTRAL if components.get(c) is None else components[c] for c in COMPONENTS}
    return sum(w[c] * value[c] for c in COMPONENTS) / total


def min_max(values: dict[int, float | None], invert: bool = False) -> dict[int, float | None]:
    """Scale the known values to [0, 1]; ``None`` stays ``None``."""
    known = [v for v in values.values() if v is not None]
    if not known:
        return {k: None for k in values}
    lo, hi = min(known), max(known)
    span = hi - lo or 1.0
    scaled = {k: None if v is None else (v - lo) / span for k, v in values.items()}
    if invert:
        scaled = {k: None if v is None else 1.0 - v for k, v in scaled.items()}
    return scaled
