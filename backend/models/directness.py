"""Route directness score (secondary to the anomaly model and the deviation
model — cf. _directness_features.py for the rationale): empirical percentile
rank of the great-circle-distance / distance-flown ratio, per aircraft
category (avion_ligne / jet_affaire / petit_avion / helicoptere, cf.
services.aircraft_category — checked directly on real data: jet_affaire and
petit_avion have meaningfully different ratio distributions, cf.
aircraft_category.py's docstring), against its category's own training
distribution.

No Gaussian here, unlike models/anomalie.py: a single, already-bounded (≤ 1)
scalar feature has nothing to gain from the multi-feature machinery
(mean/std/log-likelihood) built to combine seven dimensions — a direct
empirical percentile rank is simpler and assumes no particular distribution
shape (the ratio is naturally skewed, capped on the right, which a Gaussian
would represent poorly).

Parameters learned offline by scripts/train_directness.py and persisted in
artifacts/directness_params.json.

Input:  real trajectory (list of points with lat/lon) and the aircraft's
        typecode (to pick the category).
Output: {"score": float, "ratio": float} — score = percentile rank (0-1,
        lower = a less direct route than normal for its category), ratio =
        the raw value (0-1, 1 = perfectly direct) for display. None if the
        signal isn't usable for this flight (cf.
        _directness_features.extract_route_directness).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from services.aircraft_category import CATEGORIES, categorize

from ._directness_features import extract_route_directness

_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "directness_params.json"


class _CategoryModel:
    def __init__(self, params: dict):
        self.breakpoints = np.array(params["percentile_breakpoints"])
        self.percentiles = np.linspace(0, 100, len(self.breakpoints))


_raw_params = json.loads(_ARTIFACT_PATH.read_text())
_models = {name: _CategoryModel(_raw_params["categories"][name]) for name in CATEGORIES}


def compute(trajectoire: list[dict], typecode: str) -> dict[str, Any] | None:
    ratio = extract_route_directness(trajectoire)
    if ratio is None:
        return None

    category = categorize(typecode)
    if category is None:
        return None
    model = _models[category]
    score = float(np.interp(ratio, model.breakpoints, model.percentiles) / 100.0)
    return {"score": score, "ratio": ratio}
