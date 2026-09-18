"""Score de trajet direct (secondaire au modèle d'anomalie et au modèle
d'écart — cf. _directness_features.py pour le rationnel) : rang percentile
empirique du ratio distance grand-cercle / distance parcourue, par
catégorie d'appareil (services.aircraft_category), contre la distribution
d'entraînement de sa catégorie.

Pas de gaussienne ici, contrairement à models/anomalie.py : une seule
feature scalaire déjà bornée (≤ 1) n'a rien à gagner de la machinerie
multi-features (moyenne/écart-type/log-vraisemblance) construite pour
combiner sept dimensions — un rang percentile empirique direct est plus
simple et ne suppose aucune forme de distribution particulière (le ratio
est naturellement asymétrique, plafonné à droite, ce qu'une gaussienne
représenterait mal).

Paramètres appris hors ligne par scripts/train_directness.py et persistés
dans artifacts/directness_params.json.

Input  : trajectoire réelle (liste de points avec lat/lon) et le typecode
          de l'appareil (pour choisir la catégorie).
Output : {"score": float, "ratio": float} — score = rang percentile (0-1,
          plus bas = trajet moins direct que la normale pour sa catégorie),
          ratio = la valeur brute (0-1, 1 = parfaitement direct) pour
          affichage. None si le signal n'est pas exploitable pour ce vol
          (cf. _directness_features.extract_route_directness).
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

    model = _models[categorize(typecode)]
    score = float(np.interp(ratio, model.breakpoints, model.percentiles) / 100.0)
    return {"score": score, "ratio": ratio}
