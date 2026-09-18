"""Frontière avion de ligne / petit avion, partagée entre les modèles qui
calibrent une distribution par catégorie plutôt qu'une seule sur tout le
parc (cf. models/anomalie.py, models/directness.py) — un jet d'affaires,
un avion de tourisme ou un hélicoptère n'a ni les mêmes vitesses/altitudes/
taux de montée qu'un avion de ligne, ni la même notion de "trajet direct"
(vol d'entraînement en circuit, survol touristique...).

La frontière retenue est "OpenAP reconnaît ce typecode" : sa base est
justement limitée aux avions de ligne (et quelques jets régionaux/
d'affaires), donc c'est une frontière déjà disponible et cohérente avec le
reste du projet (models/_ecart_features.resolve_typecode s'appuie sur la
même base), plutôt qu'une nouvelle liste à maintenir à la main.
"""

from openap.gen import FlightGenerator

CATEGORIES = ["avion_ligne", "petit_avion"]


def categorize(typecode: str) -> str:
    """"avion_ligne" si OpenAP reconnaît ce typecode, "petit_avion" sinon."""
    try:
        FlightGenerator(ac=typecode.lower(), use_synonym=True)
        return "avion_ligne"
    except ValueError:
        return "petit_avion"
