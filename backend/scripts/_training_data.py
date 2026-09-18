"""Fusion dédoublonnée des deux sources d'entraînement (train_anomalie.py,
train_directness.py) — flights_clean.parquet et
flights_historical_features.parquet couvrent toutes deux le 27/06/2022, avec
un recouvrement réel et non négligeable (vérifié empiriquement : ~2200
vols de flights_clean.parquet, sur ~3100, partagent un icao24 et une fenêtre
temporelle très largement chevauchante — souvent >80% de la durée du plus
court des deux — avec un vol de flights_historical_features.parquet). Fusionner
sans dédoublonner ferait peser ces vols deux fois dans μ/σ (et dans les
percentiles de directness), les rapprochant artificiellement de la moyenne.

Un même icao24 peut légitimement voler plusieurs fois le même jour (plusieurs
rotations) : la détection ne peut donc pas se limiter à "même icao24", il
faut aussi que les fenêtres temporelles [decollage, atterrissage] se
recouvrent significativement — un simple contact aux bornes (rotation
immédiatement enchaînée) ne doit pas compter comme doublon.

En cas de doublon, la version de flights_historical_features.parquet est
gardée (source passée en dernier dans `input_parquets`) plutôt que celle de
flights_clean.parquet : c'est le pipeline le plus récent et le plus
systématique (segmentation par écart de 30 min appliquée uniformément), donc
la référence à privilégier au fil des ré-extractions futures (élargissement
géographique, nouvelles dates...).
"""

import json

import pandas as pd

# Choisi par inspection directe des cas limites (cf. docstring du module) :
# les vraies paires dupliquées se regroupent près de 100% de recouvrement,
# les rotations distinctes dos-à-dos près de 0% — 0.5 laisse une marge large
# entre les deux sans risquer de fusionner deux vols réellement différents.
MEANINGFUL_OVERLAP_FRACTION = 0.5

_REQUIRED_COLUMNS = ["icao24", "callsign", "typecode", "still_airborne", "waypoints"]


def _time_range(waypoints_json: str) -> tuple[float, float]:
    waypoints = json.loads(waypoints_json)
    timestamps = [w["timestamp"] for w in waypoints]
    return min(timestamps), max(timestamps)


def load_deduplicated_landed_flights(input_parquets: list) -> pd.DataFrame:
    """Charge, filtre aux vols atterris et fusionne `input_parquets` (dans
    l'ordre donné), en retirant d'une source antérieure tout vol qui
    recouvre significativement (cf. MEANINGFUL_OVERLAP_FRACTION) un vol du
    même icao24 dans une source postérieure. Retourne les colonnes
    icao24/callsign/typecode/still_airborne/waypoints, prêtes pour
    l'extraction de features."""
    frames = []
    for path in input_parquets:
        df = pd.read_parquet(path)
        landed = df[~df["still_airborne"]].reset_index(drop=True)
        print(f"{path.name} : {len(landed)}/{len(df)} vols atterris")
        frame = landed[_REQUIRED_COLUMNS].copy()
        ranges = frame["waypoints"].apply(_time_range)
        frame["_takeoff"] = ranges.apply(lambda t: t[0])
        frame["_landing"] = ranges.apply(lambda t: t[1])
        frames.append(frame)

    kept_frames = []
    for i, frame in enumerate(frames):
        later_frames = frames[i + 1 :]
        if not later_frames:
            kept_frames.append(frame)
            continue
        later = pd.concat(later_frames, ignore_index=True)
        later_windows_by_icao = {
            icao: group[["_takeoff", "_landing"]].to_numpy() for icao, group in later.groupby("icao24")
        }

        def is_duplicate_of_later_source(row) -> bool:
            windows = later_windows_by_icao.get(row["icao24"])
            if windows is None:
                return False
            duration = row["_landing"] - row["_takeoff"]
            if duration <= 0:
                return False
            for later_takeoff, later_landing in windows:
                overlap = min(row["_landing"], later_landing) - max(row["_takeoff"], later_takeoff)
                if overlap / duration >= MEANINGFUL_OVERLAP_FRACTION:
                    return True
            return False

        duplicate_mask = frame.apply(is_duplicate_of_later_source, axis=1)
        n_duplicates = int(duplicate_mask.sum())
        if n_duplicates:
            print(f"  -> {n_duplicates} vol(s) déjà couverts par une source plus récente, écartés (doublon)")
        kept_frames.append(frame[~duplicate_mask])

    merged = pd.concat(kept_frames, ignore_index=True).drop(columns=["_takeoff", "_landing"])
    print(f"Total vols atterris après dédoublonnage (toutes sources confondues) : {len(merged)}")
    return merged
