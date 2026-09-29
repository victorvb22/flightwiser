"""Shared geo math — previously reimplemented identically in
services/trajectory_cleaning.py, models/_ecart_features.py, and
models/_directness_features.py (models/anomaly_type.py already imported its
copy from _ecart_features.py rather than reimplementing it again)."""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points (scalars or numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
