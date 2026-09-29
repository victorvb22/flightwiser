"""Constants shared across the three per-flight feature modules
(_anomalie_features.py, _directness_features.py, _ecart_features.py) —
previously redefined identically (same value, same purpose: the minimum
number of usable waypoints for a trajectory-derived feature to mean
anything) in all three, risking drifting out of sync if one were ever tuned
without the others."""

MIN_POINTS = 5
