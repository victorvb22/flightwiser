"""Offline training of the anomaly-TYPE classifier (go-around, holding
pattern, emergency descent) — a separate classical ML project
("ML classiques sur données de vol/classification model", notebooks
01_dataset_construction.ipynb + 02_modelling.ipynb), reconstructed here
identically so flightwiser can use it.

This script replays EXACTLY the multiclass part of 02_modelling.ipynb (same
columns, same train/test split, same hyperparameter grid, same random seeds
everywhere) — the original model was never persisted (trained in-memory in
the notebook), so the only way to get it back is to rerun the same recipe.
The original notebook's "normal vs. atypical" binary model is NOT
reconstructed here: flightwiser already has its own anomaly detector
(models/anomalie.py, a per-category Gaussian) — this classifier only steps
in downstream, to NAME the anomaly type of a flight already flagged, not to
decide whether it is one.

Complementary to models/anomalie.py, not a replacement: trained only on
SYNTHETICALLY injected anomalies (altitude rebounds, holding circuits,
constant-rate descents — cf. the source project's README) — never on a real
case. A "normal" output on a flight flightwiser itself flagged anomalous
therefore doesn't mean nothing was predicted: it means the anomaly doesn't
match any of the three patterns learned here, which remains honest
information to show as-is rather than forcing a label.

Usage: python scripts/train_anomaly_type_classifier.py
Verification: compare the printed metrics against those published in the
source project's README (accuracy 0.984, F1 per class 1.000/0.991/0.984/
0.957) — a notable gap would signal an unfaithful reconstruction rather than
plain training variance.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BACKEND_DIR.parent / "data" / "processed" / "anomaly_type_dataset.csv"
ARTIFACT_PATH = BACKEND_DIR / "models" / "artifacts" / "anomaly_type_classifier.joblib"

RANDOM_STATE = 42

FEATURE_COLS = [
    "vrate_descent_max",
    "vrate_cruise_std",
    "heading_changes_count",
    "ratio_dist_duration",
    "alt_rebound_max",
    "cruise_duration_ratio",
]

LABELS = ["emergency_descent", "go_around", "holding", "normal"]

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {df.shape[0]} flights, anomaly_type breakdown:")
    print(df["anomaly_type"].value_counts())

    X = df[FEATURE_COLS].to_numpy()
    y_binary = df["label"].to_numpy()  # used only for the stratified split, cf. the original notebook
    y_multiclass = df["anomaly_type"].to_numpy(dtype=str)

    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y_multiclass[idx_train], y_multiclass[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    print("\nGridSearchCV (multiclass)...")
    grid_search.fit(X_train, y_train)
    print(f"Best params : {grid_search.best_params_}")
    print(f"Best F1 (CV): {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)

    print("\nClassification report (compare against the source project's README):")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    f1_per_class = f1_score(y_test, y_pred, average=None, labels=LABELS)
    print("F1 per class:")
    for label, score in zip(LABELS, f1_per_class):
        print(f"  {label:<20} : {score:.4f}")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "labels": LABELS}, ARTIFACT_PATH)
    print(f"\nModel written to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
