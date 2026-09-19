"""Entraînement offline du classifieur de TYPE d'anomalie (go-around, holding
pattern, emergency descent) — un projet ML classique distinct
("ML classiques sur données de vol/classification model", notebooks
01_dataset_construction.ipynb + 02_modelling.ipynb), reconstruit ici à
l'identique pour être exploité par flightwiser.

Ce script rejoue EXACTEMENT la partie multiclasse de 02_modelling.ipynb
(mêmes colonnes, même split train/test, même grille d'hyperparamètres, mêmes
graines aléatoires partout) — le modèle original n'a jamais été persisté
(entraîné en mémoire dans le notebook), donc la seule façon de l'obtenir est
de refaire tourner la même recette. Le binaire "normal vs atypique" du
notebook original n'est PAS reconstruit ici : flightwiser a déjà son propre
détecteur d'anomalie (models/anomalie.py, gaussienne par catégorie) — ce
classifieur n'intervient qu'en aval, pour NOMMER le type d'anomalie d'un vol
déjà flagué, pas pour décider s'il l'est.

Complémentaire à models/anomalie.py, pas un remplacement : entraîné
uniquement sur des anomalies injectées SYNTHÉTIQUEMENT (rebonds d'altitude,
circuits d'attente, descentes à taux constant — cf. le README du projet
source) — jamais sur un vrai cas réel. Un score "normal" en sortie sur un
vol que flightwiser a lui-même flagué anormal ne veut donc pas dire que rien
n'est prédit : ça veut dire que l'anomalie ne correspond à aucun des trois
patterns appris ici, ce qui reste une information honnête à afficher telle
quelle plutôt que de forcer une étiquette.

Usage : python scripts/train_anomaly_type_classifier.py
Vérification : comparer les métriques imprimées à celles publiées dans le
README du projet source (accuracy 0.984, F1 par classe 1.000/0.991/0.984/
0.957) — un écart notable signalerait une reconstruction infidèle plutôt
qu'une simple variance d'entraînement.
"""

import json
import sys
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
    print(f"Dataset : {df.shape[0]} vols, répartition anomaly_type :")
    print(df["anomaly_type"].value_counts())

    X = df[FEATURE_COLS].to_numpy()
    y_binary = df["label"].to_numpy()  # sert uniquement au split stratifié, cf. notebook original
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
    print("\nGridSearchCV (multiclasse)...")
    grid_search.fit(X_train, y_train)
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur F1 (CV)     : {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)

    print("\nRapport de classification (à comparer au README du projet source) :")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    f1_per_class = f1_score(y_test, y_pred, average=None, labels=LABELS)
    print("F1 par classe :")
    for label, score in zip(LABELS, f1_per_class):
        print(f"  {label:<20} : {score:.4f}")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "labels": LABELS}, ARTIFACT_PATH)
    print(f"\nModèle écrit dans {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
