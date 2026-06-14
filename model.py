"""
model.py — Entraînement, évaluation et sérialisation du modèle IA
Early Warning System
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── Chemins ────────────────────────────────────────────────────────────────
TRAIN_FILE  = "data.csv"
MODEL_PATH  = "model.pkl"

# ── Colonnes utilisées par le modèle ───────────────────────────────────────
CRITERES_IA = [
    "attendance_pct",
    "absence_hours",
    "homework_pct",
    "study_hours_per_week",
    "midterm_score",
]


# ── Fonction principale ─────────────────────────────────────────────────────
def train_and_save(train_file: str = TRAIN_FILE, model_path: str = MODEL_PATH):
    """
    Charge data.csv, entraîne un LogisticRegression, affiche les métriques
    et sérialise (scaler + model) dans model_path.

    Retourne un dict avec les métriques clés.
    """
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"❌ Fichier d'entraînement introuvable : `{train_file}`"
        )

    # ── 1. Chargement et nettoyage ─────────────────────────────────────────
    df = pd.read_csv(train_file, sep=";")
    df.columns = df.columns.str.strip()

    X = df[CRITERES_IA]
    y = df["pass"].astype(str).str.extract(r"(\d)")[0].astype(int)

    # ── 2. Split stratifié 80/20 ───────────────────────────────────────────
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Standardisation ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)

    # ── 4. Cross-validation K=5 ────────────────────────────────────────────
    model_cv = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000, random_state=42
    )
    cv_results = cross_validate(
        model_cv,
        scaler.fit_transform(X),
        y,
        cv=5,
        scoring=["neg_log_loss", "accuracy"],
        return_train_score=True,
    )

    j_train_cv = -cv_results["train_neg_log_loss"].mean()
    j_cv       = -cv_results["test_neg_log_loss"].mean()
    accuracy_cv = cv_results["test_accuracy"].mean()
    error_pct   = (1.0 - accuracy_cv) * 100

    # ── 5. J_global (modèle entraîné sur tout le jeu) ─────────────────────
    X_full_scaled = scaler.fit_transform(X)
    model_full = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000, random_state=42
    )
    model_full.fit(X_full_scaled, y)

    y_proba = np.clip(model_full.predict_proba(X_full_scaled)[:, 1], 1e-15, 1 - 1e-15)
    m = len(y)
    j_global = -(1 / m) * np.sum(
        y * np.log(y_proba) + (1 - y) * np.log(1 - y_proba)
    )

    # ── 6. Rapport de classification (jeu de validation externe) ──────────
    model_test = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000, random_state=42
    )
    model_test.fit(X_train_scaled, y_train)
    y_val_pred = model_test.predict(X_val_scaled)

    raw_report = classification_report(
        y_val, y_val_pred, target_names=["FAIL (0)", "PASS (1)"]
    )
    formatted_report = "\n".join(
        "    " + line
        for line in raw_report.split("\n")
        if any(x in line for x in ["precision", "FAIL (0)", "PASS (1)"])
    )

    # ── 7. Affichage ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  RAPPORT D'ÉVALUATION DU MODÈLE IA")
    print("=" * 60)
    print(f"  J_train  (CV K=5)  : {j_train_cv:.4f}")
    print(f"  J_cv     (CV K=5)  : {j_cv:.4f}")
    print(f"  Accuracy (CV K=5)  : {accuracy_cv * 100:.2f}%")
    print(f"  Erreur   (CV K=5)  : {error_pct:.2f}%")
    print(f"  J_global (full)    : {j_global:.4f}")
    print()
    print("  Classification (jeu de validation 20%) :")
    print(formatted_report)
    print("=" * 60)

    # ── 8. Sérialisation ───────────────────────────────────────────────────
    bundle = {"scaler": scaler, "model": model_full, "features": CRITERES_IA}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✅ Modèle sauvegardé → {model_path}")

    return {
        "j_train_cv": j_train_cv,
        "j_cv": j_cv,
        "accuracy_cv": accuracy_cv,
        "error_pct": error_pct,
        "j_global": j_global,
    }


def load_model(model_path: str = MODEL_PATH):
    """
    Charge et retourne le bundle (scaler, model, features) depuis le fichier PKL.
    Lève FileNotFoundError si le PKL n'existe pas encore.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Modèle PKL introuvable : `{model_path}`.\n"
            "   Lancez d'abord : python model.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── Point d'entrée (génération du PKL) ────────────────────────────────────
if __name__ == "__main__":
    train_and_save()
