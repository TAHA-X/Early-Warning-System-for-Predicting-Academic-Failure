"""
api.py — Routes FastAPI de l'Early Warning System
Dépend de model.py pour charger le bundle PKL (scaler + modèle).
"""

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model import load_model, CRITERES_IA

# ── Chargement du modèle au démarrage ─────────────────────────────────────
_bundle = load_model()          # {"scaler": ..., "model": ..., "features": [...]}
scaler  = _bundle["scaler"]
model   = _bundle["model"]

# ── Application ───────────────────────────────────────────────────────────
app = FastAPI(title="Early Warning System - Core API")


# ── Schéma d'entrée ───────────────────────────────────────────────────────
class StudentInput(BaseModel):
    student_id: str

    # Critères utilisés par le modèle IA
    attendance_pct:        float
    absence_hours:         float
    homework_pct:          float
    study_hours_per_week:  float
    midterm_score:         float

    # Notes par matière (optionnelles, utilisées pour le diagnostic)
    informatique:          float = 0.0
    mathematique:          float = 0.0
    svt:                   float = 0.0
    physique:              float = 0.0
    sport:                 float = 0.0
    education_islamique:   float = 0.0
    francais:              float = 0.0
    arabe:                 float = 0.0


# ── Route de prédiction ───────────────────────────────────────────────────
@app.post("/predict")
def predict_student_status(student: StudentInput):
    # Préparation des données
    input_df = pd.DataFrame(
        [[
            student.attendance_pct,
            student.absence_hours,
            student.homework_pct,
            student.study_hours_per_week,
            student.midterm_score,
        ]],
        columns=CRITERES_IA,
    )
    input_scaled = scaler.transform(input_df)

    # Prédiction
    prediction_code = int(model.predict(input_scaled)[0])
    probabilites    = model.predict_proba(input_scaled)[0]
    prob_fail = round(float(probabilites[0]) * 100, 2)
    prob_pass = round(float(probabilites[1]) * 100, 2)

    # Construction du diagnostic
    motifs: list[str] = []

    # ── Critères comportementaux ───────────────────────────────────────────
    if student.absence_hours > 20:
        motifs.append(f"Volume d'absences critique ({student.absence_hours}h)")
    if student.attendance_pct < 75:
        motifs.append(f"Taux de présence insuffisant ({student.attendance_pct}%)")
    if student.homework_pct < 65:
        motifs.append(f"Retards répétés sur les devoirs rendus ({student.homework_pct}%)")
    if student.study_hours_per_week < 6:
        motifs.append(f"Temps d'étude personnel trop faible ({student.study_hours_per_week}h/semaine)")
    if student.midterm_score < 60:
        motifs.append(f"Note globale aux examens d'alerte ({student.midterm_score}/100)")

    # ── Notes par matière ─────────────────────────────────────────────────
    matieres = {
        "Informatique":       student.informatique,
        "Mathématiques":      student.mathematique,
        "SVT":                student.svt,
        "Physique":           student.physique,
        "Sport":              student.sport,
        "Éducation Islamique": student.education_islamique,
        "Français":           student.francais,
        "Arabe":              student.arabe,
    }
    for nom, note in matieres.items():
        if note < 50:
            motifs.append(f"Insuffisance académique majeure en {nom} ({note}/100)")

    if not motifs and prediction_code == 0:
        motifs.append("Échec comportemental global déterminé par le modèle d'IA.")

    return {
        "student_id":              student.student_id,
        "prediction_ia":           "PASS" if prediction_code == 1 else "FAIL",
        "pass_binary":             prediction_code,
        "probabilite_reussite":    f"{prob_pass}%",
        "probabilite_echec":       f"{prob_fail}%",
        "risque_valeur":           float(prob_fail),
        "diagnostic_administration": {
            "motifs_du_danger": motifs
        },
    }


# ── Lancement direct ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
