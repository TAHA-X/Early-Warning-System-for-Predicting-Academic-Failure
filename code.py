from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1. ENTRAÎNEMENT UNIQUE DU MODÈLE IA (Fichier data.csv uniquement)
# =====================================================================
TRAIN_FILE = "data.csv"

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"❌ Erreur critique : Le fichier d'entraînement `{TRAIN_FILE}` est introuvable.")

# Lecture et nettoyage des colonnes
df_base = pd.read_csv(TRAIN_FILE, sep=';')
df_base.columns = df_base.columns.str.strip()

criteres_ia = ['attendance_pct', 'absence_hours', 'homework_pct', 'study_hours_per_week', 'midterm_score']
X = df_base[criteres_ia]
y = df_base['pass']

# Standardisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèle équilibré pour contrer le déséquilibre des classes
model = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# =====================================================================
# 2. CONFIGURATION ET ROUTES DE L'API FASTAPI
# =====================================================================
app = FastAPI(title="Early Warning System - Core API")

class StudentInput(BaseModel):
    student_id: str
    attendance_pct: float
    absence_hours: float
    homework_pct: float
    study_hours_per_week: float
    midterm_score: float
    informatique: float = 0.0
    mathematique: float = 0.0
    svt: float = 0.0
    physique: float = 0.0
    sport: float = 0.0
    education_islamique: float = 0.0
    francais: float = 0.0
    arabe: float = 0.0

@app.post("/predict")
def predict_student_status(student: StudentInput):
    # Formatage des données pour le scaler et le modèle
    input_data = pd.DataFrame([[
        student.attendance_pct,
        student.absence_hours,
        student.homework_pct,
        student.study_hours_per_week,
        student.midterm_score
    ]], columns=criteres_ia)
    
    input_scaled = scaler.transform(input_data)
    
    # Exécution des prédictions IA
    prediction_code = int(model.predict(input_scaled)[0])  # 1 = PASS, 0 = FAIL
    probabilites = model.predict_proba(input_scaled)[0]
    
    prob_fail = round(probabilites[0] * 100, 2)
    prob_pass = round(probabilites[1] * 100, 2)
    
    # Détection simultanée et complète de tous les motifs métiers de défaillance
    motifs = []
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
        
    # Analyse des insuffisances par matière académique
    matieres = {
        "Informatique": student.informatique, "Mathématiques": student.mathematique,
        "SVT": student.svt, "Physique": student.physique, "Sport": student.sport,
        "Éducation Islamique": student.education_islamique, "Français": student.francais, "Arabe": student.arabe
    }
    for nom_matiere, note in matieres.items():
        if note < 50:
            motifs.append(f"Insuffisance académique majeure en {nom_matiere} ({note}/100)")

    if not motifs and prediction_code == 0:
        motifs.append("Échec comportemental global déterminé par le modèle d'IA.")

    # Dictionnaire de retour synchronisé au millimètre avec l'interface
    return {
        "student_id": student.student_id,
        "prediction_ia": "PASS" if prediction_code == 1 else "FAIL",
        "pass_binary": prediction_code,
        "probabilite_reussite": f"{prob_pass}%",
        "probabilite_echec": f"{prob_fail}%",
        "risque_valeur": float(prob_fail),
        "diagnostic_administration": {
            "motifs_du_danger": motifs
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)