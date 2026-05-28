from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =====================================================================
# STEP 1 : ENTRAÎNEMENT DE L'IA SUR LES CRITÈRES GLOBAUX UNIQUEMENT
# =====================================================================
if not os.path.exists('data.csv'):
    print("[ERREUR] Le fichier 'data.csv' est introuvable.")
    exit()

df_base = pd.read_csv('data.csv', sep=';')
df_base.columns = df_base.columns.str.strip()

# L'IA s'entraîne uniquement sur les 5 critères de base
criteres_ia = ['attendance_pct', 'absence_hours', 'homework_pct', 'study_hours_per_week', 'midterm_score']
X = df_base[criteres_ia]
y = df_base['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Utilisation de l'équilibrage des classes pour une détection fine du FAIL
model = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# =====================================================================
# STEP 2 : CONFIGURATION DE L'API FASTAPI
# =====================================================================
app = FastAPI(title="Early Warning System - API", version="6.0.0")

class StudentInput(BaseModel):
    student_id: str
    attendance_pct: float
    absence_hours: float
    homework_pct: float
    study_hours_per_week: float
    midterm_score: float
    informatique: float
    mathematique: float
    svt: float
    physique: float
    sport: float
    education_islamique: float
    francais: float
    arabe: float

# =====================================================================
# STEP 3 : LOGIQUE DE DIAGNOSTIC (LISTE COMPLÈTE SANS FILTRE)
# =====================================================================
@app.post("/predict")
def predict_student_status(student: StudentInput):
    
    donnees_ia = pd.DataFrame([{
        'attendance_pct': student.attendance_pct,
        'absence_hours': student.absence_hours,
        'homework_pct': student.homework_pct,
        'study_hours_per_week': student.study_hours_per_week,
        'midterm_score': student.midterm_score
    }])
    
    donnees_ia_scaled = scaler.transform(donnees_ia)
    
    probabilites = model.predict_proba(donnees_ia_scaled)[0]
    prob_echec = probabilites[0] * 100
    prob_reussite = probabilites[1] * 100
    prediction_finale = int(model.predict(donnees_ia_scaled)[0])
    
    verdict = "PASS" if prediction_finale == 1 else "FAIL"
    raisons_echec = []
    
    # Si l'IA détecte un FAIL, on liste TOUTES les raisons sans exception
    if verdict == "FAIL":
        
        # A. Vérification de TOUS les critères comportementaux défaillants
        if student.absence_hours > 20:
            raisons_echec.append(f"Volume d'absences trop élevé ({student.absence_hours} heures)")
            
        if student.attendance_pct < 75:
            raisons_echec.append(f"Taux de présence globale insuffisant ({student.attendance_pct}%)")
            
        if student.homework_pct < 65:
            raisons_echec.append(f"Manque d'implication dans les devoirs rendus ({student.homework_pct}%)")
            
        if student.study_hours_per_week < 6:
            raisons_echec.append(f"Temps d'étude hebdomadaire trop faible ({student.study_hours_per_week} heures)")
            
        if student.midterm_score < 60:
            raisons_echec.append(f"Score global aux examens de mi-parcours insuffisant ({student.midterm_score}/100)")

        # B. Vérification de TOUTES les matières en dessous de 50
        matieres = {
            "Informatique": student.informatique, "Mathématiques": student.mathematique,
            "SVT": student.svt, "Physique": student.physique, "Sport": student.sport,
            "Éducation Islamique": student.education_islamique, "Français": student.francais,
            "Arabe": student.arabe
        }
        for nom_matiere, note in matieres.items():
            if note < 50:
                raisons_echec.append(f"Matière critique affectant le score : {nom_matiere} ({note}/100)")

    return {
        "student_id": student.student_id,
        "prediction_ia": verdict,
        "probabilite_echec": f"{prob_echec:.2f}%",
        "probabilite_reussite": f"{prob_reussite:.2f}%",
        "diagnostic_administration": {
            "statut_alertes": "⚠️ Alerte déclenchée" if verdict == "FAIL" else "✅ Profil stable",
            "motifs_du_danger": raisons_echec if raisons_echec else ["Échec global déterminé par l'IA."]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)