from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# =====================================================================
# 1. ENTRAÎNEMENT UNIQUE ET ÉVALUATION MÉTRIQUE DU MODÈLE IA
# =====================================================================
TRAIN_FILE = "data.csv"

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"❌ Erreur critique : Le fichier d'entraînement `{TRAIN_FILE}` est introuvable.")

# Lecture et nettoyage des colonnes du fichier data.csv
df_base = pd.read_csv(TRAIN_FILE, sep=';')
df_base.columns = df_base.columns.str.strip()

criteres_ia = ['attendance_pct', 'absence_hours', 'homework_pct', 'study_hours_per_week', 'midterm_score']
X = df_base[criteres_ia]

# Nettoyage de la variable cible (Extrait uniquement le premier chiffre trouvé : 0 ou 1)
y = df_base['pass'].astype(str).str.extract(r'(\d)')[0].astype(int)

# Découpage initial stratifié (80% Train, 20% Test externe)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)

# Déclaration de notre modèle de Régression Logistique équilibré
model = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)

# --- CALCUL DU J_train ET DE LA CROSS-VALIDATION (K-FOLD K=5) ---
cv_results = cross_validate(
    model, 
    scaler.fit_transform(X), 
    y, 
    cv=5, 
    scoring=['neg_log_loss', 'accuracy'], 
    return_train_score=True
)

# Extraction des coûts moyens
j_train_kfolds = -cv_results['train_neg_log_loss'].mean()
j_cv_kfolds = -cv_results['test_neg_log_loss'].mean()
accuracy_cv = cv_results['test_accuracy'].mean()
pourcentage_erreur_cv = (1.0 - accuracy_cv) * 100

# --- CALCUL DU J GLOBAL ---
X_scaled_full = scaler.fit_transform(X)
model.fit(X_scaled_full, y)

y_proba_reussite = model.predict_proba(X_scaled_full)[:, 1]
y_proba_reussite = np.clip(y_proba_reussite, 1e-15, 1 - 1e-15)

m = len(y)
j_global = - (1 / m) * np.sum(y * np.log(y_proba_reussite) + (1 - y) * np.log(1 - y_proba_reussite))

# --- ENTRAÎNEMENT DU MODÈLE SUR LE JEU DE TEST EXTÉRIEUR POUR LA MATRICE ---
model_test = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)
model_test.fit(X_train_scaled, y_train)
y_val_pred = model_test.predict(X_val_scaled)

# Génération du rapport de classification initial
raw_report = classification_report(y_val, y_val_pred, target_names=['FAIL (0)', 'PASS (1)'])

# Formatage avec indentation exacte
formatted_report = ""
for line in raw_report.split('\n'):
    if any(x in line for x in ['precision', 'FAIL (0)', 'PASS (1)']):
        formatted_report += "    " + line + "\n"

# =====================================================================
# AFFICHAGE DU RAPPORT EXACTEMENT SELON VOTRE MODÈLE
# =====================================================================



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
    input_data = pd.DataFrame([[
        student.attendance_pct,
        student.absence_hours,
        student.homework_pct,
        student.study_hours_per_week,
        student.midterm_score
    ]], columns=criteres_ia)
    
    input_scaled = scaler.transform(input_data)
    
    prediction_code = int(model.predict(input_scaled)[0])
    probabilites = model.predict_proba(input_scaled)[0]
    
    prob_fail = round(probabilites[0] * 100, 2)
    prob_pass = round(probabilites[1] * 100, 2)
    
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
    uvicorn.run(app, host="127.0.0.1", port=8002)