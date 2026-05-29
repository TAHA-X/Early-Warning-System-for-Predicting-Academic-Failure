import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
import threading
import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# INITIALISATION ET ENTRAÎNEMENT DE L'IA EN LOCAL
TRAIN_FILE = "data.csv"
DATA_FILE = "etudiants.csv"
API_URL = "http://127.0.0.1:8000/predict"

if not os.path.exists(TRAIN_FILE):
    st.error(f"❌ Le fichier critique `{TRAIN_FILE}` est introuvable. L'IA ne peut pas s'entraîner.")
    st.stop()

df_base = pd.read_csv(TRAIN_FILE, sep=';')
df_base.columns = df_base.columns.str.strip()

criteres_ia = ['attendance_pct', 'absence_hours', 'homework_pct', 'study_hours_per_week', 'midterm_score']
X = df_base[criteres_ia]
y = df_base['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# CONFIGURATION DE L'API REST INTERNE
app_internal = FastAPI(title="Early Warning System - API Internal")

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

@app_internal.post("/predict")
def predict_student_status(student: StudentInput):
    # Appel de la même logique que code.py pour rester synchronisé
    from code import predict_student_status as real_predict
    return real_predict(student)

def run_fastapi():
    uvicorn.run("code:app", host="127.0.0.1", port=8000, log_level="warning")

if "fastapi_started" not in st.session_state:
    threading.Thread(target=run_fastapi, daemon=True).start()
    st.session_state["fastapi_started"] = True
    time.sleep(1)

# INTERFACE UTILISATEUR STREAMLIT
st.set_page_config(page_title="EWS - Early Warning System", page_icon="🎓", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Aller vers :", ["📊 Statistiques Globales", "🤖 Diagnostic & Inscription", "🚨 Cellule d'Alerte & Risques"])

# PAGE 1 : STATISTIQUES GLOBALES
if page == "📊 Statistiques Globales":
    st.title("📊 Tableau de Bord des Étudiants Inscrits")
    st.write("Analyse descriptive de la promotion actuelle issue de `etudiants.csv`.")
    st.divider()

    # SECTION IMPORTATION DE FICHIER HÉBERGÉ
    st.markdown("### 📥 Importation de données externes (.xlsx, .xls, .txt)")
    col_file, col_opt = st.columns([2, 1])
    
    with col_file:
        uploaded_file = st.file_uploader("Choisissez un fichier Excel ou Texte (sans colonne 'pass', séparateur ';')", type=["xlsx", "xls", "txt"])
    with col_opt:
        import_mode = st.radio("Méthode d'importation :", ["Ajouter aux données existantes", "Remplacer complètement le fichier"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.txt'):
                df_imported = pd.read_csv(uploaded_file, sep=';')
            else:
                df_imported = pd.read_excel(uploaded_file)
            
            df_imported.columns = df_imported.columns.str.strip()
            
            # Les colonnes que l'administration doit obligatoirement fournir
            colonnes_requises = ['student_id', 'attendance_pct', 'absence_hours', 'homework_pct', 'study_hours_per_week', 'midterm_score']
            
            if not all(col in df_imported.columns for col in colonnes_requises):
                st.error("🚨 Erreur : Le fichier doit contenir au moins les colonnes comportementales de base.")
            else:
                if st.button("Confirmer l'importation & Évaluation Automatique 💾", type="primary"):
                    with st.spinner("L'IA calcule les prédictions pour les profils importés..."):
                        
                        # 1. Nettoyage strict des lignes complètement vides
                        df_imported = df_imported.dropna(subset=colonnes_requises)
                        
                        # 2. Remplacement des valeurs manquantes (NaN) par 0 pour éviter le crash du modèle
                        df_imported = df_imported.fillna(0)

                        # 3. Évaluation automatique par lot via le modèle entraîné
                        X_imp = df_imported[criteres_ia]
                        X_imp_scaled = scaler.transform(X_imp)
                        
                        # Génération dynamique de la colonne pass (0 ou 1)
                        df_imported['pass'] = model.predict(X_imp_scaled)
                        
                        # Garantir l'ordre idéal des colonnes pour correspondre à etudiants.csv
                        ordre_cols = ['student_id', 'attendance_pct', 'absence_hours', 'homework_pct', 
                                      'study_hours_per_week', 'midterm_score', 'informatique', 'mathematique', 
                                      'svt', 'physique', 'sport', 'education_islamique', 'francais', 'arabe', 'pass']
                        
                        # Ajouter les colonnes de matières manquantes si elles n'existent pas dans l'import
                        for col_mat in ordre_cols:
                            if col_mat not in df_imported.columns:
                                df_imported[col_mat] = 0.0
                                
                        df_imported = df_imported[ordre_cols]

                        # 4. Sauvegarde physique
                        if import_mode == "Remplacer complètement le fichier" or not os.path.exists(DATA_FILE):
                            df_imported.to_csv(DATA_FILE, index=False, sep=';')
                            st.success("🎯 Fichier étudiant initialisé et entièrement calculé par l'IA !")
                        else:
                            df_existing = pd.read_csv(DATA_FILE, sep=';')
                            df_existing.columns = df_existing.columns.str.strip()
                            df_final = pd.concat([df_existing, df_imported]).drop_duplicates(subset=['student_id'], keep='last')
                            df_final.to_csv(DATA_FILE, index=False, sep=';')
                            st.success("➕ Nouveaux profils évalués par l'IA et insérés sans doublons !")
                        
                        time.sleep(1)
                        st.rerun()
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {e}")

    st.divider()

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, sep=';')
        df.columns = df.columns.str.strip()
        if not df.empty:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Étudiants Inscrits", f"{len(df)}")
            col_m2.metric("Moyenne des Absences", f"{df['absence_hours'].mean():.1f} h")
            col_m3.metric("Moyenne Mi-Parcours", f"{df['midterm_score'].mean():.1f}/100")
            st.divider()
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(px.histogram(df, x='absence_hours', title="Distribution des heures d'absence", color_discrete_sequence=['#3498db']), use_container_width=True)
            with col_g2:
                st.plotly_chart(px.scatter(df, x='study_hours_per_week', y='midterm_score', title="Étude vs Note Mi-Parcours", color='midterm_score', color_continuous_scale='Blues'), use_container_width=True)
            st.dataframe(df, height=250, use_container_width=True)
        else:
            st.info("💡 Le fichier `etudiants.csv` est vide.")
    else:
        st.info("💡 Aucun étudiant enregistré.")

# PAGE 2 : DIAGNOSTIC & INSCRIPTION UNITAIRE
elif page == "🤖 Diagnostic & Inscription":
    st.title("🤖 Assistant de Diagnostic & Inscription")
    st.divider()

    with st.form("student_form"):
        st.markdown("### 👤 Informations & Comportement")
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("ID de l'Étudiant", value="101").strip()
            attendance_pct = st.slider("Taux de présence (%)", 0, 100, 50)
            absence_hours = st.number_input("Heures d'absence", min_value=0, max_value=200, value=45)
        with col2:
            homework_pct = st.slider("Devoirs rendus (%)", 0, 100, 40)
            study_hours = st.slider("Heures d'étude / semaine", 0, 50, 2)
            midterm_score = st.slider("Note mi-parcours globale (/100)", 0, 100, 45)

        st.markdown("### 📚 Notes par Matière (/100)")
        col3, col4 = st.columns(2)
        with col3:
            info = st.slider("Informatique", 0, 100, 40)
            math = st.slider("Mathématiques", 0, 100, 35)
            svt = st.slider("SVT", 0, 100, 70)
            physique = st.slider("Physique", 0, 100, 45)
        with col4:
            sport = st.slider("Sport", 0, 100, 85)
            islamique = st.slider("Éducation Islamique", 0, 100, 90)
            francais = st.slider("Français", 0, 100, 48)
            arabe = st.slider("Arabe", 0, 100, 75)

        submit_button = st.form_submit_button("Lancer le Diagnostic & Enregistrer 💾", type="primary")

    if submit_button:
        if student_id == "":
            st.error("🚨 L'ID de l'étudiant ne peut pas être vide.")
        else:
            id_deja_existe = False
            if os.path.exists(DATA_FILE):
                try:
                    df_lecture = pd.read_csv(DATA_FILE, sep=';')
                    if student_id in df_lecture.iloc[:, 0].astype(str).str.strip().tolist():
                        id_deja_existe = True
                except Exception:
                    pass

            if id_deja_existe:
                st.error(f"🚨 REFUSÉ : L'ID **{student_id}** est déjà enregistré.")
            else:
                payload = {
                    "student_id": student_id, "attendance_pct": float(attendance_pct),
                    "absence_hours": float(absence_hours), "homework_pct": float(homework_pct),
                    "study_hours_per_week": float(study_hours), "midterm_score": float(midterm_score),
                    "informatique": float(info), "mathematique": float(math), "svt": float(svt),
                    "physique": float(physique), "sport": float(sport), "education_islamique": float(islamique),
                    "francais": float(francais), "arabe": float(arabe)
                }

                try:
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        res = response.json()
                        
                        # Ajouter le tag binaire pass calculé par l'API pour le stockage csv
                        payload['pass'] = 1 if res["prediction_ia"] == "PASS" else 0
                        nouvel_etudiant = pd.DataFrame([payload])
                        
                        if os.path.exists(DATA_FILE):
                            nouvel_etudiant.to_csv(DATA_FILE, mode='a', header=False, index=False, sep=';')
                        else:
                            nouvel_etudiant.to_csv(DATA_FILE, mode='w', header=True, index=False, sep=';')
                        
                        @st.dialog("📋 Rapport de Diagnostic Réalisé", width="large")
                        def afficher_popup():
                            st.write(f"**Étudiant ID :** {student_id}")
                            st.success("💾 Enregistré avec succès dans `etudiants.csv`.")
                            if res["prediction_ia"] == "PASS":
                                st.success(f"🎉 **Verdict IA : PASS** (Probabilité : {res['probabilite_reussite']})")
                                st.balloons()
                            else:
                                st.error(f"🚨 **Verdict IA : FAIL** (Probabilité : {res['probabilite_echec']})")
                                for motif in res["diagnostic_administration"]["motifs_du_danger"]:
                                    st.write(f"- {motif}")
                            if st.button("Fermer"): st.rerun()
                        afficher_popup()
                except Exception:
                    st.error("❌ Erreur de communication avec le moteur d'analyse.")

# PAGE 3 : CELLULE D'ALERTE
elif page == "🚨 Cellule d'Alerte & Risques":
    st.title("🚨 Cellule de Détection Préventive")
    st.divider()

    if os.path.exists(DATA_FILE):
        df_students = pd.read_csv(DATA_FILE, sep=';')
        df_students.columns = df_students.columns.str.strip()

        if df_students.empty:
            st.info("💡 Aucun étudiant enregistré dans `etudiants.csv`.")
        else:
            X_eval = df_students[criteres_ia]
            X_eval_scaled = scaler.transform(X_eval)
            df_students['Prediction_Code'] = model.predict(X_eval_scaled)
            df_students['Risque_%'] = (model.predict_proba(X_eval_scaled)[:, 0] * 100).round(2)
            
            # Profils à risque (FAIL = Code 0)
            df_en_danger = df_students[df_students['Prediction_Code'] == 0].copy()

            col_a1, col_a2 = st.columns(2)
            col_a1.metric("Total étudiants analysés", len(df_students))
            col_a2.metric("🎯 Profils À RISQUE (IA)", len(df_en_danger), delta=f"{len(df_en_danger)} à suivre", delta_color="inverse")

            st.divider()
            
            if df_en_danger.empty:
                st.success("✅ Aucun étudiant à risque détecté dans la promotion actuelle.")
            else:
                st.markdown("### 📋 Liste épurée des profils défaillants")
                df_tableau_epure = df_en_danger[['student_id', 'Risque_%', 'absence_hours']].copy()
                df_tableau_epure.columns = ['ID Étudiant', 'Probabilité Échec', 'Heures Absences']
                st.dataframe(df_tableau_epure.style.set_properties(**{'background-color': '#fdf2f2', 'color': '#9b1c1c'}, subset=['Probabilité Échec']), use_container_width=True)
                
                st.divider()
                
                st.markdown("### 🔍 Inspecter un profil critique en détail")
                liste_ids_danger = df_en_danger['student_id'].astype(str).tolist()
                id_selectionne = st.selectbox("Sélectionnez l'ID de l'étudiant à inspecter :", options=liste_ids_danger)
                
                @st.dialog("🔬 Fiche Diagnostic Détaillée - Cellule d'Alerte", width="large")
                def popup_details(student_id_target):
                    row_student = df_en_danger[df_en_danger['student_id'].astype(str) == student_id_target].iloc[0]
                    st.markdown(f"## 👤 Étudiant ID : `{student_id_target}`")
                    st.error(f"🚨 **Probabilité de Décrochage / Échec : {row_student['Risque_%']}%**")
                    st.divider()
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    col_b1.metric("Présence globale", f"{row_student['attendance_pct']}%")
                    col_b2.metric("Volume Absences", f"{row_student['absence_hours']} h")
                    col_b3.metric("Note Mi-parcours", f"{row_student['midterm_score']}/100")
                    
                    st.markdown("### 📚 Relevé complet des notes (/100)")
                    col_n1, col_n2 = st.columns(2)
                    with col_n1:
                        st.write(f"🔹 **Informatique :** {row_student['informatique']}/100")
                        st.write(f"🔹 **Mathématiques :** {row_student['mathematique']}/100")
                        st.write(f"🔹 **SVT :** {row_student['svt']}/100")
                        st.write(f"🔹 **Physique :** {row_student['physique']}/100")
                    with col_n2:
                        st.write(f"🔹 **Sport :** {row_student['sport']}/100")
                        st.write(f"🔹 **Éducation Islamique :** {row_student['education_islamique']}/100")
                        st.write(f"🔹 **Français :** {row_student['francais']}/100")
                        st.write(f"🔹 **Arabe :** {row_student['arabe']}/100")
                    
                    st.divider()
                    st.markdown("### 🎯 Facteurs métiers déclencheurs de l'alerte :")
                    motifs = []
                    if row_student['absence_hours'] > 20: motifs.append(f"• Volume d'absences critique ({row_student['absence_hours']}h)")
                    if row_student['attendance_pct'] < 75: motifs.append(f"• Taux de présence insuffisant ({row_student['attendance_pct']}%)")
                    if row_student['homework_pct'] < 65: motifs.append(f"• Retards répétés sur les devoirs rendus ({row_student['homework_pct']}%)")
                    if row_student['study_hours_per_week'] < 6: motifs.append(f"• Temps d'étude personnel trop faible ({row_student['study_hours_per_week']}h/semaine)")
                    if row_student['midterm_score'] < 60: motifs.append(f"• Note globale aux examens d'alerte ({row_student['midterm_score']}/100)")
                    
                    matieres_list = {"Informatique": row_student['informatique'], "Mathématiques": row_student['mathematique'], "SVT": row_student['svt'], "Physique": row_student['physique'], "Français": row_student['francais'], "Arabe": row_student['arabe'], "Sport": row_student['sport'], "Éducation Islamique": row_student['education_islamique']}
                    for k, v in matieres_list.items():
                        if v < 50: motifs.append(f"• Insuffisance académique majeure en **{k}** ({v}/100)")
                    
                    if motifs:
                        for m in motifs: st.write(m)
                    else:
                        st.write("• Échec comportemental global déterminé par l'IA.")
                        
                    if st.button("Fermer la fiche"): st.rerun()

                if st.button("Afficher la Fiche d'Alerte Complète 🔍", type="primary"):
                    popup_details(id_selectionne)

                # =====================================================================
                # ANALYSE SECTORIELLE (SYNCHRONISATION DES ACCENTS VALIDE)
                # =====================================================================
                st.divider()
                
                total_defaillances_matieres = {
                    "Informatique": 0,
                    "Mathématiques": 0,
                    "SVT": 0,
                    "Physique": 0,
                    "Sport": 0,
                    "Éducation Islamique": 0,
                    "Français": 0,
                    "Arabe": 0
                }
                
                for _, row in df_en_danger.iterrows():
                    matieres = {
                        "Informatique": row['informatique'], 
                        "Mathématiques": row['mathematique'], 
                        "SVT": row['svt'], 
                        "Physique": row['physique'], 
                        "Sport": row['sport'], 
                        "Éducation Islamique": row['education_islamique'], 
                        "Français": row['francais'], 
                        "Arabe": row['arabe']
                    }
                    for nom, note in matieres.items():
                        if note < 50: 
                            total_defaillances_matieres[nom] += 1
                
                df_bar_matiere = pd.DataFrame(list(total_defaillances_matieres.items()), columns=['Matière', 'Nombre d\'étudiants en échec'])
                st.plotly_chart(px.bar(df_bar_matiere, x='Matière', y='Nombre d\'étudiants en échec', title="Analyse Sectorielle : Quelles matières mettent la promo en péril ?", color='Nombre d\'étudiants en échec', color_continuous_scale='Reds'), use_container_width=True)
    else:
        st.info("💡 Base de données `etudiants.csv` inexistante.")