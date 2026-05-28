import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="EWS - Early Warning System",
    page_icon="🎓",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"
DATA_FILE = "etudiants.csv" 
TRAIN_FILE = "data.csv"

# =====================================================================
# BARRE DE NAVIGATION (SIDEBAR)
# =====================================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Aller vers :", 
    ["📊 Statistiques Globales", "🤖 Diagnostic & Inscription", "🚨 Cellule d'Alerte & Risques"]
)

st.sidebar.divider()
#st.sidebar.info("💡 **EWS v7.0**\nTrois espaces dédiés à l'administration universitaire.")#

# =====================================================================
# PAGE 1 : STATISTIQUES GLOBALES (DEPUIS ETUDIANTS.CSV)
# =====================================================================
if page == "📊 Statistiques Globales":
    st.title("📊 Tableau de Bord des Étudiants Inscrits")
    st.write("Analyse descriptive et suivi de la promotion actuelle issue du fichier `etudiants.csv`.")
    st.divider()

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, sep=';')
        df.columns = df.columns.str.strip()

        if not df.empty:
            total_etudiants = len(df)
            moyenne_absences = df['absence_hours'].mean()
            moyenne_midterm = df['midterm_score'].mean()

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Étudiants Inscrits", f"{total_etudiants}")
            col_m2.metric("Moyenne des Absences (Promo)", f"{moyenne_absences:.1f} heures")
            col_m3.metric("Moyenne Générale Mi-Parcours", f"{moyenne_midterm:.1f}/100")

            st.divider()

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("### 📊 Distribution des Heures d'Absence")
                fig_hist = px.histogram(df, x='absence_hours', 
                                        labels={'absence_hours': 'Heures d\'absence', 'count': 'Nombre d\'étudiants'},
                                        color_discrete_sequence=['#3498db'], nbins=15)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_g2:
                st.markdown("### ⏳ Corrélation : Heures d'Étude vs Note Mi-Parcours")
                fig_scatter = px.scatter(df, x='study_hours_per_week', y='midterm_score',
                                         labels={'study_hours_per_week': 'Heures d\'étude / semaine', 'midterm_score': 'Note Mi-parcours'},
                                         color='midterm_score', color_continuous_scale='Blues')
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("### 📋 Liste Globale des Étudiants (`etudiants.csv`)")
            st.dataframe(df, height=300, use_container_width=True)
        else:
            st.info("💡 Le fichier `etudiants.csv` est vide pour le moment.")
    else:
        st.info("💡 La base de données `etudiants.csv` est actuellement vide. Allez dans le deuxième onglet pour ajouter votre premier étudiant.")

# =====================================================================
# PAGE 2 : FORMULAIRE DE DIAGNOSTIC ET ENREGISTREMENT
# =====================================================================
elif page == "🤖 Diagnostic & Inscription":
    st.title("🤖 Assistant de Diagnostic & Inscription")
    st.write("Saisissez les données. Si l'ID existe déjà dans `etudiants.csv`, la soumission sera bloquée.")
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
                    df_lecture.columns = df_lecture.columns.str.strip()
                    liste_ids = df_lecture['student_id'].astype(str).str.strip().tolist()
                    if student_id in liste_ids:
                        id_deja_existe = True
                except Exception:
                    pass

            if id_deja_existe:
                st.error(f"🚨 ENREGISTREMENT REFUSÉ : L'ID **{student_id}** est déjà enregistré dans `etudiants.csv`.")
            else:
                payload = {
                    "student_id": student_id, "attendance_pct": float(attendance_pct),
                    "absence_hours": float(absence_hours), "homework_pct": float(homework_pct),
                    "study_hours_per_week": float(study_hours), "midterm_score": float(midterm_score),
                    "informatique": float(info), "mathematique": float(math), "svt": float(svt),
                    "physique": float(physique), "sport": float(sport), "education_islamique": float(islamique),
                    "francais": float(francais), "arabe": float(arabe)
                }

                with st.spinner("Analyse du profil par l'Intelligence Artificielle..."):
                    try:
                        response = requests.post(API_URL, json=payload)
                        if response.status_code == 200:
                            resultat = response.json()
                            verdict = resultat["prediction_ia"]
                            prob_echec = resultat["probabilite_echec"]
                            prob_reussite = resultat["probabilite_reussite"]
                            alertes = resultat["diagnostic_administration"]["motifs_du_danger"]
                            
                            nouvel_etudiant = pd.DataFrame([payload])
                            if os.path.exists(DATA_FILE):
                                nouvel_etudiant.to_csv(DATA_FILE, mode='a', header=False, index=False, sep=';')
                            else:
                                nouvel_etudiant.to_csv(DATA_FILE, mode='w', header=True, index=False, sep=';')
                            
                            @st.dialog("📋 Rapport de Diagnostic Réalisé", width="large")
                            def afficher_popup():
                                st.write(f"**Étudiant ID :** {student_id}")
                                st.success("💾 Les données ont été enregistrées avec succès dans `etudiants.csv`.")
                                if verdict == "PASS":
                                    st.success(f"🎉 **Verdict IA : PASS** (Probabilité de réussite : {prob_reussite})")
                                    st.balloons()
                                else:
                                    st.error(f"🚨 **Verdict IA : FAIL** (Probabilité d'échec : {prob_echec})")
                                    st.markdown("#### 🔍 Causes détectées par le système :")
                                    for motif in alertes:
                                        st.write(f"- {motif}")
                                st.divider()
                                if st.button("Fermer le rapport"):
                                    st.rerun()

                            afficher_popup()
                        else:
                            st.error("❌ Erreur de communication avec l'API.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Impossible de joindre l'API FastAPI. Activez d'abord le fichier `code.py` !")

# =====================================================================
# NOUVELLE PAGE 3 : DÉTECTION AUTOMATIQUE DES ÉTUDIANTS À RISQUE
# =====================================================================
elif page == "🚨 Cellule d'Alerte & Risques":
    st.title("🚨 Cellule de Détection Préventive")
    st.write("Cette page applique l'IA en masse sur tout le fichier `etudiants.csv` pour extraire automatiquement les profils en danger.")
    st.divider()

    if not os.path.exists(TRAIN_FILE) or not os.path.exists(DATA_FILE):
        st.error("❌ Les fichiers `data.csv` et `etudiants.csv` doivent être présents pour faire tourner l'analyse globale.")
    else:
        # 1. Chargement et entraînement local ultra-rapide pour l'analyse de masse
        df_base = pd.read_csv(TRAIN_FILE, sep=';')
        df_base.columns = df_base.columns.str.strip()
        criteres_ia = ['attendance_pct', 'absence_hours', 'homework_pct', 'study_hours_per_week', 'midterm_score']
        
        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(df_base[criteres_ia])
        model_local = LogisticRegression(class_weight='balanced', random_state=42)
        model_local.fit(X_train_scaled, df_base['pass'])

        # 2. Lecture des étudiants inscrits
        df_students = pd.read_csv(DATA_FILE, sep=';')
        df_students.columns = df_students.columns.str.strip()

        if df_students.empty:
            st.info("💡 Aucun étudiant enregistré dans `etudiants.csv`. Le tableau d'alerte est vide.")
        else:
            # Prédiction sur toute la liste d'un coup
            X_eval = df_students[criteres_ia]
            X_eval_scaled = scaler_local.transform(X_eval)
            
            df_students['Prediction_Code'] = model_local.predict(X_eval_scaled)
            df_students['Risque_%'] = (model_local.predict_proba(X_eval_scaled)[:, 0] * 100).round(2)
            
            # Filtrage : On isole uniquement les étudiants déclarés en échec (0) par l'IA
            df_en_danger = df_students[df_students['Prediction_Code'] == 0].copy()

            # Indicateurs de crise
            col_a1, col_a2 = st.columns(2)
            col_a1.metric("Nombre total d'étudiants analysés", len(df_students))
            col_a2.metric("🎯 Étudiants identifiés À RISQUE par l'IA", len(df_en_danger), delta=f"{len(df_en_danger)} profils à suivre", delta_color="inverse")

            st.divider()

            st.markdown("### 📋 Liste de Suivi des Profils Défaillants")
            if df_en_danger.empty:
                st.success("✅ Félicitations ! L'IA n'a détecté aucun étudiant à risque d'échec dans la promotion actuelle.")
            else:
                # Création des colonnes d'explication pour l'administration
                liste_affichage = []
                
                for _, row in df_en_danger.iterrows():
                    matieres_bloquantes = []
                    matieres = {
                        "Informatique": row['informatique'], "Mathématiques": row['mathematique'],
                        "SVT": row['svt'], "Physique": row['physique'], "Sport": row['sport'],
                        "Éducation Islamique": row['education_islamique'], "Français": row['francais'],
                        "Arabe": row['arabe']
                    }
                    # Identification des matières problématiques (< 50)
                    for nom, note in matieres.items():
                        if note < 50:
                            matieres_bloquantes.append(f"{nom} ({note}/100)")
                    
                    liste_affichage.append({
                        "ID Étudiant": row['student_id'],
                        "Probabilité Échec": f"{row['Risque_%']}%",
                        "Absences": f"{row['absence_hours']} h",
                        "Note Globale Mi-Parco": f"{row['midterm_score']}/100",
                        "📚 Matières Problématiques (<50)": ", ".join(matieres_bloquantes) if matieres_bloquantes else "Aucune matière < 50 (Échec purement comportemental)"
                    })

                df_dashboard_danger = pd.DataFrame(liste_affichage)
                
                # Affichage du tableau d'alerte avec coloration rouge
                st.dataframe(
                    df_dashboard_danger.style.set_properties(**{'background-color': '#fdf2f2', 'color': '#9b1c1c'}, subset=['Probabilité Échec']),
                    use_container_width=True
                )

                st.divider()
                
                # Petit graphique pour classer les pires matières de la promo à risque
                st.markdown("### 🔍 Analyse Sectorielle : Quelles matières mettent la promo en péril ?")
                total_defaillances_matieres = {}
                matieres_cles = ["informatique", "mathematique", "svt", "physique", "sport", "education_islamique", "francais", "arabe"]
                
                for m in matieres_cles:
                    total_defaillances_matieres[m.capitalize()] = (df_en_danger[m] < 50).sum()
                
                df_bar_matiere = pd.DataFrame(list(total_defaillances_matieres.items()), columns=['Matière', 'Nombre d\'étudiants en échec'])
                fig_bar = px.bar(df_bar_matiere, x='Matière', y='Nombre d\'étudiants en échec',
                                 title="Nombre d'alertes par matière chez les étudiants à risque",
                                 color='Nombre d\'étudiants en échec', color_continuous_scale='Reds')
                st.plotly_chart(fig_bar, use_container_width=True)