# 🎓 Système de Détection Préventive & Diagnostic du Décrochage Scolaire

Ce projet est une solution complète d'aide à la décision pour les établissements scolaires. Il combine l'analyse de données, le **Machine Learning** pour le calcul du risque d'échec, et une architecture moderne **API-First** (FastAPI) connectée à un **Dashboard interactif** (Streamlit). Le système permet de détecter de manière proactive les élèves en situation de décrochage afin de planifier des actions correctives.

## 🔥 Fonctionnalités Principales

* **Mode API-First :** Séparation stricte entre le moteur d'intelligence artificielle (FastAPI) et l'interface utilisateur.
* **Importation Dynamique :** Possibilité d'importer directement de nouveaux fichiers de données (`.xlsx`, `.xls`, `.txt`) pour mettre à jour la base ou remplacer complètement le fichier existant.
* **Analyse Descriptive Globale :** Statistiques en temps réel sur la promotion (moyenne des absences, notes de mi-parcours, distributions graphiques).
* **Formulaire de Diagnostic à la Volée :** Interface interactive pour simuler ou inscrire un nouvel étudiant en saisissant ses notes par matière et son comportement.
* **Cellule d'Alerte IA :** Identification automatique et instantanée des profils à risque avec calcul de la probabilité d'échec exacte (%).
* **Fiches de Diagnostic Détaillées :** Analyse sectorielle par étudiant mettant en avant les facteurs métiers déclencheurs transmis par l'API (matières critiques, manque d'implication, absentéisme).



## 📸 Aperçu du Dashboard (Interface Utilisateur)

### 📊 Page 1 : Statistiques Globales & Importation
Cette première page permet de charger de nouvelles données au format Excel ou texte pour enrichir la base de l'API. Elle affiche instantanément les indicateurs clés de la promotion (KPIs) ainsi que les graphiques de distribution.

<img width="1552" height="847" alt="dash1" src="https://github.com/user-attachments/assets/125fbe26-0bc1-4a1b-b087-8ba5557e5196" />


*Figure 1 : Première page du dashboard avec module d'importation et graphiques descriptifs.*

---

### 📝 Page 2 : Assistant de Diagnostic & Inscription
Cette interface contient un formulaire dynamique composé de curseurs (sliders) permettant de saisir le comportement (présence, devoirs rendus, heures d'étude) et les notes par matière d'un étudiant pour l'enregistrer et interroger l'API en direct.

<img width="1740" height="813" alt="dash2" src="https://github.com/user-attachments/assets/b4a39a9c-7fc6-48d8-8834-6824a00f94cb" />


*Figure 2 : Formulaire de simulation et d'inscription d'un nouvel étudiant (Mode API).*

---

### 🚨 Page 3 : Cellule de Détection Préventive (Vue d'ensemble)
Cette section liste de manière épurée l'ensemble des profils jugés "défaillants" par le modèle prédictif. Le tableau met en évidence l'ID de l'étudiant, sa probabilité d'échec précise et son volume d'absences.

<img width="1578" height="853" alt="dash3" src="https://github.com/user-attachments/assets/3e636d83-1619-484a-8a81-d862ec01867c" />


*Figure 3 : Tableau récapitulatif des profils à risque identifiés par l'IA.*

---

### 📉 Page 3 : Analyse Sectorielle des Matières Critiques
Directement sous le tableau de la page 3, un graphique en barres permet d'identifier visuellement quelles sont les matières qui mettent le plus la promotion en péril (nombre d'étudiants en situation d'échec par bloc d'enseignement).

<img width="1573" height="688" alt="dash4" src="https://github.com/user-attachments/assets/0d3397d4-8860-4927-931d-6b763fbbd47e" />


*Figure 4 : Statistiques sous format de graphe identifiant les matières critiques de la promotion.*

---

### 🔍 Focus : Fiche de Diagnostic Détaillée (Pop-up API)
Lorsqu'on inspecte un profil critique spécifique, l'API transmet une fiche complète. En plus des notes, elle explicite clairement les **facteurs métiers déclencheurs** (ex: volume d'absences trop élevé, matière critique affectant le score).

<img width="1247" height="837" alt="dash5" src="https://github.com/user-attachments/assets/58ad06da-e8f5-4943-9a12-eb17c927d596" />


*Figure 5 : Fenêtre de détails explicitant les causes du danger de décrochage pour un profil ciblé.*

---

## 🛠️ Architecture Technique

Le projet est découpé en deux composants autonomes communicant par requêtes HTTP (JSON) :

1. **Le Backend (API) :** Développé avec **FastAPI**. Il charge le modèle de Machine Learning préalablement entraîné (`Random Forest` / `Scikit-Learn`), valide les données entrantes avec `Pydantic` et effectue les prédictions de probabilité d'échec en tâche de fond.
2. **Le Frontend (Dashboard) :** Développé en **Streamlit**. Il offre une interface fluide et moderne, interroge l'API pour chaque calcul et s'occupe de la visualisation de données (`Plotly` / `Altair`).

---

## 🚀 Installation et Lancement

### 1. Prérequis & Installation des dépendances
Clonez le projet, créez un environnement virtuel, puis installez les bibliothèques requises :

pip install -r requirements.txt
(Le fichier requirements.txt doit inclure : fastapi, uvicorn, streamlit, pandas, scikit-learn, pydantic, requests, plotly).

2. Entraînement du modèle initial
Avant de lancer les serveurs, exécutez le script d'entraînement pour générer le modèle prédictif basé sur l'historique :


python code.py
3. Lancement de l'API (Backend)
Démarrez le serveur FastAPI sur le port par défaut (8000) :


uvicorn main:app --reload
La documentation interactive de l'API reste accessible sur http://127.0.0.1:8000/docs.

4. Lancement du Dashboard (Frontend)
Dans un autre terminal, lancez l'interface Streamlit :

streamlit run app.py
## 👥 Équipe de Développement
### Asmae HADOUCH & Taha ECHCHOUAL 
