# 🎓 Système de Détection Préventive & Diagnostic du Décrochage Scolaire

Ce projet est une solution complète d'aide à la décision pour les établissements scolaires. Il combine l'analyse de données, le **Machine Learning** pour le calcul du risque d'échec, et une architecture moderne **API-First** (FastAPI) connectée à un **Dashboard interactif** (Streamlit). Le système permet de détecter de manière proactive les élèves en situation de décrochage afin de planifier des actions correctives.

## 🔥 Fonctionnalités Principales

* **Mode API-First :** Séparation stricte entre le moteur d'intelligence artificielle (FastAPI) et l'interface utilisateur.
* **Importation Dynamique :** Possibilité d'importer directement de nouveaux fichiers de données (`.xlsx`, `.xls`, `.txt`) pour mettre à jour la base ou remplacer complètement le fichier existant.
* **Analyse Descriptive Globale :** Statistiques en temps réel sur la promotion (moyenne des absences, notes de mi-parcours, distributions graphiques).
* **Formulaire de Diagnostic à la Volée :** Interface interactive pour simuler ou inscrire un nouvel étudiant en saisissant ses notes par matière et son comportement.
* **Cellule d'Alerte IA :** Identification automatique et instantanée des profils à risque avec calcul de la probabilité d'échec exacte (%).
* **Fiches de Diagnostic Détaillées :** Analyse sectorielle par étudiant mettant en avant les facteurs métiers déclencheurs transmis par l'API (matières critiques, manque d'implication, absentéisme).

---

## 📸 Aperçu du Dashboard (Interface Utilisateur)

### 📊 Page 1 : Statistiques Globales & Importation
Cette première page permet de charger de nouvelles données au format Excel ou texte pour enrichir la base de l'API. Elle affiche instantanément les indicateurs clés de la promotion (KPIs) ainsi que les graphiques de distribution.

<img width="1552" height="847" alt="img1_r" src="https://github.com/user-attachments/assets/29814176-c6cf-44ed-af4f-a4b17294985b" />

*Figure 1 : Première page du dashboard avec module d'importation et graphiques descriptifs.*

---

### 📝 Page 2 : Assistant de Diagnostic & Inscription
Cette interface contient un formulaire dynamique composé de curseurs (sliders) permettant de saisir le comportement (présence, devoirs rendus, heures d'étude) et les notes par matière d'un étudiant pour l'enregistrer et interroger l'API en direct.

<img width="1740" height="813" alt="img2_r" src="https://github.com/user-attachments/assets/4cabe1d7-2828-47e2-ac15-b16d692ee52e" />

*Figure 2 : Formulaire de simulation et d'inscription d'un nouvel étudiant (Mode API).*

---

### 🚨 Page 3 : Cellule de Détection Préventive (Vue d'ensemble)
Cette section liste de manière épurée l'ensemble des profils jugés "défaillants" par le modèle prédictif. Le tableau met en évidence l'ID de l'étudiant, sa probabilité d'échec précise et son volume d'absences.

<img width="1578" height="853" alt="img3_r" src="https://github.com/user-attachments/assets/a96978b0-c5fc-4b1a-b0a7-d052d9369c66" />

*Figure 3 : Tableau récapitulatif des profils à risque identifiés par l'IA.*

---

### 📉 Page 3 : Analyse Sectorielle des Matières Critiques
Directement sous le tableau de la page 3, un graphique en barres permet d'identifier visuellement quelles sont les matières qui mettent le plus la promotion en péril (nombre d'étudiants en situation d'échec par bloc d'enseignement).

<img width="1573" height="688" alt="img4_r" src="https://github.com/user-attachments/assets/6aabdae5-9a55-4980-8d0a-5242c5d521e0" />

*Figure 4 : Statistiques sous format de graphe identifiant les matières critiques de la promotion.*

---

### 🔍 Focus : Fiche de Diagnostic Détaillée (Pop-up API)
Lorsqu'on inspecte un profil critique spécifique, l'API transmits une fiche complète. En plus des notes, elle explicite clairement les **facteurs métiers déclencheurs** (ex: volume d'absences trop élevé, matière critique affectant le score).

<img width="1247" height="837" alt="img5_r" src="https://github.com/user-attachments/assets/f12dc48e-6d29-41e3-92ac-25bc7961088a" />

*Figure 5 : Fenêtre de détails explicitant les causes du danger de décrochage pour un profil ciblé.*

---

## 🧠 Performances & Métriques de l'IA

Pour valider scientifiquement notre moteur prédictif avant son déploiement, l'algorithme a été évalué en Validation Croisée (K-Fold, $K=5$) et sur un jeu de test externe masqué. Voici le rapport généré par notre script d'entraînement :

<img width="638" height="293" alt="Capture d&#39;écran 2026-06-04 233409" src="https://github.com/user-attachments/assets/27bc2cae-1c78-47d0-a3b0-3b31822ca639" />

*Figure 6 : Capture d'écran des métriques de performance et de coût de l'algorithme.*

### 🔍 Interprétation Technique des Résultats

#### 1. Fonctions de Coût ($LogLoss$)
La $LogLoss$ évalue la pénalité des erreurs de probabilité. Plus elle est proche de $0$, plus le modèle est performant et confiant dans ses choix.
* **$J_{\text{GLOBAL}}$ (0.0977) :** Représente l'erreur globale sur l'ensemble de la base.
* **$J_{\text{train}}$ (0.0973) vs $J_{\text{CV}}$ (0.1029) :** L'écart extrêmement faible entre le coût d'entraînement et le coût de validation croisée prouve mathématiquement l'**absence de surapprentissage (overfitting)**. Notre modèle possède une excellente capacité de généralisation sur de futurs étudiants inconnus.

#### 2. Précision Globale vs Erreur
* **Taux de Précision Globale (CV Accuracy) : 98.98%** – L'algorithme classe correctement près de 99 étudiants sur 100 lors des tests croisés.
* **Pourcentage d'Erreur Générale de l'IA : 1.02%** – Le taux d'échec résiduel de la prédiction est minime, ce qui fiabilise grandement l'outil d'aide à la décision.

#### 3. Analyse fine de la Matrice de Classification
Le rapport segmente les performances selon les deux profils réels rencontrés (*support* indique l'effectif testé : 117 cas réels de FAIL, 177 cas réels de PASS) :

* **Classe `FAIL (0)` (Profils en situation de décrochage) :**
  * **Précision (0.97) :** Lorsque l'IA signale qu'un élève va échouer, elle voit juste dans **97%** des cas.
  * **Rappel / Recall (1.00) :** C'est la métrique clé de notre objectif préventif. Le score parfait de **100%** garantit qu'**aucun élève en situation de décrochage n'échappe à la vigilance du système**.

* **Classe `PASS (1)` (Profils en situation de réussite) :**
  * **Précision (1.00) :** Lorsque le modèle prédit la réussite d'un étudiant, sa certitude est absolue (**100%**).
  * **Rappel / Recall (0.98) :** L'IA identifie correctement 98% de la population globale qui va valider son parcours.

* **F1-Score (0.99) :** Moyenne harmonique de la précision et du rappel, cette valeur confirme la robustesse et l'équilibre optimal du modèle sur l'ensemble des cas de figure de l'établissement scolaire.

---

## 🛠️ Architecture Technique

Le projet est découpé en deux composants autonomes communicant par requêtes HTTP (JSON) :

1. **Le Backend (API) :** Développé avec **FastAPI**. Il charge le modèle de Machine Learning préalablement entraîné (`Logistic Regression` / `Scikit-Learn`), valide les données entrantes avec `Pydantic` et effectue les prédictions de probabilité d'échec en tâche de fond.
2. **Le Frontend (Dashboard) :** Développé en **Streamlit**. Il offre une interface fluide et moderne, interroge l'API pour chaque calcul et s'occupe de la visualisation de données (`Plotly` / `Altair`).

---

## 🚀 Installation et Lancement

### 1. Prérequis & Installation des dépendances
Clonez le projet, créez un environnement virtuel, puis installez les bibliothèques requises :

```bash
pip install -r requirements.txt

## 👥 Équipe de Développement

## Asmae HADOUCH & Taha ECHCHOUAL 
