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

<img width="1602" height="821" alt="Capture d&#39;écran 2026-06-11 001841" src="https://github.com/user-attachments/assets/44362643-4350-4eec-9400-e13f9f806f3d" />

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


### 🔍 Navbar : 3 liens de navigations

<img width="405" height="832" alt="Capture d&#39;écran 2026-06-11 003315" src="https://github.com/user-attachments/assets/f9aea091-7bb0-4e48-852e-6be94494192f" />

---

## 🧠 Performances & Métriques de l'IA

Pour valider scientifiquement notre moteur prédictif avant son déploiement, l'algorithme a été évalué en Validation Croisée (K-Fold, $K=5$) et sur un jeu de test externe masqué. Voici le rapport généré par notre script d'entraînement :

<img width="647" height="343" alt="stat" src="https://github.com/user-attachments/assets/12ea06ff-bf9e-4ea2-9d4d-f09231fc278a" />

*Figure 6 : Capture d'écran des métriques de performance et de coût de l'algorithme.*

### 📊 Résultats & Interprétation du Modèle

Algorithme utilisé : Logistic Regression
Évaluation : Entraînement · Validation Croisée · Jeu de Test Extérieur


## 🎯 Performance Globale
#### J Global (LogLoss total) 0.1749Coût moyen très bas — le modèle est bien calibré et confiant dans ses prédictions
#### J_train (Coût entraînement) 0.1729  Apprentissage efficace
#### J_CV (Coût validation croisée) 0.1783 Très proche de J_train → pas de surapprentissage (overfitting)
#### CV Accuracy 91.23% L'IA classe correctement 9 cas sur 10 sur des données 
#### Taux d'erreur 8.77% Marge d'erreur résiduelle faible, attendue sur données réelles

## 💡 L'écart minime entre J_train (0.1729) et J_CV (0.1783) confirme que le modèle généralise bien — il ne mémorise pas les données d'entraînement mais apprend des patterns réels.


## 🧪 Rapport de Classification — Jeu de Test Extérieur
ClassePrécisionRappelF1-ScoreSupport❌ Fail (0)0.880.850.86264✅ Pass (1)0.930.940.94546

## 🔍 Lecture détaillée par classe
❌ Classe Fail (0) — F1 : 0.86

Précision 0.88 : Quand le modèle prédit un échec, il a raison dans 88% des cas — peu de fausses alarmes.
Rappel 0.85 : Il détecte 85% des vrais échecs — quelques cas critiques peuvent passer inaperçus (15% de faux négatifs).
⚠️ C'est la classe minoritaire (264 cas vs 546), ce qui explique un score légèrement inférieur. À surveiller si le coût d'un faux négatif est élevé dans le contexte métier.

✅ Classe Pass (1) — F1 : 0.94

Précision 0.93 : 93% des prédictions "Pass" sont correctes.
Rappel 0.94 : Le modèle identifie 94% des vrais succès — performance excellente.
La classe majoritaire (546 cas) est très bien apprise, avec un F1 de 0.94 qui témoigne d'un équilibre précision/rappel solide.

---

## 🛠️ Architecture Technique

Le projet est découpé en deux composants autonomes communicant par requêtes HTTP (JSON) :

1. **Le Backend (API) :** Développé avec **FastAPI**. Il charge le modèle de Machine Learning préalablement entraîné (`Logistic Regression` / `Scikit-Learn`), valide les données entrantes avec `Pydantic` et effectue les prédictions de probabilité d'échec en tâche de fond.
2. **Le Frontend (Dashboard) :** Développé en **Streamlit**. Il offre une interface fluide et moderne, interroge l'API pour chaque calcul et s'occupe de la visualisation de données (`Plotly` / `Altair`).

---

## 🚀 Installation et Lancement

### 1. Prérequis & Installation des dépendances
Clonez le projet, créez un environnement virtuel, puis installez les bibliothèques requises :

pip install -r requirements.txt

### 👥 Équipe de Développement

## Asmae HADOUCH & Taha ECHCHOUAL 
