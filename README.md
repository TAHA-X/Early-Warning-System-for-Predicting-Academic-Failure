# 🎓 Early Warning System (EWS) - Pilotage de la Réussite Académique

L'**Early Warning System (EWS)** est une application web d'aide à la décision et de diagnostic prédictif conçue pour les administrations universitaires. Grâce à une approche hybride combinant la puissance de l'**Intelligence Artificielle** (Machine Learning) et la clarté d'un **système de diagnostic métier**, EWS permet de détecter en amont les étudiants à risque de décrochage ou d'échec scolaire.

---

## 🚀 Architecture Globale du Projet

Le projet repose sur une architecture découplée et unifiée :
1. **La Base de Données Globale (`data.csv` & `etudiants.csv`)** : Séparation stricte entre les données historiques d'entraînement (`data.csv`) et la base des étudiants actifs inscrits au cours de l'année (`etudiants.csv`).
2. **Le Moteur Prédictif (FastAPI)** : Une API REST intégrée qui charge un modèle de **Régression Logistique** (scikit-learn) entraîné avec un équilibrage des classes (`class_weight='balanced'`) pour évaluer scientifiquement le risque.
3. **L'Interface Utilisateur (Streamlit)** : Un tableau de bord moderne, interactif et multi-pages structuré en 3 espaces de navigation.

---

## 🛠️ Fonctionnalités Clés & Aperçu de l'Interface

L'interface Streamlit est segmentée en trois volets décisionnels majeurs :

### 1. 📊 Tableau de Bord des Étudiants Inscrits
* **Analyse Descriptive** : Calcul en direct des indicateurs macro de la promotion (Effectif total, moyenne des absences, moyenne générale de mi-parcours) à partir du fichier de suivi administratif `etudiants.csv`.
* **Visualisations Dynamiques** : Graphiques interactifs affichant la distribution des absences et la corrélation entre le volume d'étude hebdomadaire et les notes obtenues.

![Aperçu du Tableau de Bord Statistique]

<img width="1822" height="870" alt="im1" src="https://github.com/user-attachments/assets/9a011af1-8398-4fb5-aefb-06a114e13db0" />


### 2. 🤖 Assistant de Diagnostic & Inscription
* **Sécurité Anti-Doublon Stricte** : Algorithme de vérification instantanée lors de la soumission. Si l'ID de l'étudiant existe déjà dans la base, le système bloque l'enregistrement pour préserver l'intégrité des données.
* **Évaluation Hybride** : L'IA détient le pouvoir décisionnel exclusif (`PASS` ou `FAIL`) en analysant de manière non linéaire les données comportementales. Si le modèle prédit un échec, le système applique un filtre de diagnostic déterministe pour lister l'ensemble complet des causes comportementales et des matières défaillantes (Note < 50/100).

![Formulaire d'Inscription et de Diagnostic]

<img width="1482" height="780" alt="img2" src="https://github.com/user-attachments/assets/9c8f8cbc-5ad7-4189-874f-834a70757c83" />


### 3. 🚨 Cellule de Détection Préventive (Risques & Matières)
* **Détection Globale en Masse** : Scan complet du fichier administratif `etudiants.csv` à l'aide du modèle prédictif pour extraire instantanément et isoler la liste de tous les profils en danger avec leur probabilité d'échec exacte.
* **Identification Sectorielle des Matières** : Repérage automatique, pour chaque étudiant en échec, des matières spécifiques en dessous du seuil critique (notes < 50/100), complété par un graphique à barres global mettant en évidence les enseignements qui mettent la promotion en péril.

![Cellule de Détection Préventive - Liste des Profils]

<img width="1503" height="881" alt="img3" src="https://github.com/user-attachments/assets/69fd7f4d-c11d-4723-80f0-48696836653b" />


![Analyse Sectorielle des Matières à Risque]

<img width="1455" height="697" alt="img4" src="https://github.com/user-attachments/assets/cb528234-08b0-4140-a0c3-a3b6ceba99ed" />


## 📦 Installation et Lancement du Projet

### Prérequis
* Python 3.10 ou supérieur

### 1. Installation des dépendances
Ouvrez votre terminal et installez les bibliothèques requises :
```bash
pip install fastapi uvicorn pandas scikit-learn streamlit plotly requests
