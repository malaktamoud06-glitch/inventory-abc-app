# Inventory ABC Decision Lab

Application Streamlit illustrant le devoir d'analyse décisionnelle pour la gestion des stocks multi-attributs.

## Fonctionnalités

- Visualisation de la base `inventory_data.csv`
- Transformation des variables qualitatives selon la Table 1
- Construction des critères agrégés selon la Table 2
- Application de TOPSIS
- Classement des articles et classification ABC
- Entraînement de 4 modèles de Machine Learning :
  - Naive Bayes
  - SVM
  - Random Forest
  - ANN
- Tableau de comparaison Accuracy / Precision / Recall / F1-score
- Matrices de confusion interactives
- Prédiction directe de la classe A/B/C pour un nouvel article
- Illustration d'une généralisation floue de la classification ABC
- Téléchargement de la base enrichie

## Méthodologie implémentée

### 1. Transformation qualitative

Les modalités de `Risk`, `Demand fluctuation`, `Consignment stock` et `Unit size`
sont converties selon les scores normalisés donnés dans le devoir.

### 2. Critères agrégés

- `Criticality = 0.78 × Risk + 0.22 × Demand fluctuation`
- `Demand = 0.71 × Daily usage + 0.29 × Average stock`
- `Supply = 0.75 × Lead time + 0.25 × Consignment stock`

### 3. TOPSIS

Les cinq critères utilisés sont :

- Criticality
- Demand
- Supply
- Unit cost
- Unit size

Poids utilisés :

- Criticality : 0.33
- Demand : 0.15
- Supply : 0.18
- Unit cost : 0.12
- Unit size : 0.22

Ces poids sont repris de l'article de référence, où ils ont été déterminés par AHP.

### 4. ABC

Après tri décroissant du score TOPSIS :

- A : pourcentage cumulé ≤ 20 %
- B : 20 % < pourcentage cumulé ≤ 50 %
- C : pourcentage cumulé > 50 %

### 5. Machine Learning

Les modèles sont entraînés avec les 8 attributs transformés comme variables d'entrée,
et `Classe` comme label.

Le découpage utilisé est proche de celui de l'article :

- ~66,67 % entraînement
- ~33,33 % test
- stratification selon les classes A/B/C
- `random_state=42`

## Lancer dans VS Code

### 1. Ouvrir le dossier

Ouvrir `inventory_abc_app` dans VS Code.

### 2. Créer un environnement virtuel

Sous Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Sous macOS/Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

Puis ouvrir l'adresse affichée dans le terminal, généralement :

`http://localhost:8501`

## Structure

```text
inventory_abc_app/
│
├── app.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   └── inventory_data.csv
│
└── .streamlit/
    └── config.toml
```

## Remarque

L'approche floue proposée est une généralisation illustrative autour des seuils crisp de 20 % et 50 %.
Les zones de transition 15–25 % et 45–55 % sont des choix de modélisation et ne sont pas imposées par le sujet.
