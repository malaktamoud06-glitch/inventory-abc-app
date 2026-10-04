import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import (
    transform_qualitative,
    build_aggregated_criteria,
    apply_topsis,
    add_abc_classes,
    fuzzy_memberships,
)

# -----------------------------
# Configuration générale
# -----------------------------
st.set_page_config(
    page_title="Inventory ABC Decision Lab",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Style pastel
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #fbfbff 0%, #f7fbff 100%);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        h1, h2, h3 {
            color: #2f3e5c;
        }

        [data-testid="stSidebar"] {
            background: #f2f0ff;
            border-right: 1px solid #e4e1f7;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #eceaf6;
            border-radius: 16px;
            padding: 12px;
            box-shadow: 0 4px 16px rgba(70, 70, 110, 0.05);
        }

        .hero {
            padding: 26px 28px;
            border-radius: 22px;
            background: linear-gradient(135deg, #ebe7ff 0%, #e7f6ff 55%, #eefbf3 100%);
            border: 1px solid #ded9f5;
            margin-bottom: 22px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            color: #283753;
        }

        .hero p {
            margin: 8px 0 0 0;
            color: #53627a;
            font-size: 1.02rem;
        }

        .info-card {
            background: #ffffff;
            border: 1px solid #eceaf6;
            border-radius: 18px;
            padding: 18px;
            height: 100%;
            box-shadow: 0 4px 16px rgba(70, 70, 110, 0.04);
        }

        .soft-blue {
            background: #edf7ff;
            border-left: 5px solid #a9d8f5;
        }

        .soft-green {
            background: #effaf4;
            border-left: 5px solid #a9dfc0;
        }

        .soft-purple {
            background: #f3efff;
            border-left: 5px solid #c6b8f5;
        }

        .soft-pink {
            background: #fff2f6;
            border-left: 5px solid #efbdd0;
        }

        .small-note {
            color: #667085;
            font-size: 0.92rem;
        }

        .prediction-box {
            border-radius: 18px;
            padding: 22px;
            background: linear-gradient(135deg, #f0ecff, #ecf9f2);
            border: 1px solid #dcd6f2;
            margin-top: 14px;
        }

        .class-a {
            background: #f6eefe;
            border: 1px solid #dac9ef;
        }

        .class-b {
            background: #eef7ff;
            border: 1px solid #c9def3;
        }

        .class-c {
            background: #eff9f3;
            border: 1px solid #c9e6d4;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .stButton > button {
            border-radius: 12px;
            border: 0;
            background: #7667b8;
            color: white;
            font-weight: 600;
            padding: 0.55rem 1.1rem;
        }

        .stButton > button:hover {
            background: #6659a5;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Chargement et pipeline MCDM
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/inventory_data.csv")
    df.insert(0, "Article_ID", np.arange(1, len(df) + 1))
    return df


@st.cache_data
def prepare_decision_data(raw_df):
    transformed = transform_qualitative(raw_df.copy())
    aggregated = build_aggregated_criteria(transformed.copy())
    topsis_df, topsis_details = apply_topsis(aggregated.copy())
    abc_df = add_abc_classes(topsis_df.copy())
    abc_df = fuzzy_memberships(abc_df.copy())
    return transformed, aggregated, abc_df, topsis_details


@st.cache_resource
def train_models(ml_df):
    features = [
        "Risk",
        "Demand fluctuation",
        "Average stock",
        "Daily usage",
        "Unit cost",
        "Lead time",
        "Consignment stock",
        "Unit size",
    ]

    X = ml_df[features]
    y = ml_df["Classe"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Naive Bayes": GaussianNB(),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=None,
        ),
        "ANN": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(20,),
                max_iter=1500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
            ),
        ),
    }

    metrics_rows = []
    predictions = {}
    reports = {}
    matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[name] = y_pred
        matrices[name] = confusion_matrix(
            y_test,
            y_pred,
            labels=["A", "B", "C"],
        )

        reports[name] = pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                labels=["A", "B", "C"],
                output_dict=True,
                zero_division=0,
            )
        ).T

        metrics_rows.append(
            {
                "Modèle": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    best_model_name = metrics_df.sort_values(
        "F1-score", ascending=False
    ).iloc[0]["Modèle"]

    return {
        "features": features,
        "models": models,
        "metrics": metrics_df,
        "predictions": predictions,
        "reports": reports,
        "matrices": matrices,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_model_name": best_model_name,
    }


raw_df = load_data()
transformed_df, aggregated_df, decision_df, topsis_details = prepare_decision_data(raw_df)
ml_results = train_models(decision_df)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## Inventory ABC Lab")
st.sidebar.caption("MCDM • TOPSIS • ABC • Machine Learning")

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d'ensemble",
        "1. Base de données",
        "2. Transformation",
        "3. Critères agrégés",
        "4. TOPSIS",
        "5. Classification ABC",
        "6. Machine Learning",
        "7. Prédire un article",
        "8. Approche floue",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Pipeline : données → scores → critères agrégés → TOPSIS → ABC → ML → prédiction."
)

# -----------------------------
# En-tête commun
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Analyse décisionnelle des stocks multi-attributs</h1>
        <p>Application interactive illustrant le devoir : TOPSIS, classification ABC, apprentissage automatique et approche floue.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Vue d'ensemble
# -----------------------------
if page == "Vue d'ensemble":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", len(raw_df))
    c2.metric("Critères initiaux", 8)
    c3.metric("Critères TOPSIS", 5)
    c4.metric("Modèles ML", len(ml_results["models"]))

    st.markdown("### Chaîne de traitement")

    a, b, c, d = st.columns(4)
    with a:
        st.markdown(
            """
            <div class="info-card soft-purple">
                <b>1. Préparation</b><br>
                Conversion des 4 variables qualitatives en scores numériques normalisés.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="info-card soft-blue">
                <b>2. MCDM / TOPSIS</b><br>
                Agrégation des critères, pondération AHP issue de l'article, puis score TOPSIS.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="info-card soft-green">
                <b>3. ABC</b><br>
                Classement décroissant, pourcentage cumulé puis classes A, B et C.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d:
        st.markdown(
            """
            <div class="info-card soft-pink">
                <b>4. Machine Learning</b><br>
                Apprentissage sur les 8 attributs afin de prédire directement la classe d'un nouvel article.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Répartition ABC")
    counts = decision_df["Classe"].value_counts().reindex(["A", "B", "C"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'articles")
    ax.set_title("Répartition des classes ABC")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Meilleur modèle actuel")
    best_name = ml_results["best_model_name"]
    best_row = ml_results["metrics"].set_index("Modèle").loc[best_name]
    st.success(
        f"Le modèle ayant le meilleur F1-score macro sur le jeu de test est **{best_name}** "
        f"(F1 = {best_row['F1-score']:.3f}, Accuracy = {best_row['Accuracy']:.3f})."
    )

# -----------------------------
# Base de données
# -----------------------------
elif page == "1. Base de données":
    st.markdown("## 1. Base de données")
    st.write(
        "La base contient 700 articles décrits par 8 attributs initiaux. "
        "Les variables qualitatives et quantitatives sont affichées ci-dessous."
    )

    type_table = pd.DataFrame(
        {
            "Variable": [
                "Risk",
                "Demand fluctuation",
                "Average stock",
                "Daily usage",
                "Unit cost",
                "Lead time",
                "Consignment stock",
                "Unit size",
            ],
            "Nature": [
                "Qualitative",
                "Qualitative",
                "Quantitative",
                "Quantitative",
                "Quantitative",
                "Quantitative",
                "Qualitative",
                "Qualitative",
            ],
        }
    )

    left, right = st.columns([1, 2])
    with left:
        st.dataframe(type_table, use_container_width=True, hide_index=True)
    with right:
        st.dataframe(raw_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("### Contrôle rapide")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", raw_df.shape[0])
    c2.metric("Colonnes initiales", raw_df.shape[1] - 1)
    c3.metric("Valeurs manquantes", int(raw_df.isna().sum().sum()))

# -----------------------------
# Transformation qualitative
# -----------------------------
elif page == "2. Transformation":
    st.markdown("## 2. Transformation des variables qualitatives")
    st.write(
        "Les modalités linguistiques sont remplacées par les scores normalisés fournis dans la Table 1 du devoir."
    )

    mapping_df = pd.DataFrame(
        [
            ["Risk", "High", 0.47],
            ["Risk", "Normal", 0.35],
            ["Risk", "Low", 0.18],
            ["Demand fluctuation", "Increasing", 0.36],
            ["Demand fluctuation", "Stable", 0.28],
            ["Demand fluctuation", "Unknown", 0.20],
            ["Demand fluctuation", "Decreasing", 0.16],
            ["Demand fluctuation", "Ending", 0.00],
            ["Consignment stock", "No", 0.80],
            ["Consignment stock", "Yes", 0.20],
            ["Unit size", "Large", 0.53],
            ["Unit size", "Medium", 0.31],
            ["Unit size", "Small", 0.13],
        ],
        columns=["Attribut", "Modalité", "Score normalisé"],
    )

    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    st.markdown("### Aperçu après transformation")
    cols = [
        "Article_ID",
        "Risk",
        "Demand fluctuation",
        "Average stock",
        "Daily usage",
        "Unit cost",
        "Lead time",
        "Consignment stock",
        "Unit size",
    ]
    st.dataframe(transformed_df[cols].head(20), use_container_width=True, hide_index=True)

# -----------------------------
# Critères agrégés
# -----------------------------
elif page == "3. Critères agrégés":
    st.markdown("## 3. Construction des critères agrégés")

    st.latex(r"Criticality = 0.78 \times Risk + 0.22 \times Demand\ fluctuation")
    st.latex(r"Demand = 0.71 \times Daily\ usage + 0.29 \times Average\ stock")
    st.latex(r"Supply = 0.75 \times Lead\ time + 0.25 \times Consignment")

    st.info(
        "Ces critères synthétisent les dimensions principales du problème : criticité, demande et approvisionnement."
    )

    display_cols = [
        "Article_ID",
        "Criticality",
        "Demand",
        "Supply",
        "Unit cost",
        "Unit size",
    ]
    st.dataframe(
        aggregated_df[display_cols].head(25),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# TOPSIS
# -----------------------------
elif page == "4. TOPSIS":
    st.markdown("## 4. Application de TOPSIS")
    st.write(
        "TOPSIS recherche les articles proches de la solution idéale positive et éloignés de la solution idéale négative."
    )

    st.markdown("### Poids des critères")
    weights_table = pd.DataFrame(
        {
            "Critère": [
                "Criticality",
                "Demand",
                "Supply",
                "Unit cost",
                "Unit size",
            ],
            "Poids AHP": [0.33, 0.15, 0.18, 0.12, 0.22],
        }
    )
    st.dataframe(weights_table, use_container_width=True, hide_index=True)
    st.caption(
        "Ces poids sont repris de l'article de référence, où ils ont été obtenus par AHP."
    )

    st.markdown("### Étapes")
    st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}")
    st.latex(r"v_{ij}=w_j r_{ij}")
    st.latex(r"RC_i=\frac{S_i^-}{S_i^-+S_i^+}")

    st.markdown("### Matrice normalisée (aperçu)")
    st.dataframe(
        pd.DataFrame(
            topsis_details["R"],
            columns=topsis_details["criteria"],
            index=aggregated_df["Article_ID"],
        ).head(12),
        use_container_width=True,
    )

    st.markdown("### Matrice normalisée et pondérée (aperçu)")
    st.dataframe(
        pd.DataFrame(
            topsis_details["V"],
            columns=topsis_details["criteria"],
            index=aggregated_df["Article_ID"],
        ).head(12),
        use_container_width=True,
    )

    st.markdown("### Scores TOPSIS")
    st.dataframe(
        decision_df[
            ["Article_ID", "TOPSIS_score", "Rang", "Pourcentage_cumule"]
        ].head(30),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# ABC
# -----------------------------
elif page == "5. Classification ABC":
    st.markdown("## 5. Classification ABC")
    st.write(
        "Les articles sont triés par score TOPSIS décroissant. "
        "La classe A correspond aux 20 % premiers, B aux 30 % suivants et C aux 50 % restants."
    )

    st.dataframe(
        decision_df[
            [
                "Article_ID",
                "TOPSIS_score",
                "Rang",
                "Pourcentage_cumule",
                "Classe",
            ]
        ].head(60),
        use_container_width=True,
        hide_index=True,
    )

    counts = decision_df["Classe"].value_counts().reindex(["A", "B", "C"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Classe A", int(counts["A"]))
    c2.metric("Classe B", int(counts["B"]))
    c3.metric("Classe C", int(counts["C"]))

    csv = decision_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger la base enrichie",
        data=csv,
        file_name="inventory_data_with_classes.csv",
        mime="text/csv",
    )

# -----------------------------
# Machine Learning
# -----------------------------
elif page == "6. Machine Learning":
    st.markdown("## 6. Évaluation des modèles de Machine Learning")
    st.write(
        "Les 8 attributs initiaux servent de variables explicatives et la classe ABC sert de label. "
        "La base est séparée en 80 % d'entraînement et 20 % de test, avec stratification."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Train", len(ml_results["X_train"]))
    c2.metric("Test", len(ml_results["X_test"]))
    c3.metric("Meilleur F1", ml_results["best_model_name"])

    st.markdown("### Tableau comparatif")
    metrics_display = ml_results["metrics"].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        metrics_display[col] = metrics_display[col].round(3)

    st.dataframe(
        metrics_display.sort_values("F1-score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Comparaison graphique")
    chart_df = ml_results["metrics"].set_index("Modèle")
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_df[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des performances des modèles")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Indicateurs", loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Matrices de confusion")
    selected_model = st.selectbox(
        "Choisir un modèle",
        list(ml_results["models"].keys()),
    )

    cm = ml_results["matrices"][selected_model]
    fig_cm, ax_cm = plt.subplots(figsize=(5.8, 4.8))
    im = ax_cm.imshow(cm, cmap="Purples")
    ax_cm.set_xticks([0, 1, 2], labels=["A", "B", "C"])
    ax_cm.set_yticks([0, 1, 2], labels=["A", "B", "C"])
    ax_cm.set_xlabel("Classe prédite")
    ax_cm.set_ylabel("Classe réelle")
    ax_cm.set_title(f"Matrice de confusion - {selected_model}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig_cm.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    st.pyplot(fig_cm, clear_figure=True)

    st.markdown("### Rapport par classe")
    report = ml_results["reports"][selected_model].copy()
    st.dataframe(report.round(3), use_container_width=True)

# -----------------------------
# Prédiction d'un nouvel article
# -----------------------------
elif page == "7. Prédire un article":
    st.markdown("## 7. Prédire directement la classe d'un nouvel article")
    st.write(
        "Saisissez les 8 caractéristiques originales de l'article. "
        "L'application applique uniquement le prétraitement nécessaire au modèle ML puis prédit directement A, B ou C."
    )

    best_name = ml_results["best_model_name"]
    st.info(
        f"Par défaut, l'application utilise **{best_name}**, sélectionné selon le meilleur F1-score macro sur le jeu de test."
    )

    model_choice = st.selectbox(
        "Modèle de prédiction",
        list(ml_results["models"].keys()),
        index=list(ml_results["models"].keys()).index(best_name),
    )

    with st.form("new_item_form"):
        c1, c2 = st.columns(2)

        with c1:
            risk = st.selectbox("Risk", ["High", "Normal", "Low"])
            demand_fluctuation = st.selectbox(
                "Demand fluctuation",
                ["Increasing", "Stable", "Unknown", "Decreasing", "Ending"],
            )
            average_stock = st.number_input(
                "Average stock",
                min_value=0.0,
                value=float(raw_df["Average stock"].median()),
                step=1.0,
            )
            daily_usage = st.number_input(
                "Daily usage",
                min_value=0.0,
                value=float(raw_df["Daily usage"].median()),
                step=0.1,
                format="%.3f",
            )

        with c2:
            unit_cost = st.number_input(
                "Unit cost",
                min_value=0.0,
                value=float(raw_df["Unit cost"].median()),
                step=0.1,
                format="%.5f",
            )
            lead_time = st.number_input(
                "Lead time",
                min_value=0,
                value=int(raw_df["Lead time"].median()),
                step=1,
            )
            consignment = st.selectbox(
                "Consignment stock",
                ["No", "Yes"],
            )
            unit_size = st.selectbox(
                "Unit size",
                ["Large", "Medium", "Small"],
            )

        submitted = st.form_submit_button("Prédire la classe")

    if submitted:
        new_raw = pd.DataFrame(
            {
                "Risk": [risk],
                "Demand fluctuation": [demand_fluctuation],
                "Average stock": [average_stock],
                "Daily usage": [daily_usage],
                "Unit cost": [unit_cost],
                "Lead time": [lead_time],
                "Consignment stock": [consignment],
                "Unit size": [unit_size],
            }
        )

        new_numeric = transform_qualitative(new_raw.copy())
        X_new = new_numeric[ml_results["features"]]

        model = ml_results["models"][model_choice]
        predicted_class = model.predict(X_new)[0]

        class_css = {
            "A": "class-a",
            "B": "class-b",
            "C": "class-c",
        }[predicted_class]

        st.markdown(
            f"""
            <div class="prediction-box {class_css}">
                <h2>Classe prédite : {predicted_class}</h2>
                <p>Modèle utilisé : <b>{model_choice}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            classes = model.classes_
            proba_df = pd.DataFrame(
                {
                    "Classe": classes,
                    "Probabilité": proba,
                }
            ).sort_values("Probabilité", ascending=False)

            st.markdown("### Probabilités prédites")
            st.dataframe(proba_df.round(4), use_container_width=True, hide_index=True)

# -----------------------------
# Fuzzy
# -----------------------------
elif page == "8. Approche floue":
    st.markdown("## 8. Généralisation à une approche floue")
    st.write(
        "Au lieu d'une frontière rigide entre A, B et C, chaque article possède des degrés d'appartenance compris entre 0 et 1."
    )

    st.markdown(
        """
        - Transition A → B autour de 20 % : zone 15–25 %
        - Transition B → C autour de 50 % : zone 45–55 %
        """
    )

    display = decision_df[
        [
            "Article_ID",
            "Pourcentage_cumule",
            "Classe",
            "Mu_A",
            "Mu_B",
            "Mu_C",
            "Classe_floue",
        ]
    ].copy()

    st.dataframe(display.head(80), use_container_width=True, hide_index=True)

    article_rank = st.slider(
        "Observer un article selon son rang",
        min_value=1,
        max_value=len(decision_df),
        value=140,
    )

    row = decision_df.iloc[article_rank - 1]

    st.markdown(
        f"**Rang {article_rank} — Pourcentage cumulé : {row['Pourcentage_cumule']:.2f}%**"
    )

    fuzzy_df = pd.DataFrame(
        {
            "Classe": ["A", "B", "C"],
            "Degré": [row["Mu_A"], row["Mu_B"], row["Mu_C"]],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(fuzzy_df["Classe"], fuzzy_df["Degré"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Degré d'appartenance")
    ax.set_title("Appartenance floue de l'article sélectionné")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "Les zones 15–25 % et 45–55 % sont un choix de modélisation illustratif autour des seuils crisp 20 % et 50 %."
    )
