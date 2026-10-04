import numpy as np
import pandas as pd


RISK_MAP = {
    "High": 0.47,
    "Normal": 0.35,
    "Low": 0.18,
}

DEMAND_FLUCTUATION_MAP = {
    "Increasing": 0.36,
    "Stable": 0.28,
    "Unknown": 0.20,
    "Decreasing": 0.16,
    "Ending": 0.00,
}

CONSIGNMENT_MAP = {
    "No": 0.80,
    "Yes": 0.20,
}

UNIT_SIZE_MAP = {
    "Large": 0.53,
    "Medium": 0.31,
    "Small": 0.13,
}


def transform_qualitative(df):
    """Convertit les 4 variables qualitatives selon la Table 1."""
    df["Risk"] = df["Risk"].map(RISK_MAP)
    df["Demand fluctuation"] = df["Demand fluctuation"].map(
        DEMAND_FLUCTUATION_MAP
    )
    df["Consignment stock"] = df["Consignment stock"].map(
        CONSIGNMENT_MAP
    )
    df["Unit size"] = df["Unit size"].map(UNIT_SIZE_MAP)

    if df[
        ["Risk", "Demand fluctuation", "Consignment stock", "Unit size"]
    ].isna().any().any():
        raise ValueError(
            "Une modalité qualitative ne correspond pas aux règles de conversion."
        )

    return df


def build_aggregated_criteria(df):
    """Construit Criticality, Demand et Supply selon la Table 2."""
    df["Criticality"] = (
        0.78 * df["Risk"]
        + 0.22 * df["Demand fluctuation"]
    )

    df["Demand"] = (
        0.71 * df["Daily usage"]
        + 0.29 * df["Average stock"]
    )

    df["Supply"] = (
        0.75 * df["Lead time"]
        + 0.25 * df["Consignment stock"]
    )

    return df


def apply_topsis(df):
    """Applique TOPSIS avec les poids AHP repris de l'article."""
    criteria = [
        "Criticality",
        "Demand",
        "Supply",
        "Unit cost",
        "Unit size",
    ]

    weights = np.array([0.33, 0.15, 0.18, 0.12, 0.22], dtype=float)

    X = df[criteria].to_numpy(dtype=float)

    denominators = np.sqrt((X ** 2).sum(axis=0))
    if np.any(denominators == 0):
        raise ValueError(
            "Impossible de normaliser TOPSIS : une colonne a une norme nulle."
        )

    # Étape 1 : normalisation vectorielle
    R = X / denominators

    # Étape 2 : pondération
    V = R * weights

    # Étape 3 : idéal positif / négatif
    # Dans ce devoir, les 5 critères sont utilisés comme critères croissants
    # d'importance de gestion.
    ideal_pos = V.max(axis=0)
    ideal_neg = V.min(axis=0)

    # Étape 4 : distances
    S_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))

    # Étape 5 : coefficient de proximité
    denom = S_pos + S_neg
    score = np.divide(
        S_neg,
        denom,
        out=np.zeros_like(S_neg),
        where=denom != 0,
    )

    df["TOPSIS_score"] = score

    details = {
        "criteria": criteria,
        "weights": weights,
        "R": R,
        "V": V,
        "ideal_pos": ideal_pos,
        "ideal_neg": ideal_neg,
        "S_pos": S_pos,
        "S_neg": S_neg,
    }

    return df, details


def add_abc_classes(df):
    """Trie par TOPSIS et crée rang, pourcentage cumulé et Classe."""
    df = df.sort_values(
        "TOPSIS_score",
        ascending=False,
    ).copy()

    df["Rang"] = np.arange(1, len(df) + 1)

    df["Pourcentage_cumule"] = (
        df["Rang"] / len(df) * 100
    )

    df["Classe"] = np.select(
        [
            df["Pourcentage_cumule"] <= 20,
            df["Pourcentage_cumule"] <= 50,
        ],
        ["A", "B"],
        default="C",
    )

    return df


def _fuzzy_abc(p):
    # A : plein jusqu'à 15 %, transition vers B entre 15 et 25 %
    if p <= 15:
        mu_a = 1.0
    elif p <= 25:
        mu_a = (25 - p) / 10
    else:
        mu_a = 0.0

    # B : transition depuis A, plein entre 25 et 45 %, transition vers C
    if p <= 15 or p >= 55:
        mu_b = 0.0
    elif p <= 25:
        mu_b = (p - 15) / 10
    elif p <= 45:
        mu_b = 1.0
    else:
        mu_b = (55 - p) / 10

    # C : transition depuis B entre 45 et 55 %, puis plein
    if p <= 45:
        mu_c = 0.0
    elif p <= 55:
        mu_c = (p - 45) / 10
    else:
        mu_c = 1.0

    return mu_a, mu_b, mu_c


def fuzzy_memberships(df):
    """Ajoute les degrés d'appartenance floue A/B/C."""
    values = df["Pourcentage_cumule"].apply(_fuzzy_abc)

    membership_df = pd.DataFrame(
        values.tolist(),
        columns=["Mu_A", "Mu_B", "Mu_C"],
        index=df.index,
    )

    df = pd.concat([df, membership_df], axis=1)

    df["Classe_floue"] = (
        df[["Mu_A", "Mu_B", "Mu_C"]]
        .idxmax(axis=1)
        .str.replace("Mu_", "", regex=False)
    )

    return df
