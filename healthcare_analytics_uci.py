# -*- coding: utf-8 -*-
"""
Healthcare Analytics: 30-Day Hospital Readmission Prediction
Dataset: UCI Diabetes 130-US Hospitals (1999–2008) [ID: 296]
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Run non-interactively without GUI popups
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# Define output directory for plots and saved models
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def categorize_diagnosis(code):
    """
    Groups raw ICD-9 codes into broader clinical diagnostic categories.
    """
    if pd.isna(code):
        return "Unknown"
    code_str = str(code).strip()
    try:
        num = float(code_str)
    except (ValueError, TypeError):
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif 250 <= num < 251:
        return "Diabetes"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 800 <= num <= 999:
        return "Injury"
    else:
        return "Other"


def load_and_preprocess_data():
    """
    Loads the UCI Diabetes dataset, applies clinical cleaning, resolves patient leakage,
    engineers features, and sets up binary classification target.
    """
    print("=" * 60)
    print("Step 1: Fetching UCI Diabetes dataset (ID: 296)...")
    print("=" * 60)
    dataset = fetch_ucirepo(id=296)

    raw_ids = dataset.data.ids
    raw_X = dataset.data.features
    raw_y = dataset.data.targets

    df = pd.concat([raw_ids, raw_X, raw_y], axis=1)
    print(f"Raw dataset shape: {df.shape}")

    # Standardize missing value representations
    df = df.replace("?", np.nan)

    # 1. Prevent patient-level leakage across encounters:
    # Retain the first encounter per patient so the model doesn't evaluate on the same individual
    initial_rows = len(df)
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    print(f"Deduplicated by patient_nbr: retained {len(df)} encounters (dropped {initial_rows - len(df)} duplicates)")

    # 2. Target Definition:
    # Binary classification: Readmitted within 30 days (<30) = 1, otherwise (>30 or NO) = 0
    df["readmitted_30"] = df["readmitted"].apply(lambda x: 1 if str(x).strip() == "<30" else 0)

    # 3. Save class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="readmitted_30", palette=["#4A90E2", "#E94E77"])
    plt.title("30-Day Hospital Readmission Distribution")
    plt.xlabel("Readmitted within 30 Days (0 = No, 1 = Yes)")
    plt.ylabel("Patient Count")
    dist_path = os.path.join(ARTIFACTS_DIR, "class_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved target distribution plot to: {dist_path}")

    # 4. Remove extremely sparse features and clinical identifiers
    # 'weight', 'payer_code', 'medical_specialty' have high missingness
    columns_to_drop = [
        "weight",
        "payer_code",
        "medical_specialty",
        "encounter_id",
        "patient_nbr",
        "readmitted",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # 5. Convert age intervals to approximate numeric midpoints
    age_mapping = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }
    df["age_numeric"] = df["age"].map(age_mapping)
    df = df.drop(columns=["age"], errors="ignore")

    # 6. Diagnosis feature engineering: categorize ICD-9 codes
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col + "_category"] = df[col].apply(categorize_diagnosis)
            df = df.drop(columns=[col])

    X = df.drop(columns=["readmitted_30"])
    y = df["readmitted_30"]

    print(f"\nFinal feature set: {X.shape[1]} features across {len(X)} records.")
    print(f"Target distribution:\n{y.value_counts(normalize=True).mul(100).round(2).to_dict()}%")

    return X, y


def build_preprocessor(X_train):
    """
    Builds the ColumnTransformer pipeline for numeric scaling and categorical one-hot encoding.
    """
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def train_and_evaluate(X, y):
    """
    Splits data, trains Logistic Regression and Random Forest models with balanced class weights,
    evaluates performance, and generates diagnostic charts.
    """
    print("\n" + "=" * 60)
    print("Step 2: Train/Test Split (Stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)

    # Model 1: Logistic Regression
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )

    # Model 2: Random Forest Classifier
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_pipeline,
    }

    results = {}

    print("\n" + "=" * 60)
    print("Step 3: Training & Evaluating Models...")
    print("=" * 60)

    for name, pipeline in models.items():
        print(f"\n--- Training {name} ---")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "pipeline": pipeline,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": roc,
            },
        }

        print(f"\n{name} Results:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}  (Crucial for catching 30-day readmissions)")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["No/Later (>30)", "Readmitted (<30)"]))

    # Step 4: Visualizations
    print("\n" + "=" * 60)
    print("Step 4: Generating Evaluation Charts...")
    print("=" * 60)

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["No/Later", "Readmitted <30"],
            yticklabels=["No/Later", "Readmitted <30"],
        )
        axes[idx].set_title(f"Confusion Matrix: {name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices to: {cm_path}")

    # 2. ROC Curves Comparison
    plt.figure(figsize=(7, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_score = res["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve: 30-Day Readmission Models")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_path = os.path.join(ARTIFACTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC Curve comparison to: {roc_path}")

    # 3. Top Feature Importances (Random Forest)
    try:
        rf_model = results["Random Forest"]["pipeline"].named_steps["classifier"]
        fitted_prep = results["Random Forest"]["pipeline"].named_steps["preprocessor"]

        cat_encoder = fitted_prep.named_transformers_["cat"].named_steps["encoder"]
        num_cols = fitted_prep.transformers_[0][2]
        cat_cols = fitted_prep.transformers_[1][2]
        encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols)

        all_features = list(num_cols) + list(encoded_cat_cols)
        importances = rf_model.feature_importances_

        feat_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
        top_15 = feat_df.sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_15, x="Importance", y="Feature", palette="viridis")
        plt.title("Top 15 Predictive Features (Random Forest)")
        plt.xlabel("Feature Importance")
        feat_path = os.path.join(ARTIFACTS_DIR, "top_features.png")
        plt.savefig(feat_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Top Feature Importances to: {feat_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

    # Step 5: Save Best Model
    best_model_path = os.path.join(ARTIFACTS_DIR, "best_rf_model.joblib")
    joblib.dump(results["Random Forest"]["pipeline"], best_model_path)
    print(f"\nSaved Best Model Pipeline to: {best_model_path}")

    print("\n" + "=" * 60)
    print("Execution complete! All artifacts and models saved successfully.")
    print("=" * 60)


def main():
    X, y = load_and_preprocess_data()
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
