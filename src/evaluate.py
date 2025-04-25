
"""
Evaluation / visualization script.

Loads saved models, evaluates them on the held-out test set, and writes:
* classification report printed to console
* confusion-matrix PNG
* ROC-curve PNG
*  summary CSV of F1 / Precision / Recall / AUC

CLI
---
Evaluate *all* models
> python src/evaluate.py --models all
"""

import argparse
import csv
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "preprocessed_data.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)


def _load_test_split():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



def _save_cm(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_cm.png")
    plt.close()


def _save_roc(name, y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title(f"{name} – ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_roc.png")
    plt.close()


def evaluate_model(name, model, X, y, is_keras=False):
    if is_keras:
        y_proba = model.predict(X).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

    print(f"\n===== {name} =====")
    print(classification_report(y, y_pred))

    _save_cm(name, y, y_pred)
    _save_roc(name, y, y_proba)

    return {
        "Model": name,
        "F1": f1_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved models")
    parser.add_argument("--models", "-m", nargs="+",
                        choices=["logreg", "svm", "rf", "xgb", "dnn", "all"],
                        default=["all"])
    args = parser.parse_args()
    targets = set(args.models)
    if "all" in targets:
        targets = {"logreg", "svm", "rf", "xgb", "dnn"}

    _, X_test, _, y_test = _load_test_split()
    summary_rows = []

    for model_name in targets:
        print(f"[▶] Evaluating {model_name.upper()}")
        if model_name == "dnn":
            model = load_model(MODEL_DIR / "dnn_model.h5")
            metrics = evaluate_model("Deep Neural Network", model, X_test, y_test, True)
        else:
            model = joblib.load(MODEL_DIR / f"{model_name}.pkl")
            metrics = evaluate_model(model_name.upper(), model, X_test, y_test)
        summary_rows.append(metrics)

    # Save summary CSV
    csv_path = RESULT_DIR / "evaluation_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SUCCESS] Summary CSV saved → {csv_path}")

    # Load back the summary for plotting
    summary_df = pd.read_csv(csv_path)

    # Performance Metrics by Model
    melted_df = pd.melt(
        summary_df,
        id_vars="Model",
        value_vars=["F1", "Precision", "Recall", "AUC"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Performance Metrics by Model")
    plt.ylim(0.5, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "Performance Metrics by Model.png")
    plt.close()

    # Train vs Test Gap (load from hardcoded values or another CSV)
    from sklearn.metrics import f1_score, roc_auc_score

    train_test_metrics = []

    for model_name in targets:
        if model_name == "dnn":
            model = load_model(MODEL_DIR / "dnn_model.h5")

            def pred(X):
                return (model.predict(X).flatten() >= 0.5).astype(int)

            def proba(X):
                return model.predict(X).flatten()
        else:
            model = joblib.load(MODEL_DIR / f"{model_name}.pkl")

            def pred(X):
                return model.predict(X)

            def proba(X):
                return model.predict_proba(X)[:, 1]

        X_train, X_test, y_train, y_test = _load_test_split()

        f1_train = f1_score(y_train, pred(X_train))
        f1_test = f1_score(y_test, pred(X_test))

        auc_train = roc_auc_score(y_train, proba(X_train))
        auc_test = roc_auc_score(y_test, proba(X_test))

        train_test_metrics.append({
            "Model": {
                "logreg": "Logistic Regression",
                "svm": "SVM",
                "rf": "Random Forest",
                "xgb": "XGBoost",
                "dnn": "Deep Neural Network"
            }[model_name],
            "F1 Gap": f1_train - f1_test,
            "AUC Gap": auc_train - auc_test
        })

    df_gap = pd.DataFrame(train_test_metrics)
    melted_gap = pd.melt(df_gap, id_vars="Model", value_vars=["F1 Gap", "AUC Gap"],
                         var_name="Metric", value_name="Gap")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_gap, x="Model", y="Gap", hue="Metric", palette="Set2")
    plt.title("Train vs Test Gap for F1 and AUC by Model")
    plt.ylabel("Gap (Train - Test)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "Train vs Test.png")
    plt.close()

if __name__ == "__main__":
    main()