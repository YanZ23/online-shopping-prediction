
"""
Model-training script.

Implements the same pipelines + hyper-parameter grids used in the notebooks:
* Logistic Regression (class_weight="balanced")
* Support Vector Machine  (class_weight="balanced")
* Random Forest (with SMOTE)               – ensemble
* XGBoost     (with SMOTE)                 – ensemble
* Deep Neural Network (Keras Sequential)

All models are persisted to ../models/ as .pkl (sk-learn) or .h5 (Keras).

CLI
---
Train *all* models
> python src/train.py --models all

Train only RF and XGB
> python src/train.py --models rf xgb
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# xgboost
from xgboost import XGBClassifier

# keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "preprocessed_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _load_train_set():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ------------------------------------------------------------------------------
#  classic ML models
# ------------------------------------------------------------------------------
def train_logreg(X_train, y_train):
    pipeline = LogisticRegression(class_weight="balanced",
                                  solver="liblinear", max_iter=1000)
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    gs = GridSearchCV(pipeline, {"C": param_grid["C"]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("[LogReg] best params →", gs.best_params_)
    return gs.best_estimator_


def train_svm(X_train, y_train):
    pipeline = SVC(class_weight="balanced", probability=True)
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }
    gs = GridSearchCV(pipeline, param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("[SVM] best params →", gs.best_params_)
    return gs.best_estimator_


def train_rf(X_train, y_train):
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [5, 10, None],
        "rf__min_samples_split": [2, 5]
    }
    gs = GridSearchCV(pipeline, param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("[RF] best params →", gs.best_params_)
    return gs.best_estimator_


def train_xgb(X_train, y_train):
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("xgb", XGBClassifier(use_label_encoder=False,
                              eval_metric="logloss",
                              random_state=42))
    ])
    param_grid = {
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [3, 5, 8],
        "xgb__learning_rate": [0.01, 0.1, 0.3]
    }
    gs = GridSearchCV(pipeline, param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("[XGB] best params →", gs.best_params_)
    return gs.best_estimator_


# ------------------------------------------------------------------------------
#  Deep Neural Network
# ------------------------------------------------------------------------------
def train_dnn(X_train, y_train):
    tf.random.set_seed(42)
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_split=0.2,
              epochs=100,
              batch_size=32,
              callbacks=[es],
              verbose=1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classification models")
    parser.add_argument("--models", "-m", nargs="+",
                        choices=["logreg", "svm", "rf", "xgb", "dnn", "all"],
                        default=["all"],
                        help="Which models to train")
    args = parser.parse_args()
    models_requested = set(args.models)
    if "all" in models_requested:
        models_requested = {"logreg", "svm", "rf", "xgb", "dnn"}

    X_train, X_test, y_train, y_test = _load_train_set()

    for name in models_requested:
        print(f"\n[▶] Training {name.upper()} ...")
        if name == "logreg":
            model = train_logreg(X_train, y_train)
            joblib.dump(model, MODEL_DIR / "logreg.pkl")
        elif name == "svm":
            model = train_svm(X_train, y_train)
            joblib.dump(model, MODEL_DIR / "svm.pkl")
        elif name == "rf":
            model = train_rf(X_train, y_train)
            joblib.dump(model, MODEL_DIR / "rf.pkl")
        elif name == "xgb":
            model = train_xgb(X_train, y_train)
            joblib.dump(model, MODEL_DIR / "xgb.pkl")
        elif name == "dnn":
            model = train_dnn(X_train, y_train)
            model.save(MODEL_DIR / "dnn_model.h5")
        print(f"[✔] {name.upper()} saved")


if __name__ == "__main__":
    main()