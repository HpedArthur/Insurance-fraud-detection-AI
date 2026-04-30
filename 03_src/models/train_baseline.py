"""
Entrainement de modeles baseline sur les embeddings CLIP.

Compare LogisticRegression, RandomForest, XGBoost.
Sauvegarde le meilleur modele selon ROC-AUC sur generic_val.
Evalue sur generic_test ET domain_test (transferabilite).

Usage :
    python 03_src/models/train_baseline.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(dataset_path: Path, embeddings_path: Path):
    df = pd.read_parquet(dataset_path)
    X_all = np.load(embeddings_path)
    if len(X_all) != len(df):
        raise RuntimeError(f"Mismatch : embeddings={len(X_all)} vs df={len(df)}")
    df = df.reset_index(drop=True)
    return df, X_all


def split_xy(df: pd.DataFrame, X_all: np.ndarray, split_name: str):
    mask = (df["split"] == split_name).values
    return X_all[mask], df.loc[mask, "label"].values, df.loc[mask].reset_index(drop=True)


def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray, label: str = "test") -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "split": label,
        "n": len(y_test),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n--- {name} on {label} ---")
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "  ROC-AUC : N/A")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred) :\n{np.array(metrics['confusion_matrix'])}")
    print(classification_report(y_test, pred, target_names=["real", "fake"]))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Entrainement baseline")
    parser.add_argument("--dataset", type=Path, default=Path("01_data/processed/dataset_with_clip.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("04_models"))
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()

    df, X_all = load_data(args.dataset, args.embeddings)
    logger.info("Dataset : %d lignes, embeddings dim=%d", len(df), X_all.shape[1])

    X_train, y_train, _ = split_xy(df, X_all, "generic_train")
    X_val, y_val, _ = split_xy(df, X_all, "generic_val")
    X_test, y_test, _ = split_xy(df, X_all, "generic_test")
    X_dom_test, y_dom_test, _ = split_xy(df, X_all, "domain_test")

    logger.info("Train : %d (label rep : %s)", len(y_train), np.bincount(y_train).tolist())
    logger.info("Val   : %d", len(y_val))
    logger.info("Test  : %d", len(y_test))
    logger.info("Domain test : %d", len(y_dom_test))

    if not args.no_smote and len(y_train) > 0:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info("Apres SMOTE : %d (label rep : %s)", len(y_train), np.bincount(y_train).tolist())

    models = {
        "logreg": LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42),
        "xgb": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, eval_metric="auc",
            tree_method="hist",
        ),
    }

    all_metrics = []
    val_aucs = {}
    fitted = {}

    for name, model in models.items():
        logger.info("Entrainement %s...", name)
        model.fit(X_train, y_train)
        fitted[name] = model

        m_val = evaluate(name, model, X_val, y_val, label="generic_val")
        m_test = evaluate(name, model, X_test, y_test, label="generic_test")
        if len(y_dom_test) > 0:
            m_dom = evaluate(name, model, X_dom_test, y_dom_test, label="domain_test")
        else:
            m_dom = None

        all_metrics.extend([m_val, m_test] + ([m_dom] if m_dom else []))
        val_aucs[name] = m_val["roc_auc"] or 0.0

    # Sauvegarde du meilleur sur generic_val
    best_name = max(val_aucs, key=val_aucs.get)
    logger.info("Meilleur modele : %s (val ROC-AUC = %.4f)", best_name, val_aucs[best_name])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], args.out_dir / "baseline_best.joblib")
    with (args.out_dir / "baseline_metrics.json").open("w") as f:
        json.dump({"best_model": best_name, "metrics": all_metrics}, f, indent=2)
    logger.info("Modele et metriques sauvegardes dans %s", args.out_dir)


if __name__ == "__main__":
    main()
