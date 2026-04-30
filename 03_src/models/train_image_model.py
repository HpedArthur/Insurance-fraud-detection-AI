"""
Entrainement du classifieur IMAGE (axe principal du projet).

On entraine deux variantes :

  full  : utilise CLIP + BLIP-2 + LLaVA + EXIF (riche, pour le rapport)
  lite  : utilise CLIP + EXIF uniquement (deployable sur HF Spaces gratuit)

Compare LogReg / RandomForest / XGBoost. Sauvegarde le meilleur sur generic_val.

Usage :
    python 03_src/models/train_image_model.py --variant full
    python 03_src/models/train_image_model.py --variant lite
    python 03_src/models/train_image_model.py --variant both     # par defaut
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Listes de colonnes par variante (les colonnes manquantes sont ignorees gracefully)
LITE_FEATURES = [
    "exif_present", "has_gps",
]

FULL_EXTRA_FEATURES = [
    "blip2_score_real", "blip2_score_ai", "blip2_score_artifact",
    "llava_score_real", "llava_score_artifact", "llava_score_coherence",
]


def load_data(dataset_path: Path, embeddings_path: Path):
    df = pd.read_parquet(dataset_path)
    X_clip = np.load(embeddings_path)
    if len(X_clip) != len(df):
        raise RuntimeError(f"Mismatch : embeddings={len(X_clip)} vs df={len(df)}")
    return df.reset_index(drop=True), X_clip


def build_features(df: pd.DataFrame, X_clip: np.ndarray, variant: str) -> tuple[np.ndarray, list[str]]:
    """Construit la matrice X selon la variante."""
    cols = LITE_FEATURES.copy()
    if variant == "full":
        for c in FULL_EXTRA_FEATURES:
            if c in df.columns:
                cols.append(c)

    extras = []
    extra_names = []
    for c in cols:
        if c in df.columns:
            v = df[c].astype(float).fillna(0.5).values.reshape(-1, 1)
            extras.append(v)
            extra_names.append(c)
        else:
            logger.warning("Colonne absente, ignoree : %s", c)

    if extras:
        X_extra = np.hstack(extras)
    else:
        X_extra = np.zeros((len(df), 0), dtype=np.float32)

    feature_names = [f"clip_{i}" for i in range(X_clip.shape[1])] + extra_names
    X = np.hstack([X_clip.astype(np.float32), X_extra.astype(np.float32)])
    return X, feature_names


def split(df: pd.DataFrame, X: np.ndarray, name: str):
    mask = (df["split"] == name).values
    return X[mask], df.loc[mask, "label"].values


def evaluate(model_name: str, model, X, y, label: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "model": model_name, "split": label, "n": int(len(y)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "f1": float(f1_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    print(f"\n[{model_name}] {label}  n={len(y)}  ROC-AUC={out['roc_auc']}  F1={out['f1']:.4f}  Acc={out['accuracy']:.4f}")
    print(np.array(out["confusion_matrix"]))
    print(classification_report(y, pred, target_names=["real", "fake"], zero_division=0))
    return out


def train_one_variant(df: pd.DataFrame, X_clip: np.ndarray, variant: str, out_dir: Path,
                      no_smote: bool = False) -> dict:
    logger.info("=== Variante : %s ===", variant)
    X, feature_names = build_features(df, X_clip, variant)
    logger.info("Features : %d (clip=%d + extras=%d)", X.shape[1], X_clip.shape[1], X.shape[1] - X_clip.shape[1])

    X_train, y_train = split(df, X, "generic_train")
    X_val, y_val = split(df, X, "generic_val")
    X_test, y_test = split(df, X, "generic_test")
    X_dom_test, y_dom_test = split(df, X, "domain_test")

    scaler = StandardScaler(with_mean=False)  # CLIP est deja normalise L2, on garde
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    X_dom_s = scaler.transform(X_dom_test) if len(X_dom_test) > 0 else None

    if not no_smote and len(y_train) > 0:
        smote = SMOTE(random_state=42)
        X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
        logger.info("Apres SMOTE : %d (%s)", len(y_train), np.bincount(y_train).tolist())

    models = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=42),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=20, n_jobs=-1, random_state=42),
        "xgb": XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.08,
                             n_jobs=-1, random_state=42, eval_metric="auc",
                             tree_method="hist", colsample_bytree=0.8),
    }

    fitted = {}
    val_aucs = {}
    metrics = []

    for name, m in models.items():
        logger.info("Fit %s...", name)
        m.fit(X_train_s, y_train)
        fitted[name] = m
        m_val = evaluate(name, m, X_val_s, y_val, "generic_val")
        m_test = evaluate(name, m, X_test_s, y_test, "generic_test")
        m_dom = evaluate(name, m, X_dom_s, y_dom_test, "domain_test") if X_dom_s is not None and len(y_dom_test) > 0 else None
        metrics.extend(filter(None, [m_val, m_test, m_dom]))
        val_aucs[name] = m_val["roc_auc"] or 0.0

    best_name = max(val_aucs, key=val_aucs.get)
    logger.info("Meilleur modele variante %s : %s (val AUC = %.4f)", variant, best_name, val_aucs[best_name])

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": fitted[best_name],
        "scaler": scaler,
        "feature_names": feature_names,
        "variant": variant,
        "best_algo": best_name,
    }, out_dir / f"image_model_{variant}.joblib")

    summary = {
        "variant": variant,
        "best_algo": best_name,
        "best_val_auc": val_aucs[best_name],
        "feature_count": len(feature_names),
        "metrics": metrics,
    }
    with (out_dir / f"image_model_{variant}_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Entrainement classifieur IMAGE")
    parser.add_argument("--dataset", type=Path,
                        default=Path("01_data/processed/dataset_with_lmm.parquet"),
                        help="Si absent, fallback vers dataset_with_blip2 puis dataset_with_clip puis dataset.")
    parser.add_argument("--embeddings", type=Path, default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("04_models"))
    parser.add_argument("--variant", choices=["lite", "full", "both"], default="both")
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()

    # Resolution du dataset (fallback ordonne)
    candidates = [
        args.dataset,
        Path("01_data/processed/dataset_with_blip2.parquet"),
        Path("01_data/processed/dataset_with_clip.parquet"),
        Path("01_data/processed/dataset.parquet"),
    ]
    dataset_path = next((p for p in candidates if p.exists()), None)
    if dataset_path is None:
        raise SystemExit("Aucun dataset.parquet trouve. Lance build_dataset.py d'abord.")
    logger.info("Dataset : %s", dataset_path)

    df, X_clip = load_data(dataset_path, args.embeddings)
    logger.info("Total : %d lignes, embeddings dim=%d", len(df), X_clip.shape[1])

    variants = ["lite", "full"] if args.variant == "both" else [args.variant]
    summaries = {}
    for v in variants:
        summaries[v] = train_one_variant(df, X_clip, v, args.out_dir, no_smote=args.no_smote)

    print("\n=== RESUME ===")
    for v, s in summaries.items():
        print(f"  {v}: best={s['best_algo']}, val_auc={s['best_val_auc']:.4f}, features={s['feature_count']}")


if __name__ == "__main__":
    main()
