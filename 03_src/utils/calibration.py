"""
Calibration des seuils de decision selon des criteres metier.

Trois strategies disponibles :
  - max_f1                : seuil qui maximise le F1
  - max_recall_at_precision : seuil qui maximise le recall sous contrainte de precision min
  - youden                : maximise sensibilite + specificite (Youden's J)

Sortie : un fichier JSON contenant trois seuils (low / mid / high) pour les decisions
  Legitime / A expertiser / Fraude probable.

Usage :
    python 03_src/utils/calibration.py --variant lite --strategy max_recall_at_precision \
                                       --min-precision 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Seuil qui maximise le F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # precision_recall_curve renvoie 1 element de plus que thresholds
    f1 = f1[:-1]
    if len(f1) == 0:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    idx = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def threshold_max_recall_at_precision(y_true, y_proba, min_precision: float = 0.85) -> dict:
    """Seuil qui maximise le recall sous contrainte precision >= min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]
    valid = precision >= min_precision
    if not valid.any():
        logger.warning("Aucun seuil ne satisfait precision >= %.2f, fallback sur max_f1", min_precision)
        return threshold_max_f1(y_true, y_proba)
    idx_in_valid = np.argmax(recall[valid])
    valid_indices = np.where(valid)[0]
    idx = int(valid_indices[idx_in_valid])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(2 * precision[idx] * recall[idx] / (precision[idx] + recall[idx] + 1e-12)),
        "min_precision_constraint": min_precision,
    }


def threshold_youden(y_true, y_proba) -> dict:
    """Seuil qui maximise Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "threshold": float(thresholds[idx]),
        "tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "youden_j": float(j[idx]),
    }


def calibrate_three_thresholds(y_true, y_proba, strategy: str, min_precision: float = 0.85) -> dict:
    """
    Determine 3 seuils (low / mid / high) pour la decision a 3 niveaux :
      - score < low  : Legitime
      - low <= score < high : A expertiser
      - score >= high : Fraude probable

    low = seuil qui maximise le recall sur la classe LEGITIME (label=0).
          On veut pouvoir auto-valider les dossiers legitimes avec confiance.
    high = seuil qui maximise le recall sur la classe FRAUDE (label=1) sous
           contrainte de precision elevee.
    mid = seuil intermediaire (point d'equilibre F1).
    """
    if strategy == "max_f1":
        mid = threshold_max_f1(y_true, y_proba)
    elif strategy == "youden":
        mid = threshold_youden(y_true, y_proba)
    else:
        mid = threshold_max_recall_at_precision(y_true, y_proba, min_precision=min_precision)

    # Seuil bas : on veut auto-valider sans risque les dossiers legitimes.
    # On cherche le plus grand seuil tel que P(legitime | score < t) >= 0.95
    sorted_proba = np.sort(y_proba)
    low_threshold = 0.1
    for t in sorted_proba:
        below = y_proba < t
        if below.sum() == 0:
            continue
        purity_legit = (y_true[below] == 0).mean()
        if purity_legit >= 0.95:
            low_threshold = float(t)
        else:
            break

    # Seuil haut : symetrique pour la fraude
    high_threshold = 0.9
    for t in sorted_proba[::-1]:
        above = y_proba >= t
        if above.sum() == 0:
            continue
        purity_fraud = (y_true[above] == 1).mean()
        if purity_fraud >= 0.85:
            high_threshold = float(t)
        else:
            break

    # Garde-fou : low < mid < high
    low_threshold = min(low_threshold, mid["threshold"] - 0.05)
    high_threshold = max(high_threshold, mid["threshold"] + 0.05)
    low_threshold = max(0.0, low_threshold)
    high_threshold = min(1.0, high_threshold)

    return {
        "low": low_threshold,
        "mid": mid["threshold"],
        "high": high_threshold,
        "strategy": strategy,
        "mid_metrics": mid,
    }


# =============================================================================
# Pipeline en ligne de commande
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibration des seuils de decision")
    parser.add_argument("--variant", choices=["lite", "full"], default="lite")
    parser.add_argument("--model-type", choices=["image", "multimodal"], default="image")
    parser.add_argument("--model-dir", type=Path, default=Path("04_models"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("01_data/processed/dataset_with_lmm.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--strategy", choices=["max_f1", "max_recall_at_precision", "youden"],
                        default="max_recall_at_precision")
    parser.add_argument("--min-precision", type=float, default=0.85)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.model_type == "image":
        model_path = args.model_dir / f"image_model_{args.variant}.joblib"
    else:
        model_path = args.model_dir / f"multimodal_model_{args.variant}.joblib"
    if not model_path.exists():
        raise SystemExit(f"Modele absent : {model_path}")

    pkg = joblib.load(model_path)
    model = pkg["model"]
    scaler = pkg.get("scaler")
    feature_names = pkg["feature_names"]

    # On evalue sur generic_val (image) ou domain_val (multimodal)
    df_candidates = [
        args.dataset,
        Path("01_data/processed/dataset_with_blip2.parquet"),
        Path("01_data/processed/dataset_with_clip.parquet"),
        Path("01_data/processed/dataset.parquet"),
    ]
    dataset_path = next((p for p in df_candidates if p.exists()), None)
    if dataset_path is None:
        raise SystemExit("Aucun dataset.parquet trouve.")
    df = pd.read_parquet(dataset_path)
    X_clip = np.load(args.embeddings)
    df = df.reset_index(drop=True)

    # Reconstruire X selon les feature_names attendues
    n_clip = sum(1 for n in feature_names if n.startswith("clip_"))
    extras = [n for n in feature_names if not n.startswith("clip_")]
    extra_arrays = []
    for c in extras:
        if c in df.columns:
            extra_arrays.append(df[c].astype(float).fillna(0.5).values.reshape(-1, 1))
        else:
            extra_arrays.append(np.full((len(df), 1), 0.5))
    X_extra = np.hstack(extra_arrays) if extra_arrays else np.zeros((len(df), 0))
    X = np.hstack([X_clip[:, :n_clip].astype(np.float32), X_extra.astype(np.float32)])

    val_split = "generic_val" if args.model_type == "image" else "domain_val"
    mask = (df["split"] == val_split).values
    X_val = X[mask]
    y_val = df.loc[mask, "label"].values

    if len(y_val) == 0:
        logger.warning("Pas de donnees dans %s, fallback sur generic_test", val_split)
        mask = (df["split"] == "generic_test").values
        X_val = X[mask]
        y_val = df.loc[mask, "label"].values

    if scaler:
        X_val_s = scaler.transform(X_val)
    else:
        X_val_s = X_val
    proba = model.predict_proba(X_val_s)[:, 1]

    cal = calibrate_three_thresholds(y_val, proba, strategy=args.strategy, min_precision=args.min_precision)
    logger.info("Calibration : %s", cal)

    out = args.out or args.model_dir / f"thresholds_{args.model_type}_{args.variant}.json"
    with out.open("w") as f:
        json.dump(cal, f, indent=2)
    logger.info("Seuils sauvegardes : %s", out)


if __name__ == "__main__":
    main()
