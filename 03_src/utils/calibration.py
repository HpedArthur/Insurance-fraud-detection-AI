"""
Calibration des seuils de decision selon des criteres metier.

Trois strategies disponibles :
  - max_f1                  : seuil qui maximise le F1
  - max_recall_at_precision : seuil qui maximise le recall sous contrainte
                              de precision min
  - youden                  : maximise sensibilite + specificite (Youden's J)

Sortie : un fichier JSON contenant trois seuils (low / mid / high) pour les
decisions Legitime / A expertiser / Fraude probable.

Pour le mode multimodal, ce script reconstruit les features comme le fait
train_multimodal_model.py : il joint le dataset image avec le dataset texte
sur (category, label) et calcule le score_image via le modele image charge.
Cela evite l'erreur frequente d'imputer les features texte par 0.5 (qui
fausserait completement la recherche du seuil).

Usage :
    python 03_src/utils/calibration.py --variant lite --model-type multimodal \
        --strategy max_recall_at_precision --min-precision 0.85
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
    f1_score,
    precision_recall_curve,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HANDCRAFTED_TEXT_COLS = [
    "txt_n_chars", "txt_n_words", "txt_n_sentences", "txt_ttr",
    "txt_n_dates", "txt_n_money", "txt_n_loc", "txt_n_persons", "txt_n_emphatic",
]
JUDGE_COLS = [
    "judge_specificity", "judge_coherence", "judge_plausibility",
    "judge_red_flags", "judge_overall_genuine",
]


def threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Seuil qui maximise le F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
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
    """Seuil qui maximise Youden's J = TPR - FPR.

    Patch : roc_curve renvoie un seuil "inf" en tete de liste pour le point
    (0,0) ; il faut l'exclure pour que np.argmax renvoie un seuil valide
    coherent avec l'optimum de J.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]
    if len(thresholds) == 0:
        return {"threshold": 0.5, "tpr": 0.0, "fpr": 0.0, "youden_j": 0.0}
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
      - score < low          : Legitime
      - low <= score < high  : A expertiser
      - score >= high        : Fraude probable

    low  = plus grand seuil tel que P(legitime | score < t) >= 0.95
    high = plus petit seuil tel que P(fraude   | score >= t) >= 0.85
    mid  = selon la strategie (max_f1 / max_recall_at_precision / youden)
    """
    if strategy == "max_f1":
        mid = threshold_max_f1(y_true, y_proba)
    elif strategy == "youden":
        mid = threshold_youden(y_true, y_proba)
    else:
        mid = threshold_max_recall_at_precision(y_true, y_proba, min_precision=min_precision)

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
# Construction des features attendues par le modele
# =============================================================================

def pair_image_with_text_for_calibration(image_df: pd.DataFrame, text_df: pd.DataFrame,
                                          seed: int = 42) -> pd.DataFrame:
    """Reproduit l'appariement de train_multimodal_model.py de maniere deterministe."""
    domain_df = image_df[image_df["split"].astype(str).str.startswith("domain_")].copy()
    if len(domain_df) == 0:
        raise SystemExit("Pas d'images domain pour calibrer.")
    rng = np.random.default_rng(seed)
    paired_rows = []
    for _, img_row in domain_df.iterrows():
        cat = img_row["category"]
        lab = img_row["label"]
        candidates = text_df[(text_df["category"] == cat) & (text_df["label"] == lab)]
        if len(candidates) == 0:
            candidates = text_df[text_df["label"] == lab]
        if len(candidates) == 0:
            candidates = text_df
        chosen = candidates.iloc[rng.integers(0, len(candidates))]
        new_row = img_row.to_dict()
        for col in ["text"] + HANDCRAFTED_TEXT_COLS + JUDGE_COLS:
            if col in chosen.index:
                new_row[col] = chosen[col]
        paired_rows.append(new_row)
    return pd.DataFrame(paired_rows).reset_index(drop=True)


def build_X_image(df: pd.DataFrame, X_clip: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Reconstruit la matrice X attendue par un modele image, en respectant l'ordre."""
    n_clip = sum(1 for n in feature_names if n.startswith("clip_"))
    extras = [n for n in feature_names if not n.startswith("clip_")]
    extra_arrays = []
    for c in extras:
        if c in df.columns:
            extra_arrays.append(df[c].astype(float).fillna(0.5).values.reshape(-1, 1))
        else:
            extra_arrays.append(np.full((len(df), 1), 0.5))
    X_extra = np.hstack(extra_arrays) if extra_arrays else np.zeros((len(df), 0))
    return np.hstack([X_clip[:, :n_clip].astype(np.float32), X_extra.astype(np.float32)])


def build_X_multimodal(paired: pd.DataFrame, X_clip_paired: np.ndarray,
                       image_pkg: dict, mm_feature_names: list[str]) -> np.ndarray:
    """Reconstruit la matrice X multimodale exactement comme au training :
    score_image (calcule via le modele image) + features texte ordonnees."""
    image_model = image_pkg["model"]
    image_scaler = image_pkg["scaler"]
    X_image = build_X_image(paired, X_clip_paired, image_pkg["feature_names"])
    X_image_s = image_scaler.transform(X_image)
    score_image = image_model.predict_proba(X_image_s)[:, 1]

    text_arrays = []
    for c in mm_feature_names:
        if c == "score_image":
            text_arrays.append(score_image.reshape(-1, 1))
        elif c in paired.columns:
            text_arrays.append(paired[c].astype(float).fillna(0.5).values.reshape(-1, 1))
        else:
            logger.warning("Feature multimodale absente : %s -> impute 0.5", c)
            text_arrays.append(np.full((len(paired), 1), 0.5))
    return np.hstack(text_arrays).astype(np.float32)


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
    parser.add_argument("--text-dataset", type=Path,
                        default=Path("01_data/processed/claim_texts_with_features.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--strategy", choices=["max_f1", "max_recall_at_precision", "youden"],
                        default="max_recall_at_precision")
    parser.add_argument("--min-precision", type=float, default=0.85)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
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

    # Resolution du dataset image (fallback ordonne)
    df_candidates = [
        args.dataset,
        Path("01_data/processed/dataset_with_blip2.parquet"),
        Path("01_data/processed/dataset_with_clip.parquet"),
        Path("01_data/processed/dataset.parquet"),
    ]
    dataset_path = next((p for p in df_candidates if p.exists()), None)
    if dataset_path is None:
        raise SystemExit("Aucun dataset image trouve.")
    logger.info("Dataset image : %s", dataset_path)
    image_df = pd.read_parquet(dataset_path).reset_index(drop=True)
    X_clip = np.load(args.embeddings)

    if args.model_type == "image":
        # Calibration sur generic_val (le modele image y est entraine)
        X_full = build_X_image(image_df, X_clip, feature_names)
        mask = (image_df["split"] == "generic_val").values
        if not mask.any():
            logger.warning("Pas de donnees dans generic_val, fallback sur generic_test")
            mask = (image_df["split"] == "generic_test").values
        X_val = X_full[mask]
        y_val = image_df.loc[mask, "label"].values

    else:
        # Multimodal : il faut joindre image + texte comme au training
        if not args.text_dataset.exists():
            raise SystemExit(f"Dataset texte absent : {args.text_dataset}")
        text_df = pd.read_parquet(args.text_dataset)

        paired = pair_image_with_text_for_calibration(image_df, text_df, seed=args.seed)
        # Realigner X_clip sur les images paires
        pos_by_path = {p: i for i, p in enumerate(image_df["image_path"].values)}
        indices = [pos_by_path[p] for p in paired["image_path"].values]
        X_clip_paired = X_clip[indices]

        # Charger le modele image associe pour calculer score_image proprement
        image_pkg_path = args.model_dir / f"image_model_{args.variant}.joblib"
        if not image_pkg_path.exists():
            raise SystemExit(f"Modele image absent (necessaire pour multimodal) : {image_pkg_path}")
        image_pkg = joblib.load(image_pkg_path)

        X_full = build_X_multimodal(paired, X_clip_paired, image_pkg, feature_names)

        # Refaire le meme split que le training (graine identique) pour avoir
        # le set de validation interne du multimodal.
        from sklearn.model_selection import train_test_split
        y_paired = paired["label"].values
        # 1er split : moitie train_pool / moitie test
        X_pool, _, y_pool, _ = train_test_split(
            X_full, y_paired, test_size=0.5, stratify=y_paired, random_state=args.seed
        )
        # 2eme split sur le pool : 70% train / 30% val
        _, X_val, _, y_val = train_test_split(
            X_pool, y_pool, test_size=0.3, stratify=y_pool, random_state=args.seed
        )
        logger.info("Validation multimodale reconstruite : n=%d", len(y_val))

    if scaler is not None:
        X_val_s = scaler.transform(X_val)
    else:
        X_val_s = X_val
    proba = model.predict_proba(X_val_s)[:, 1]

    # Garde-fou : on impose un minimum de variance dans les scores
    if proba.std() < 1e-3:
        logger.warning("Distribution des scores quasi-constante (std=%.5f) : la calibration "
                       "ne sera pas pertinente. Verifier le modele et les features.", proba.std())

    cal = calibrate_three_thresholds(y_val, proba, strategy=args.strategy,
                                      min_precision=args.min_precision)
    logger.info("Calibration : %s", cal)

    out = args.out or args.model_dir / f"thresholds_{args.model_type}_{args.variant}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)
    logger.info("Seuils sauvegardes : %s", out)


if __name__ == "__main__":
    main()
