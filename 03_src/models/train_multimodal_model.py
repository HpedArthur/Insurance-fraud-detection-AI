"""
Entrainement du classifieur MULTIMODAL.

Prend en entree :
  - le score image (sortie du classifieur image charge depuis 04_models/image_model_<variant>.joblib)
  - les features texte extraites par extract_text.py
  - quelques features de coherence inter-champs (calculees ici, voir cross-field)

Variantes :
  full  : score image full + texte handcrafted + texte LLM-judge
  lite  : score image lite + texte handcrafted (sans LLM-judge, deployable HF Spaces)

L'idee est d'evaluer le GAIN apporte par le texte sur le scoring final.

LIMITES METHODOLOGIQUES (a documenter dans le rapport)
======================================================
1. Appariement (image, texte) : nous ne disposons pas de vraies declarations
   liees aux photos Wikimedia/SDXL. Pour chaque image domain, on tire un texte
   genere par Mistral parmi les declarations de meme (categorie, label).
   Cela suppose qu'un fraudeur "coherent" aligne son texte avec sa photo.
   Les performances rapportees constituent donc un MAJORANT des perfs reelles,
   ou un texte authentique pourrait etre utilise avec une fausse photo (dossier
   probablement plus difficile a detecter).

2. Imputation : pour les champs texte absents, on utilise des valeurs par defaut
   FIXES (0 pour les compteurs, 0.5 pour les scores) calculees independamment des
   donnees. Cela evite toute fuite val/test -> imputation.

3. Score_image en feature : le modele image qu'on injecte ici a ete entraine
   uniquement sur generic_train (CIFAKE + ArtiFact). Il ne connait pas les images
   domain. Le score_image sur les paires est donc une prediction generalisee, pas
   memorisee.

Usage :
    python 03_src/models/train_multimodal_model.py --variant both
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
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def pair_image_with_text(image_df: pd.DataFrame, text_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Apparie chaque image du domaine habitation avec une declaration texte de meme
    (categorie, label). Pour les images sans declaration possible, on utilise une
    declaration generique vide.

    Cette etape est artificielle : dans la realite chaque dossier a un texte unique.
    Pour le projet pedagogique, on cree des paires en respectant categorie + label.
    """
    domain_df = image_df[image_df["split"].astype(str).str.startswith("domain_")].copy()
    if len(domain_df) == 0:
        raise SystemExit("Aucune image dans le domaine habitation. As-tu telecharge wikimedia/pexels/sdxl ?")

    text_pool = text_df.copy()

    # Strategie : pour chaque (categorie, label), on tire au sort une declaration
    rng = np.random.default_rng(seed)
    paired_rows = []
    for _, img_row in domain_df.iterrows():
        cat = img_row["category"]
        lab = img_row["label"]
        candidates = text_pool[(text_pool["category"] == cat) & (text_pool["label"] == lab)]
        if len(candidates) == 0:
            # fallback : meme label seulement
            candidates = text_pool[text_pool["label"] == lab]
        if len(candidates) == 0:
            # fallback : tout
            candidates = text_pool
        chosen = candidates.iloc[rng.integers(0, len(candidates))]

        new_row = img_row.to_dict()
        for col in ["text"] + HANDCRAFTED_TEXT_COLS + JUDGE_COLS:
            if col in chosen.index:
                new_row[col] = chosen[col]
        paired_rows.append(new_row)

    return pd.DataFrame(paired_rows).reset_index(drop=True)


# Valeurs d'imputation deterministes (independantes des donnees, donc pas de fuite)
# - compteurs txt_n_* : 0 (rien n'a ete detecte)
# - txt_ttr            : 0.5 (richesse vocabulaire moyenne)
# - judge_*            : 0.5 (verdict neutre)
TEXT_DEFAULTS = {
    "txt_n_chars": 0.0, "txt_n_words": 0.0, "txt_n_sentences": 0.0,
    "txt_ttr": 0.5,
    "txt_n_dates": 0.0, "txt_n_money": 0.0, "txt_n_loc": 0.0, "txt_n_persons": 0.0,
    "txt_n_emphatic": 0.0,
    "judge_specificity": 0.5, "judge_coherence": 0.5,
    "judge_plausibility": 0.5, "judge_red_flags": 0.5,
    "judge_overall_genuine": 0.5,
}


def build_X_text(df: pd.DataFrame, variant: str) -> tuple[np.ndarray, list[str]]:
    """Construit la matrice texte. Imputation par valeur fixe (pas de fuite)."""
    cols = HANDCRAFTED_TEXT_COLS.copy()
    if variant == "full":
        cols.extend([c for c in JUDGE_COLS if c in df.columns])

    arrays = []
    names = []
    for c in cols:
        if c in df.columns:
            default = TEXT_DEFAULTS.get(c, 0.0)
            arrays.append(df[c].astype(float).fillna(default).values.reshape(-1, 1))
            names.append(c)
    if not arrays:
        return np.zeros((len(df), 0), dtype=np.float32), []
    return np.hstack(arrays).astype(np.float32), names


def evaluate(name: str, model, X, y, label: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "model": name, "split": label, "n": int(len(y)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    print(f"\n[{name}] {label}  n={len(y)}  ROC-AUC={out['roc_auc']}  F1={out['f1']:.4f}  Acc={out['accuracy']:.4f}")
    print(np.array(out["confusion_matrix"]))
    print(classification_report(y, pred, target_names=["real", "fake"], zero_division=0))
    return out


def train_one_variant(paired: pd.DataFrame, X_clip: np.ndarray, image_model_pkg: dict,
                      variant: str, out_dir: Path, seed: int = 42) -> dict:
    logger.info("=== MULTIMODAL %s ===", variant)
    image_model = image_model_pkg["model"]
    image_scaler = image_model_pkg["scaler"]

    # Calcul du score image pour chaque ligne. On utilise les CLIP embeddings + extras
    # selon ce que connait le scaler de l'image_model. On reconstruit le X_image attendu.
    feature_names = image_model_pkg["feature_names"]
    n_clip = sum(1 for n in feature_names if n.startswith("clip_"))
    extras = [n for n in feature_names if not n.startswith("clip_")]

    extra_arrays = []
    for c in extras:
        if c in paired.columns:
            extra_arrays.append(paired[c].astype(float).fillna(0.5).values.reshape(-1, 1))
        else:
            extra_arrays.append(np.full((len(paired), 1), 0.5, dtype=np.float32))
    X_extra = np.hstack(extra_arrays) if extra_arrays else np.zeros((len(paired), 0), dtype=np.float32)
    X_img_full = np.hstack([X_clip[:len(paired)].astype(np.float32), X_extra.astype(np.float32)])
    X_img_full_s = image_scaler.transform(X_img_full)
    score_image = image_model.predict_proba(X_img_full_s)[:, 1]
    paired = paired.copy()
    paired["score_image"] = score_image

    # Matrice multimodale = score image + features texte
    X_text, text_names = build_X_text(paired, variant)
    X = np.hstack([score_image.reshape(-1, 1), X_text]).astype(np.float32)
    feature_names_mm = ["score_image"] + text_names
    y = paired["label"].values

    # Split val/test sur le domaine habitation, stratifie label
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)

    # On utilise une portion comme train (beaucoup d'overlap avec val par construction).
    # Pour la rigueur, on scinde val en (train, val) interne :
    X_train, X_val_inner, y_train, y_val_inner = train_test_split(X_val, y_val, test_size=0.3,
                                                                   stratify=y_val, random_state=seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val_inner)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          n_jobs=-1, random_state=seed, eval_metric="auc",
                          tree_method="hist", colsample_bytree=0.8)
    model.fit(X_train_s, y_train)

    m_val = evaluate(f"mm_{variant}", model, X_val_s, y_val_inner, "domain_val")
    m_test = evaluate(f"mm_{variant}", model, X_test_s, y_test, "domain_test")

    # Pour comparer au modele image seul, on calcule le score image sur le test
    score_image_test = X_test[:, 0]
    pred_img = (score_image_test >= 0.5).astype(int)
    auc_img_only = float(roc_auc_score(y_test, score_image_test)) if len(np.unique(y_test)) > 1 else None
    f1_img_only = float(f1_score(y_test, pred_img, zero_division=0))
    delta_auc = (m_test["roc_auc"] - auc_img_only) if (m_test["roc_auc"] and auc_img_only) else None
    print(f"\n>> Comparaison sur domain_test :")
    print(f"   image seule  : ROC-AUC={auc_img_only}, F1={f1_img_only:.4f}")
    print(f"   multimodal   : ROC-AUC={m_test['roc_auc']}, F1={m_test['f1']:.4f}")
    print(f"   GAIN AUC     : {delta_auc:+.4f}" if delta_auc is not None else "")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names_mm,
        "variant": variant,
    }, out_dir / f"multimodal_model_{variant}.joblib")

    summary = {
        "variant": variant,
        "metrics": [m_val, m_test],
        "comparison_image_only": {
            "roc_auc": auc_img_only, "f1": f1_img_only,
            "delta_auc_vs_multimodal": delta_auc,
        },
    }
    with (out_dir / f"multimodal_model_{variant}_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Entrainement classifieur MULTIMODAL")
    parser.add_argument("--image-dataset", type=Path,
                        default=Path("01_data/processed/dataset_with_lmm.parquet"))
    parser.add_argument("--text-dataset", type=Path,
                        default=Path("01_data/processed/claim_texts_with_features.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("04_models"))
    parser.add_argument("--variant", choices=["lite", "full", "both"], default="both")
    args = parser.parse_args()

    # Image dataset (fallback ordonne)
    image_candidates = [
        args.image_dataset,
        Path("01_data/processed/dataset_with_blip2.parquet"),
        Path("01_data/processed/dataset_with_clip.parquet"),
        Path("01_data/processed/dataset.parquet"),
    ]
    image_path = next((p for p in image_candidates if p.exists()), None)
    if image_path is None:
        raise SystemExit("Aucun dataset image trouve.")
    if not args.text_dataset.exists():
        raise SystemExit(f"Dataset texte absent : {args.text_dataset}")

    image_df = pd.read_parquet(image_path)
    text_df = pd.read_parquet(args.text_dataset)
    X_clip = np.load(args.embeddings)

    paired = pair_image_with_text(image_df, text_df)
    # Realigner X_clip sur les indices des images dans paired
    # Trick : paired contient image_path, on retrouve l'index du dataset image
    image_df_indexed = image_df.reset_index(drop=True)
    pos_by_path = {p: i for i, p in enumerate(image_df_indexed["image_path"].values)}
    indices = [pos_by_path[p] for p in paired["image_path"].values]
    X_clip_paired = X_clip[indices]
    logger.info("Paires creees : %d", len(paired))

    variants = ["lite", "full"] if args.variant == "both" else [args.variant]
    for v in variants:
        image_model_path = args.out_dir / f"image_model_{v}.joblib"
        if not image_model_path.exists():
            logger.warning("Modele image %s absent, skip variante %s", image_model_path, v)
            continue
        image_model_pkg = joblib.load(image_model_path)
        train_one_variant(paired, X_clip_paired, image_model_pkg, v, args.out_dir)


if __name__ == "__main__":
    main()
