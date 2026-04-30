"""
Utilitaires d'explicabilite.

Deux approches complementaires :
  1. Occlusion sensitivity sur l'image : on glisse un patch sur l'image, on observe
     la baisse de score. Donne une heatmap des zones critiques. Model-agnostic
     (compatible avec notre pipeline sklearn sur embeddings CLIP).
  2. SHAP TreeExplainer sur les classifieurs XGBoost / RandomForest : explique la
     contribution de chaque feature pour une prediction donnee.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# Occlusion sensitivity
# =============================================================================

def occlusion_heatmap(
    image: Image.Image,
    predict_fn: Callable[[Image.Image], float],
    patch_size: int = 56,
    stride: int = 28,
    fill_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    Calcule une heatmap par occlusion : pour chaque position, on couvre une zone
    de patch_size x patch_size pixels en gris, on appelle predict_fn et on note
    de combien le score chute.

    Args:
        image : PIL.Image RGB.
        predict_fn : fonction (image) -> score de fraude dans [0, 1].
        patch_size : taille du patch d'occlusion (en pixels image).
        stride : pas du sliding window.
        fill_color : couleur de masquage.

    Returns:
        heatmap (H, W) float32 dans [0, 1]. Plus une zone est "rouge", plus le
        modele en dependait pour son score.
    """
    img_rgb = image.convert("RGB")
    W, H = img_rgb.size
    base_score = predict_fn(img_rgb)

    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = img_rgb.copy()
            patch = Image.new("RGB", (patch_size, patch_size), fill_color)
            occluded.paste(patch, (x, y))
            score = predict_fn(occluded)
            delta = base_score - score  # positif si le score baisse a cause de l'occlusion
            heatmap[y:y + patch_size, x:x + patch_size] += delta
            counts[y:y + patch_size, x:x + patch_size] += 1

    counts = np.maximum(counts, 1)
    heatmap = heatmap / counts

    # Normalisation [0, 1] sur la valeur absolue (les zones positives ET negatives
    # sont interessantes : positif = utile pour predire, negatif = trompeur).
    abs_max = np.max(np.abs(heatmap)) or 1.0
    heatmap_norm = (heatmap / abs_max + 1.0) / 2.0  # mapping vers [0, 1] pour viz
    return heatmap_norm


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superpose la heatmap sur l'image. Couleurs : bleu (zone non utilisee) -> rouge (utilisee)."""
    import matplotlib.cm as cm

    img_rgb = image.convert("RGB")
    if heatmap.shape != (img_rgb.height, img_rgb.width):
        from PIL import Image as PILImage
        heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(img_rgb.size, PILImage.Resampling.BILINEAR)
        heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

    cmap = cm.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    base = np.array(img_rgb)
    blended = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(blended)


# =============================================================================
# SHAP wrappers
# =============================================================================

def shap_explain_tree(model, X_sample: np.ndarray, feature_names: list[str], max_display: int = 15):
    """Renvoie un objet shap.Explanation utilisable pour les plots SHAP standard.

    Marche avec XGBoost, RandomForest, LightGBM (TreeExplainer).
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values, explainer


def shap_explain_logreg(model, scaler, X_sample: np.ndarray, feature_names: list[str]):
    """Pour LogisticRegression on utilise LinearExplainer (rapide)."""
    import shap
    explainer = shap.LinearExplainer(model, scaler.transform(X_sample) if scaler else X_sample)
    shap_values = explainer(scaler.transform(X_sample) if scaler else X_sample)
    return shap_values, explainer
