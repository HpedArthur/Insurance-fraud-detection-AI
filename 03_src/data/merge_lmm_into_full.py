"""Fusion LMM features dans dataset complet."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LMM_NUMERIC_DEFAULTS = {
    "blip2_score_real": 0.5, "blip2_score_ai": 0.5, "blip2_score_artifact": 0.5,
    "llava_score_real": 0.5, "llava_score_artifact": 0.5, "llava_score_coherence": 0.5,
}
LMM_TEXT_DEFAULTS = {"blip2_caption": "", "llava_observation": ""}


def main():
    base_path = Path("01_data/processed/dataset_with_clip.parquet")
    domain_lmm_path = Path("01_data/processed/dataset_with_lmm.parquet")
    out_path = Path("01_data/processed/dataset_with_lmm.parquet")

    if not base_path.exists():
        raise SystemExit(f"Fichier manquant : {base_path}")
    if not domain_lmm_path.exists():
        raise SystemExit(f"Fichier manquant : {domain_lmm_path}")

    base = pd.read_parquet(base_path)
    domain = pd.read_parquet(domain_lmm_path)
    logger.info("Base (CLIP, complet) : %d lignes", len(base))
    logger.info("Domain (avec LMM)    : %d lignes", len(domain))

    if len(base) == len(domain):
        logger.info("Tailles identiques, rien a fusionner.")
        return
    if len(domain) > len(base):
        raise SystemExit("Domain plus grand que base, anormal.")

    all_lmm_cols = list(LMM_NUMERIC_DEFAULTS) + list(LMM_TEXT_DEFAULTS)
    lmm_cols = [c for c in all_lmm_cols if c in domain.columns]
    logger.info("Colonnes LMM a fusionner : %s", lmm_cols)
    if not lmm_cols:
        return

    if "image_path" not in domain.columns or "image_path" not in base.columns:
        raise SystemExit("Colonne image_path manquante.")

    domain_indexed = domain.set_index("image_path")[lmm_cols]
    base_indexed = base.set_index("image_path").copy()

    for c in lmm_cols:
        default = LMM_NUMERIC_DEFAULTS.get(c)
        if default is None:
            default = LMM_TEXT_DEFAULTS.get(c, "")
        mapped = base_indexed.index.to_series().map(domain_indexed[c])
        base_indexed[c] = mapped.fillna(default).values

    out = base_indexed.reset_index()
    assert len(out) == len(base)

    out.to_parquet(out_path, index=False)
    logger.info("Sauvegarde : %s (%d lignes, %d colonnes)", out_path, len(out), len(out.columns))

    try:
        if "split" in out.columns and "blip2_score_real" in out.columns:
            is_domain = out["split"].astype(str).str.startswith("domain_")
            n_domain = int(is_domain.sum())
            mask = ((out["blip2_score_real"] - 0.5).abs() > 1e-6) & is_domain
            logger.info("Domain avec LMM reelles : %d / %d", int(mask.sum()), n_domain)
    except Exception as e:
        logger.warning("Stats en echec (non critique) : %s", e)


if __name__ == "__main__":
    main()
