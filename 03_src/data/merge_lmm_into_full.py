"""
Fusion des features BLIP-2 / LLaVA calculees uniquement sur le sous-ensemble
domain (1170 lignes) vers le dataset complet (31062 lignes).

Probleme : extract_blip2.py et extract_llava.py utilisent --limit-to-domain
qui filtre le dataset a 1170 lignes domain et ECRASE le fichier de sortie.
Du coup dataset_with_lmm.parquet ne contient que 1170 lignes, ce qui est
incompatible avec clip_embeddings.npy (31062 vecteurs) lors du train.

Solution : fusionner le subset domain (avec ses features LMM) dans le dataset
complet (31062 lignes). Pour les images NON-domain (CIFAKE/ArtiFact), on
impute des valeurs neutres (0.5 pour les scores, "" pour les captions).

Cela ne pose pas de probleme scientifique :
- variant lite n'utilise PAS BLIP-2/LLaVA, donc rien ne change
- variant full utilise BLIP-2/LLaVA, mais comme les valeurs sont constantes
  (0.5) sur generic_train, l'arbre XGBoost ne splittera pas dessus pour
  generic. En revanche sur domain_test ou les valeurs varient reellement,
  les features LMM seront actives. C'est le comportement souhaite.

Usage :
    python 03_src/data/merge_lmm_into_full.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


LMM_NUMERIC_DEFAULTS = {
    "blip2_score_real": 0.5,
    "blip2_score_ai": 0.5,
    "blip2_score_artifact": 0.5,
    "llava_score_real": 0.5,
    "llava_score_artifact": 0.5,
    "llava_score_coherence": 0.5,
}
LMM_TEXT_DEFAULTS = {
    "blip2_caption": "",
    "llava_observation": "",
}


def main():
    base_path = Path("01_data/processed/dataset_with_clip.parquet")
    domain_lmm_path = Path("01_data/processed/dataset_with_lmm.parquet")
    out_path = Path("01_data/processed/dataset_with_lmm.parquet")

    if not base_path.exists():
        raise SystemExit(f"Fichier manquant : {base_path}. Lance d'abord extract_clip.py.")
    if not domain_lmm_path.exists():
        raise SystemExit(
            f"Fichier manquant : {domain_lmm_path}. Lance d'abord extract_blip2.py et extract_llava.py."
        )

    base = pd.read_parquet(base_path)
    domain = pd.read_parquet(domain_lmm_path)
    logger.info("Base (CLIP, complet) : %d lignes", len(base))
    logger.info("Domain (avec LMM)    : %d lignes", len(domain))

    if len(base) == len(domain):
        logger.info("Tailles identiques, rien a fusionner. Le fichier est deja au bon format.")
        return

    if len(domain) > len(base):
        raise SystemExit(
            f"Le subset domain ({len(domain)}) ne peut pas etre plus grand que la base ({len(base)})."
        )

    # Selection des colonnes LMM disponibles dans domain
    all_lmm_cols = list(LMM_NUMERIC_DEFAULTS) + list(LMM_TEXT_DEFAULTS)
    lmm_cols = [c for c in all_lmm_cols if c in domain.columns]
    logger.info("Colonnes LMM a fusionner : %s", lmm_cols)

    if not lmm_cols:
        logger.warning("Aucune colonne LMM trouvee dans domain. Sortie sans modification.")
        return

    # Index domain par image_path
    if "image_path" not in domain.columns:
        raise SystemExit("Colonne image_path absente de domain LMM.")
    if "image_path" not in base.columns:
        raise SystemExit("Colonne image_path absente de la base.")

    domain_indexed = domain.set_index("image_path")[lmm_cols]
    base_indexed = base.set_index("image_path").copy()

    # Mapper chaque colonne LMM depuis domain ; defaut si absent
    for c in lmm_cols:
        default = LMM_NUMERIC_DEFAULTS.get(c)
        if default is None:
            default = LMM_TEXT_DEFAULTS.get(c, "")
        # Map renvoie NaN pour les image_path non presents dans domain
        mapped = base_indexed.index.to_series().map(domain_indexed[c])
        base_indexed[c] = mapped.fillna(default).values

    out = base_indexed.reset_index()

    # Verifications
    assert len(out) == len(base), f"Taille apres merge incorrecte : {len(out)} != {len(base)}"

    if "split" in out.columns and "blip2_score_real" in out.columns:
        is_domain = out["split"].astype(str).str.startswith("domain_")
        n_domain = int(is_domain.sum())
        # Une ligne domain a un BLIP-2 calcule (donc != 0.5 en general)
        n_with_real_lmm = int(((out["blip2_score_real"] - 0.5).abs() > 1e-6 & is_domain).sum())
        logger.info("Domain avec features LMM reelles : %d / %d", n_with_real_lmm, n_domain)
        logger.info("Total lignes : %d (dont %d domain, %d generic)",
                    len(out), n_domain, len(out) - n_domain)

    out.to_parquet(out_path, index=False)
    logger.info("Sauvegarde : %s (%d lignes, %d colonnes)", out_path, len(out), len(out.columns))


if __name__ == "__main__":
    main()
