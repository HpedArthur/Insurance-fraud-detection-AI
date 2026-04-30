"""
Consolidation : parcourt les dossiers d'images et construit dataset.parquet.

Sources scannees :
  - 01_data/external/cifake/{train,test}/{real,fake}/*.jpg
  - 01_data/external/artifact/{real,fake}/*.jpg
  - 01_data/raw/wikimedia/{water,fire,glass,vandalism}/*.jpg     -> label REAL
  - 01_data/raw/pexels/{water,fire,glass,vandalism}/*.jpg        -> label REAL
  - 01_data/synthetic/sdxl/{water,fire,glass,vandalism}/*.jpg    -> label FAKE

Sortie : 01_data/processed/dataset.parquet avec colonnes :
  image_path, label, category, source, generator_model, prompt,
  width, height, exif_present, exif_camera, exif_datetime, has_gps,
  phash, split

Splits :
  - generic_train / generic_val / generic_test : sur cifake + artifact (80/10/10)
  - domain_val / domain_test : sur wikimedia + pexels + sdxl (30/70)

Usage :
    python 03_src/data/build_dataset.py
    python 03_src/data/build_dataset.py --no-phash       # plus rapide, sans hash
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.image_meta import extract_exif, image_dimensions, perceptual_hash  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration des sources : (path_pattern, label, category_func, source, generator, prompt_loader)
SOURCES = [
    # CIFAKE
    {"glob": "01_data/external/cifake/*/real/*.jpg", "label": 0, "category": "generic", "source": "cifake", "generator": None},
    {"glob": "01_data/external/cifake/*/fake/*.jpg", "label": 1, "category": "generic", "source": "cifake", "generator": "stable-diffusion-1.4"},
    # ArtiFact
    {"glob": "01_data/external/artifact/real/*.jpg", "label": 0, "category": "generic", "source": "artifact", "generator": None},
    {"glob": "01_data/external/artifact/fake/*.jpg", "label": 1, "category": "generic", "source": "artifact", "generator": "various"},
    # Wikimedia (reelles habitation)
    {"glob": "01_data/raw/wikimedia/*/*.jpg", "label": 0, "category": "from_path", "source": "wikimedia", "generator": None},
    # Pexels (reelles habitation)
    {"glob": "01_data/raw/pexels/*/*.jpg", "label": 0, "category": "from_path", "source": "pexels", "generator": None},
    # SDXL (synthetiques habitation)
    {"glob": "01_data/synthetic/sdxl/*/*.jpg", "label": 1, "category": "from_path", "source": "sdxl", "generator": "sdxl-turbo"},
]


def load_sdxl_manifest() -> dict:
    """Charge le manifest des prompts SDXL pour enrichir le dataset."""
    manifest_path = Path("01_data/synthetic/sdxl/manifest.json")
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    # Index par chemin relatif
    return {item["path"]: item for item in manifest}


def scan_sources(root: Path, compute_phash: bool = True) -> pd.DataFrame:
    """Scanne toutes les sources et construit un DataFrame brut."""
    sdxl_manifest = load_sdxl_manifest()
    rows = []

    for src in SOURCES:
        paths = sorted(root.glob(src["glob"]))
        logger.info("Source %s : %d fichiers trouves", src["source"], len(paths))

        for path in tqdm(paths, desc=src["source"], leave=False):
            rel_path = str(path.relative_to(root)).replace("\\", "/")
            category = path.parent.name if src["category"] == "from_path" else src["category"]

            w, h = image_dimensions(path)
            exif = extract_exif(path)
            phash = perceptual_hash(path) if compute_phash else None

            sdxl_info = sdxl_manifest.get(rel_path, {})

            rows.append({
                "image_path": rel_path,
                "label": src["label"],
                "category": category,
                "source": src["source"],
                "generator_model": sdxl_info.get("generator") or src["generator"],
                "prompt": sdxl_info.get("prompt"),
                "width": w,
                "height": h,
                "exif_present": exif["exif_present"],
                "exif_camera": exif["camera"],
                "exif_datetime": exif["datetime"],
                "has_gps": exif["has_gps"],
                "phash": phash,
            })

    df = pd.DataFrame(rows)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime doublons exacts (meme phash). Garde une copie."""
    if "phash" not in df or df["phash"].isna().all():
        return df
    before = len(df)
    df = df.drop_duplicates(subset=["phash"], keep="first")
    after = len(df)
    if before > after:
        logger.info("Deduplication : %d -> %d (%d doublons supprimes)", before, after, before - after)
    return df


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assigne les splits selon la categorie et la source."""
    df = df.copy()
    df["split"] = None

    # Etage 1 - generique : 80/10/10
    is_generic = df["category"] == "generic"
    generic = df[is_generic]
    if len(generic) > 0:
        train, temp = train_test_split(generic, test_size=0.2, stratify=generic["label"], random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
        df.loc[train.index, "split"] = "generic_train"
        df.loc[val.index, "split"] = "generic_val"
        df.loc[test.index, "split"] = "generic_test"

    # Etage 2 - domaine habitation : 30/70 (val_domain / test_domain)
    is_domain = ~is_generic
    domain = df[is_domain]
    if len(domain) > 0:
        # Stratifier sur (label, category) pour garder l'equilibre
        domain["_strat"] = domain["label"].astype(str) + "_" + domain["category"].astype(str)
        try:
            val_d, test_d = train_test_split(
                domain, test_size=0.7, stratify=domain["_strat"], random_state=seed
            )
        except ValueError:
            # Fallback si certaines combinaisons trop rares
            val_d, test_d = train_test_split(domain, test_size=0.7, random_state=seed)
        df.loc[val_d.index, "split"] = "domain_val"
        df.loc[test_d.index, "split"] = "domain_test"

    return df


def main():
    parser = argparse.ArgumentParser(description="Construction dataset.parquet")
    parser.add_argument("--root", type=Path, default=Path("."), help="Racine du projet")
    parser.add_argument("--out", type=Path, default=Path("01_data/processed/dataset.parquet"))
    parser.add_argument("--no-phash", action="store_true", help="Skip perceptual hash (plus rapide)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = scan_sources(args.root, compute_phash=not args.no_phash)
    if len(df) == 0:
        logger.warning("Aucune image trouvee. As-tu lance les scripts d'acquisition ?")
        return

    df = deduplicate(df)
    df = assign_splits(df, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Dataset sauvegarde : %s (%d lignes)", args.out, len(df))

    # Summary
    print("\n=== RESUME DATASET ===")
    print(f"Total : {len(df)} images")
    print("\nPar split :")
    print(df["split"].value_counts(dropna=False).to_string())
    print("\nPar source :")
    print(df.groupby(["source", "label"]).size().to_string())
    print("\nPar categorie :")
    print(df.groupby(["category", "label"]).size().to_string())


if __name__ == "__main__":
    main()
