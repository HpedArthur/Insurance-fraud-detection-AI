"""
Consolidation : parcourt les dossiers d'images et construit dataset.parquet.

Sources scannees :
  - 01_data/external/cifake/{train,test}/{real,fake}/*.jpg
  - 01_data/external/artifact/{real,fake}/*.jpg
  - 01_data/raw/wikimedia/{water,fire,glass,vandalism}/*.jpg     -> label REAL
  - 01_data/raw/pexels/{water,fire,glass,vandalism}/*.jpg        -> label REAL
  - 01_data/raw/personal/{water,fire,glass,vandalism}/*.jpg      -> label REAL (vos photos)
  - 01_data/synthetic/sdxl/{water,fire,glass,vandalism}/*.jpg    -> label FAKE

Sortie : 01_data/processed/dataset.parquet

Splits :
  - generic_train / generic_val / generic_test : sur cifake + artifact (80/10/10)
  - domain_val / domain_test : sur wikimedia + pexels + personal + sdxl (30/70)

Usage :
    python 03_src/data/build_dataset.py
    python 03_src/data/build_dataset.py --no-phash       # plus rapide
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.image_meta import extract_exif, image_dimensions, perceptual_hash  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOURCES = [
    {"glob": "01_data/external/cifake/*/real/*.jpg", "label": 0, "category": "generic", "source": "cifake", "generator": None},
    {"glob": "01_data/external/cifake/*/fake/*.jpg", "label": 1, "category": "generic", "source": "cifake", "generator": "stable-diffusion-1.4"},
    {"glob": "01_data/external/artifact/real/*.jpg", "label": 0, "category": "generic", "source": "artifact", "generator": None},
    {"glob": "01_data/external/artifact/fake/*.jpg", "label": 1, "category": "generic", "source": "artifact", "generator": "various"},
    {"glob": "01_data/raw/wikimedia/*/*.jpg", "label": 0, "category": "from_path", "source": "wikimedia", "generator": None},
    {"glob": "01_data/raw/pexels/*/*.jpg", "label": 0, "category": "from_path", "source": "pexels", "generator": None},
    # Photos perso de l'equipe : 01_data/raw/personal/{water,fire,glass,vandalism}/*
    {"glob": "01_data/raw/personal/*/*.jpg", "label": 0, "category": "from_path", "source": "personal", "generator": None},
    {"glob": "01_data/raw/personal/*/*.jpeg", "label": 0, "category": "from_path", "source": "personal", "generator": None},
    {"glob": "01_data/raw/personal/*/*.png", "label": 0, "category": "from_path", "source": "personal", "generator": None},
    {"glob": "01_data/synthetic/sdxl/*/*.jpg", "label": 1, "category": "from_path", "source": "sdxl", "generator": "sdxl-turbo"},
]


def load_sdxl_manifest():
    manifest_path = Path("01_data/synthetic/sdxl/manifest.json")
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    return {item["path"]: item for item in manifest}


def scan_sources(root, compute_phash=True):
    sdxl_manifest = load_sdxl_manifest()
    rows = []
    for src in SOURCES:
        paths = sorted(root.glob(src["glob"]))
        logger.info("Source %s : %d fichiers", src["source"], len(paths))
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
                "width": w, "height": h,
                "exif_present": exif["exif_present"],
                "exif_camera": exif["camera"],
                "exif_datetime": exif["datetime"],
                "has_gps": exif["has_gps"],
                "phash": phash,
            })
    return pd.DataFrame(rows)


def deduplicate(df):
    if "phash" not in df or df["phash"].isna().all():
        return df
    before = len(df)
    df = df.drop_duplicates(subset=["phash"], keep="first")
    after = len(df)
    if before > after:
        logger.info("Deduplication : %d -> %d (%d doublons)", before, after, before - after)
    return df


def assign_splits(df, seed=42):
    df = df.copy()
    df["split"] = None
    is_generic = df["category"] == "generic"
    generic = df[is_generic]
    if len(generic) > 0:
        train, temp = train_test_split(generic, test_size=0.2, stratify=generic["label"], random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
        df.loc[train.index, "split"] = "generic_train"
        df.loc[val.index, "split"] = "generic_val"
        df.loc[test.index, "split"] = "generic_test"
    is_domain = ~is_generic
    domain = df[is_domain].copy()
    if len(domain) > 0:
        domain["_strat"] = domain["label"].astype(str) + "_" + domain["category"].astype(str)
        try:
            val_d, test_d = train_test_split(domain, test_size=0.7, stratify=domain["_strat"], random_state=seed)
        except ValueError:
            val_d, test_d = train_test_split(domain, test_size=0.7, random_state=seed)
        df.loc[val_d.index, "split"] = "domain_val"
        df.loc[test_d.index, "split"] = "domain_test"
    return df


def main():
    parser = argparse.ArgumentParser(description="Construction dataset.parquet")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("01_data/processed/dataset.parquet"))
    parser.add_argument("--no-phash", action="store_true")
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
