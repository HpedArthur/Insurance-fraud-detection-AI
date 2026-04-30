"""
Telechargement du dataset CIFAKE depuis HuggingFace.

CIFAKE = 60 000 images CIFAR-10 (reelles) + 60 000 generees par Stable Diffusion 1.4.
Reference : Bird & Lotfi (2024), IEEE Access.

Usage :
    python 03_src/data/download_cifake.py --n 1000 --out 01_data/external/cifake
    python 03_src/data/download_cifake.py --n -1 --out 01_data/external/cifake   # tout
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "dragonintelligence/CIFAKE-image-dataset"


def download_cifake(out_dir: Path, n: int = -1, image_size: int = 224) -> dict:
    """Telecharge CIFAKE et sauvegarde en JPG, separe REAL / FAKE.

    Args:
        out_dir : repertoire de sortie (sera cree).
        n : nombre d'images max par split et par classe. -1 = tout.
        image_size : taille de redimensionnement (cote court).

    Returns:
        dict avec les statistiques de telechargement.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Telechargement metadata CIFAKE depuis HuggingFace...")
    ds = load_dataset(DATASET_ID)

    stats = {"train_real": 0, "train_fake": 0, "test_real": 0, "test_fake": 0}

    for split_name in ["train", "test"]:
        split_ds = ds[split_name]
        if n > 0:
            # On limite par classe pour garder l'equilibre
            split_ds = split_ds.shuffle(seed=42).select(range(min(2 * n, len(split_ds))))

        for example in tqdm(split_ds, desc=f"Split {split_name}"):
            label = example["label"]  # 0 = FAKE, 1 = REAL (CIFAKE convention)
            label_name = "real" if label == 1 else "fake"
            key = f"{split_name}_{label_name}"

            if n > 0 and stats[key] >= n:
                continue

            img: Image.Image = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

            sub = out_dir / split_name / label_name
            sub.mkdir(parents=True, exist_ok=True)
            img.save(sub / f"cifake_{stats[key]:06d}.jpg", quality=92)
            stats[key] += 1

    logger.info("Telechargement termine. Stats : %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Telechargement CIFAKE")
    parser.add_argument("--n", type=int, default=1000, help="Nombre d'images par split et par classe (-1 = tout)")
    parser.add_argument("--out", type=Path, default=Path("01_data/external/cifake"), help="Repertoire de sortie")
    parser.add_argument("--size", type=int, default=224, help="Taille de redimensionnement (cote court)")
    args = parser.parse_args()

    stats = download_cifake(args.out, n=args.n, image_size=args.size)
    total = sum(stats.values())
    print(f"\nTotal : {total} images telechargees.")
    print(f"Repartition : {stats}")


if __name__ == "__main__":
    main()
