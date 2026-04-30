"""
Telechargement d'un sous-ensemble du dataset ArtiFact depuis HuggingFace.

ArtiFact = images reelles (COCO, FFHQ, ImageNet, AFHQ) + synthetiques de 25 generateurs.
Reference : Rahman et al. (2023), ICIP.

Usage :
    python 03_src/data/download_artifact.py --n 5000 --out 01_data/external/artifact
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

# Plusieurs miroirs ArtiFact existent sur HF. On prend un subset deja prepare.
DATASET_ID = "Hemg/AI-Generated-vs-Real-Images-Datasets"


def download_artifact(out_dir: Path, n: int = 5000, image_size: int = 224) -> dict:
    """Telecharge un sous-ensemble ArtiFact-like.

    Args:
        out_dir : repertoire de sortie.
        n : nombre max d'images par classe.
        image_size : taille redimensionnement.

    Returns:
        dict des stats.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Telechargement %s depuis HuggingFace...", DATASET_ID)
    ds = load_dataset(DATASET_ID, split="train")
    logger.info("Dataset charge : %d exemples au total", len(ds))

    # Inspection structure
    sample = ds[0]
    logger.info("Cles disponibles : %s", list(sample.keys()))

    # Heuristique : trouver la colonne label et image
    label_col = next((c for c in ds.column_names if "label" in c.lower()), None)
    image_col = next((c for c in ds.column_names if "image" in c.lower() or c == "img"), None)
    if label_col is None or image_col is None:
        raise RuntimeError(f"Impossible d'identifier label/image cols : {ds.column_names}")

    ds = ds.shuffle(seed=42)
    stats = {"real": 0, "fake": 0}

    for example in tqdm(ds, desc="ArtiFact"):
        label = example[label_col]
        # Convention typique : 0 = real, 1 = fake (a verifier dans EDA)
        if isinstance(label, str):
            label_name = "real" if label.lower() in ("real", "0", "false") else "fake"
        else:
            label_name = "real" if int(label) == 0 else "fake"

        if stats[label_name] >= n:
            if all(v >= n for v in stats.values()):
                break
            continue

        img = example[image_col]
        if not isinstance(img, Image.Image):
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

        sub = out_dir / label_name
        sub.mkdir(parents=True, exist_ok=True)
        img.save(sub / f"artifact_{stats[label_name]:06d}.jpg", quality=92)
        stats[label_name] += 1

    logger.info("Termine. Stats : %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Telechargement ArtiFact subset")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("01_data/external/artifact"))
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    stats = download_artifact(args.out, n=args.n, image_size=args.size)
    print(f"\nTotal : {sum(stats.values())} images")
    print(f"Repartition : {stats}")


if __name__ == "__main__":
    main()
