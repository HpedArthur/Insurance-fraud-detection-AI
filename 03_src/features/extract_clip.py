"""
Extraction des embeddings CLIP pour toutes les images du dataset.

Charge dataset.parquet, calcule l'embedding pour chaque image, sauvegarde un .npy
+ un dataset.parquet enrichi pointant vers les indices.

CLIP utilise : ViT-L/14 OpenAI (768 dimensions).

Usage :
    python 03_src/features/extract_clip.py
    python 03_src/features/extract_clip.py --model "ViT-B-32" --pretrained "openai"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_clip(model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Chargement CLIP %s/%s sur %s...", model_name, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    return model, preprocess, device


@torch.no_grad()
def embed_batch(model, preprocess, paths: list[Path], device: str, batch_size: int = 32) -> np.ndarray:
    """Calcule les embeddings pour une liste de chemins."""
    all_features = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP batches"):
        batch_paths = paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                logger.warning("Echec lecture %s : %s", p, e)
                # Placeholder : zero vector pour garder l'alignement, sera filtre apres
                images.append(torch.zeros(3, 224, 224))
        batch = torch.stack(images).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalisation L2
        all_features.append(feats.cpu().numpy())
    return np.vstack(all_features).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extraction embeddings CLIP")
    parser.add_argument("--dataset", type=Path, default=Path("01_data/processed/dataset.parquet"))
    parser.add_argument("--out-emb", type=Path, default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--out-df", type=Path, default=Path("01_data/processed/dataset_with_clip.parquet"))
    parser.add_argument("--model", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    logger.info("Dataset : %d lignes", len(df))

    paths = [args.root / p for p in df["image_path"].tolist()]

    model, preprocess, device = load_clip(args.model, args.pretrained)
    embeddings = embed_batch(model, preprocess, paths, device, batch_size=args.batch)
    logger.info("Embeddings shape : %s", embeddings.shape)

    args.out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_emb, embeddings)
    logger.info("Embeddings sauvegardes : %s", args.out_emb)

    # Sauvegarde un parquet avec un index sur la position dans le .npy
    df = df.copy()
    df["clip_emb_index"] = np.arange(len(df))
    df.to_parquet(args.out_df, index=False)
    logger.info("Dataset enrichi sauvegarde : %s", args.out_df)


if __name__ == "__main__":
    main()
