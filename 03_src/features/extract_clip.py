"""
Extraction des embeddings CLIP avec sauvegarde incremental et resume.

Usage :
    python 03_src/features/extract_clip.py --batch 64 --save-every 50
    # Si interrompu, relancer la meme commande pour reprendre
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
def extract_with_resume(model, preprocess, paths, device, out_path: Path,
                          batch_size: int = 32, save_every: int = 50) -> np.ndarray:
    """Extraction CLIP avec sauvegarde tous les save_every batches + resume."""
    n = len(paths)

    # Determiner la dim d'embedding via une 1ere image
    img0 = Image.open(paths[0]).convert("RGB")
    test_feat = model.encode_image(preprocess(img0).unsqueeze(0).to(device))
    emb_dim = test_feat.shape[-1]

    # Charger le partiel existant si valide
    embeddings = None
    done_count = 0
    if out_path.exists():
        try:
            embeddings = np.load(out_path)
            if embeddings.shape == (n, emb_dim):
                norms = np.linalg.norm(embeddings, axis=1)
                done_count = int(np.sum(norms > 0.01))
                logger.info("Resume detecte : %d / %d embeddings deja calcules", done_count, n)
            else:
                logger.warning("Fichier existant taille incorrecte (%s), on recommence", embeddings.shape)
                embeddings = None
        except Exception as e:
            logger.warning("Impossible de charger le partiel : %s", e)
            embeddings = None

    if embeddings is None:
        embeddings = np.zeros((n, emb_dim), dtype=np.float32)

    # Aligner sur la frontiere de batch
    start_idx = (done_count // batch_size) * batch_size
    n_batches = (n + batch_size - 1) // batch_size

    pbar = tqdm(total=n_batches, initial=start_idx // batch_size, desc="CLIP batches")
    saved_since = 0
    for i in range(start_idx, n, batch_size):
        batch_paths = paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                logger.warning("Echec lecture %s : %s", p, e)
                images.append(torch.zeros(3, 224, 224))
        batch = torch.stack(images).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[i:i + len(batch_paths)] = feats.cpu().numpy().astype(np.float32)

        pbar.update(1)
        saved_since += 1
        if saved_since >= save_every:
            np.save(out_path, embeddings)
            saved_since = 0

    pbar.close()
    np.save(out_path, embeddings)
    logger.info("Sauvegarde finale : %s", out_path)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extraction embeddings CLIP avec resume")
    parser.add_argument("--dataset", type=Path,
                        default=Path("01_data/processed/dataset.parquet"))
    parser.add_argument("--out-emb", type=Path,
                        default=Path("04_models/clip_embeddings.npy"))
    parser.add_argument("--out-df", type=Path,
                        default=Path("01_data/processed/dataset_with_clip.parquet"))
    parser.add_argument("--model", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Sauvegarde tous les N batches (defaut 50)")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    logger.info("Dataset : %d lignes", len(df))

    paths = [args.root / p for p in df["image_path"].tolist()]

    model, preprocess, device = load_clip(args.model, args.pretrained)

    args.out_emb.parent.mkdir(parents=True, exist_ok=True)
    embeddings = extract_with_resume(
        model, preprocess, paths, device, args.out_emb,
        batch_size=args.batch, save_every=args.save_every,
    )

    logger.info("Embeddings shape : %s", embeddings.shape)

    df = df.copy()
    df["clip_emb_index"] = np.arange(len(df))
    df.to_parquet(args.out_df, index=False)
    logger.info("Dataset enrichi sauvegarde : %s", args.out_df)


if __name__ == "__main__":
    main()
