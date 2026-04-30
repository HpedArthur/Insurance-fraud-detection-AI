"""
Generation d'images synthetiques avec Stable Diffusion XL Turbo.

Necessite un GPU (Colab T4 ou local CUDA). Sur CPU, indecemment lent.

Usage :
    python 03_src/data/generate_sdxl.py --per-cat 100 --out 01_data/synthetic/sdxl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "stabilityai/sdxl-turbo"

# Prompts varies par categorie pour eviter le sur-apprentissage sur 1 prompt
PROMPTS = {
    "water": [
        "interior of a flooded apartment, water damage on wooden floor, photorealistic, dslr photo, natural light",
        "leaking ceiling in a living room, water stains on walls, realistic interior photography",
        "soaked carpet and damaged furniture after pipe burst, indoor home photo",
        "flooded basement with floating debris, photorealistic insurance claim documentation",
        "water damaged kitchen with peeling wallpaper, daylight, realistic photo",
    ],
    "fire": [
        "burned kitchen after fire, soot on walls, charred furniture, insurance claim photo",
        "fire damaged bedroom with melted plastic and ash, realistic interior, daylight",
        "scorched living room walls black with smoke residue, photographic realism",
        "burnt staircase in a house after a fire, insurance documentation photo",
        "fire damaged office with destroyed desk and equipment, realistic photo",
    ],
    "glass": [
        "broken window in a living room, glass shards on floor, after burglary, daylight, realistic",
        "shattered glass door in modern home, photorealistic interior",
        "smashed window pane on wooden frame, indoor photo, realistic lighting",
        "vandalized storefront window with cracks, photorealistic",
        "broken bathroom mirror, glass fragments in sink, realistic photo",
    ],
    "vandalism": [
        "vandalized apartment interior, overturned furniture, graffiti on wall",
        "ransacked bedroom with drawers pulled out, clothes everywhere, realistic photo",
        "burglarized living room, broken lamp, scattered objects on floor, daylight",
        "graffiti tags on white interior wall in abandoned house, photorealistic",
        "destroyed kitchen with broken plates and overturned chairs, realistic interior",
    ],
}


def load_pipeline(device: str = "cuda"):
    logger.info("Chargement du pipeline SDXL Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    return pipe


def generate(pipe, prompt: str, seed: int, image_size: int = 512) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,        # SDXL Turbo : 1 a 4 steps suffisent
        guidance_scale=0.0,           # SDXL Turbo : pas de CFG
        height=image_size,
        width=image_size,
        generator=generator,
    ).images[0]
    return image


def run(out_dir: Path, per_cat: int, image_size: int = 512, seed_base: int = 1000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("Pas de GPU detectee. La generation sera tres lente.")
    pipe = load_pipeline(device)

    manifest = []
    stats = {}

    for category, prompts in PROMPTS.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        seed = seed_base

        with tqdm(total=per_cat, desc=f"SDXL {category}") as pbar:
            while generated < per_cat:
                prompt = prompts[generated % len(prompts)]
                try:
                    img = generate(pipe, prompt, seed=seed, image_size=image_size)
                    out_path = cat_dir / f"sdxl_{category}_{generated:04d}.jpg"
                    img.save(out_path, quality=92)
                    manifest.append({
                        "path": str(out_path.relative_to(out_dir.parent.parent)),
                        "category": category,
                        "prompt": prompt,
                        "seed": seed,
                        "generator": "sdxl-turbo",
                    })
                    generated += 1
                    seed += 1
                    pbar.update(1)
                except Exception as e:
                    logger.warning("Echec generation : %s", e)
                    seed += 1
                    continue

        stats[category] = generated

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Manifest sauvegarde : %s", manifest_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generation SDXL Turbo")
    parser.add_argument("--per-cat", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("01_data/synthetic/sdxl"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args()

    stats = run(args.out, per_cat=args.per_cat, image_size=args.size, seed_base=args.seed_base)
    total = sum(stats.values())
    print(f"\nTotal : {total} images generees")
    for cat, n in stats.items():
        print(f"  {cat:12s} : {n}")


if __name__ == "__main__":
    main()
