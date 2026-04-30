"""
Scraping de Pexels pour images authentiques de degats habitation.

Pexels API gratuite : https://www.pexels.com/api/
Necessite une cle API (gratuite, instantanee).

Usage :
    export PEXELS_API_KEY="ta_cle_ici"   # ou .env
    python 03_src/data/scrape_pexels.py --out 01_data/raw/pexels --per-cat 200
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://api.pexels.com/v1/search"

# Mots-cles par categorie (FR + EN pour maximiser la couverture)
QUERIES = {
    "water": ["flood damage interior", "water damage home", "flooded house", "leak ceiling damage"],
    "fire": ["fire damage interior", "burned house interior", "house fire aftermath", "kitchen fire damage"],
    "glass": ["broken window home", "shattered glass interior", "vandalized window"],
    "vandalism": ["vandalized apartment", "burglary aftermath", "ransacked room", "graffiti interior wall"],
}


def search_pexels(api_key: str, query: str, per_page: int = 80, max_pages: int = 3):
    headers = {"Authorization": api_key}
    for page in range(1, max_pages + 1):
        params = {"query": query, "per_page": per_page, "page": page, "size": "medium"}
        r = requests.get(API_URL, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            logger.warning("Rate limit atteint, pause 60s")
            time.sleep(60)
            continue
        r.raise_for_status()
        data = r.json()
        photos = data.get("photos", [])
        if not photos:
            break
        for photo in photos:
            yield photo
        time.sleep(0.5)


def download_image(url: str, out_path: Path, image_size: int = 1024) -> bool:
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        img.save(out_path, quality=90)
        return True
    except Exception as e:
        logger.debug("Echec download %s : %s", url, e)
        return False


def scrape(out_dir: Path, api_key: str, per_cat: int = 200, image_size: int = 1024) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for category, queries in QUERIES.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        for query in queries:
            if downloaded >= per_cat:
                break
            logger.info("Categorie %s, query : %s", category, query)
            for photo in tqdm(search_pexels(api_key, query), desc=query, leave=False):
                if downloaded >= per_cat:
                    break
                src = photo.get("src", {}).get("large") or photo.get("src", {}).get("medium")
                if not src:
                    continue
                out_path = cat_dir / f"pexels_{category}_{photo['id']}.jpg"
                if out_path.exists():
                    continue
                if download_image(src, out_path, image_size=image_size):
                    downloaded += 1

        stats[category] = downloaded
        logger.info("Categorie %s : %d images", category, downloaded)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Scraping Pexels")
    parser.add_argument("--out", type=Path, default=Path("01_data/raw/pexels"))
    parser.add_argument("--per-cat", type=int, default=200)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        raise SystemExit(
            "PEXELS_API_KEY manquante. Cree un compte gratuit sur https://www.pexels.com/api/ "
            "puis ajoute la cle dans un fichier .env :\n  PEXELS_API_KEY=xxx"
        )

    stats = scrape(args.out, api_key, per_cat=args.per_cat, image_size=args.size)
    total = sum(stats.values())
    print(f"\nTotal : {total} images")
    for cat, n in stats.items():
        print(f"  {cat:12s} : {n}")


if __name__ == "__main__":
    main()
