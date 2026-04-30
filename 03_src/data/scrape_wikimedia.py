"""
Scraping de Wikimedia Commons pour images de degats habitation.

Avantage Wikimedia : pas besoin d'API key, licences claires (CC0/CC-BY).
On interroge l'API MediaWiki pour les categories pertinentes.

Usage :
    python 03_src/data/scrape_wikimedia.py --out 01_data/raw/wikimedia --max 500
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ISFA-DataMining-Project/1.0 (educational research)"

# Categories Wikimedia Commons par type de degat
CATEGORIES = {
    "water": [
        "Water_damage",
        "Flooded_buildings",
        "Flood_damage",
    ],
    "fire": [
        "Fire_damage",
        "Burned_buildings",
        "House_fires",
    ],
    "glass": [
        "Broken_windows",
        "Broken_glass",
    ],
    "vandalism": [
        "Vandalism",
        "Damaged_buildings",
    ],
}


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def list_category_images(session: requests.Session, category: str, limit: int = 100) -> Iterator[dict]:
    """Genere les images d'une categorie Wikimedia Commons."""
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": min(limit, 500),
    }
    while True:
        r = session.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data.get("query", {}).get("categorymembers", []):
            yield item
        if "continue" in data:
            params.update(data["continue"])
        else:
            break


def get_image_url(session: requests.Session, file_title: str) -> str | None:
    """Recupere l'URL directe du fichier image."""
    params = {
        "action": "query",
        "format": "json",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    r = session.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [])
        if info:
            return info[0].get("url")
    return None


def scrape(out_dir: Path, max_per_cat: int = 500, image_size: int = 1024) -> dict:
    """Scrape Wikimedia Commons pour toutes les categories.

    Returns:
        dict des statistiques par categorie.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = get_session()
    stats = {}

    for category_name, wiki_cats in CATEGORIES.items():
        cat_dir = out_dir / category_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        for wiki_cat in wiki_cats:
            if downloaded >= max_per_cat:
                break
            logger.info("Categorie %s -> %s (telecharges : %d/%d)", category_name, wiki_cat, downloaded, max_per_cat)
            try:
                for item in tqdm(list_category_images(session, wiki_cat, limit=max_per_cat * 2),
                                 desc=wiki_cat, leave=False):
                    if downloaded >= max_per_cat:
                        break
                    title = item["title"]
                    if not title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        continue
                    try:
                        url = get_image_url(session, title)
                        if not url:
                            continue
                        img_resp = session.get(url, timeout=60, stream=True)
                        img_resp.raise_for_status()
                        from io import BytesIO
                        img = Image.open(BytesIO(img_resp.content))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                        out_path = cat_dir / f"wikimedia_{category_name}_{downloaded:04d}.jpg"
                        img.save(out_path, quality=90)
                        downloaded += 1
                        time.sleep(0.5)  # rate limiting poli
                    except Exception as e:
                        logger.debug("Echec sur %s : %s", title, e)
                        continue
            except Exception as e:
                logger.warning("Echec categorie %s : %s", wiki_cat, e)
                continue

        stats[category_name] = downloaded
        logger.info("Categorie %s : %d images", category_name, downloaded)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Scraping Wikimedia Commons")
    parser.add_argument("--out", type=Path, default=Path("01_data/raw/wikimedia"))
    parser.add_argument("--max", type=int, default=300, help="Max d'images par categorie")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    stats = scrape(args.out, max_per_cat=args.max, image_size=args.size)
    total = sum(stats.values())
    print(f"\nTotal : {total} images")
    for cat, n in stats.items():
        print(f"  {cat:12s} : {n}")


if __name__ == "__main__":
    main()
