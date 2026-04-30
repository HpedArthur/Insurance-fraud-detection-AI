"""
Scraping de Wikimedia Commons pour images de degats habitation.

Avec retry exponentiel sur 429 et User-Agent compliant.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ISFA-DataMining-Bot/1.0 (https://github.com/HpedArthur; educational use)"

CATEGORIES = {
    "water": ["Water_damage", "Flooded_buildings", "Flood_damage", "Water_damage_to_homes"],
    "fire": ["Fire_damage", "Burned_buildings", "House_fires", "Building_fires"],
    "glass": ["Broken_windows", "Broken_glass"],
    "vandalism": ["Vandalism", "Damaged_buildings"],
}


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip",
    })
    return s


def request_with_retry(session, url, params=None, max_retries=5, base_delay=2.0):
    """Requete avec retry exponentiel sur 429 et 503."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code == 503:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                logger.info("Status %d, retry dans %.1fs (tentative %d/%d)",
                            r.status_code, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
            logger.debug("Erreur reseau %s, retry dans %.1fs", e, wait)
            time.sleep(wait)
    logger.warning("Toutes les retries ont echoue pour %s", url)
    return None


def list_category_images(session, category, limit=100):
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": min(limit, 500),
    }
    while True:
        r = request_with_retry(session, API_URL, params=params)
        if r is None:
            return
        data = r.json()
        for item in data.get("query", {}).get("categorymembers", []):
            yield item
        if "continue" in data:
            params.update(data["continue"])
            time.sleep(1.5)  # entre pages
        else:
            return


def get_image_url(session, file_title):
    params = {
        "action": "query",
        "format": "json",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    r = request_with_retry(session, API_URL, params=params)
    if r is None:
        return None
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [])
        if info:
            return info[0].get("url")
    return None


def scrape(out_dir, max_per_cat=300, image_size=1024):
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
            logger.info("Categorie %s -> %s (telecharges : %d/%d)",
                        category_name, wiki_cat, downloaded, max_per_cat)
            try:
                for item in tqdm(list_category_images(session, wiki_cat, limit=max_per_cat * 2),
                                 desc=wiki_cat, leave=False):
                    if downloaded >= max_per_cat:
                        break
                    title = item["title"]
                    if not title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        continue
                    url = get_image_url(session, title)
                    if not url:
                        continue
                    try:
                        img_resp = session.get(url, timeout=60)
                        img_resp.raise_for_status()
                        img = Image.open(BytesIO(img_resp.content))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                        out_path = cat_dir / f"wikimedia_{category_name}_{downloaded:04d}.jpg"
                        img.save(out_path, quality=90)
                        downloaded += 1
                        time.sleep(1.0)  # rate limit poli
                    except Exception as e:
                        logger.debug("Echec %s : %s", title, e)
            except Exception as e:
                logger.warning("Echec global categorie %s : %s", wiki_cat, e)
            time.sleep(2.0)  # entre categories

        stats[category_name] = downloaded
        logger.info("Categorie %s : %d images", category_name, downloaded)
        time.sleep(3.0)  # entre familles

    return stats


def main():
    parser = argparse.ArgumentParser(description="Scraping Wikimedia Commons (avec retry)")
    parser.add_argument("--out", type=Path, default=Path("01_data/raw/wikimedia"))
    parser.add_argument("--max", type=int, default=200, help="Max d'images par categorie")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    stats = scrape(args.out, max_per_cat=args.max, image_size=args.size)
    total = sum(stats.values())
    print(f"\nTotal : {total} images")
    for cat, n in stats.items():
        print(f"  {cat:12s} : {n}")


if __name__ == "__main__":
    main()
