"""Helpers pour metadonnees image : EXIF, dimensions, hash perceptuel."""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import ExifTags, Image


def extract_exif(img_path: Path) -> dict:
    """Extrait les champs EXIF utiles (camera, date, GPS).

    Returns:
        dict avec exif_present, camera, datetime, has_gps.
    """
    info = {"exif_present": False, "camera": None, "datetime": None, "has_gps": False}
    try:
        with Image.open(img_path) as img:
            exif = img.getexif()
            if not exif:
                return info
            tag_dict = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            info["exif_present"] = True
            info["camera"] = tag_dict.get("Model")
            info["datetime"] = tag_dict.get("DateTime") or tag_dict.get("DateTimeOriginal")
            # GPS info
            gps_tag_id = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
            if gps_tag_id is not None and gps_tag_id in exif:
                info["has_gps"] = True
    except Exception:
        pass
    return info


def perceptual_hash(img_path: Path, hash_size: int = 16) -> str | None:
    """Calcule un perceptual hash (pHash) de l'image. Utile pour detecter doublons."""
    try:
        with Image.open(img_path) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None


def image_dimensions(img_path: Path) -> tuple[int | None, int | None]:
    """Renvoie (width, height) ou (None, None) si echec."""
    try:
        with Image.open(img_path) as img:
            return img.width, img.height
    except Exception:
        return None, None
