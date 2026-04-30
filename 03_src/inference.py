"""
Wrapper d'inference unifie pour la detection de fraude image + texte.

Utilisation typique (depuis l'app Streamlit) :

    from inference import FraudDetector
    detector = FraudDetector(model_dir="04_models", variant="lite")
    result = detector.predict(image=pil_img, claim_form={
        "claim_type": "water",
        "claim_date": "2026-04-15",
        "discovery_date": "2026-04-16",
        "circumstances": "...",
        ...
    })

Le wrapper detecte automatiquement quels modeles sont disponibles :
- CLIP : indispensable
- BLIP-2 / LLaVA : optionnels (mode degrade si absents)
- spaCy + Mistral : optionnels pour le LLM-judge
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers : extraction des features image et texte
# =============================================================================

def _safe_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


def extract_exif_features(image) -> dict:
    """Renvoie un dict de flags EXIF (toujours present, valeurs 0/1)."""
    from PIL import ExifTags
    info = {"exif_present": 0.0, "has_gps": 0.0}
    try:
        exif = image.getexif()
        if exif:
            info["exif_present"] = 1.0
            gps_tag_id = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
            if gps_tag_id is not None and gps_tag_id in exif:
                info["has_gps"] = 1.0
    except Exception:
        pass
    return info


# =============================================================================
# FraudDetector
# =============================================================================

class FraudDetector:
    """Detecteur de fraude unifie. Charge ce qui est disponible dans model_dir."""

    def __init__(self, model_dir: str | Path = "04_models", variant: str = "lite",
                 device: str | None = None, load_lmm: bool = False, load_judge: bool = False):
        """
        Args:
            model_dir : repertoire contenant les .joblib des classifieurs.
            variant : 'lite' ou 'full'.
            device : 'cuda' / 'cpu' / None (auto).
            load_lmm : si True, tente de charger BLIP-2 et LLaVA pour le mode 'full'.
                       Sur HF Spaces gratuit, garder a False (CPU only).
            load_judge : si True, tente de charger Mistral pour le LLM-as-judge.
        """
        self.model_dir = Path(model_dir)
        self.variant = variant
        self.load_lmm_flag = load_lmm
        self.load_judge_flag = load_judge

        # Lazy-loaded components
        self._clip_model = None
        self._clip_preprocess = None
        self._blip2 = None
        self._blip2_proc = None
        self._llava = None
        self._llava_proc = None
        self._judge_model = None
        self._judge_tokenizer = None
        self._spacy_nlp = None

        # Device
        torch = _safe_import("torch")
        if device is None:
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.device = device
        self.torch = torch

        # Classifieurs
        self.image_model_pkg = self._load_or_warn(self.model_dir / f"image_model_{variant}.joblib", "image_model")
        self.multimodal_model_pkg = self._load_or_warn(self.model_dir / f"multimodal_model_{variant}.joblib",
                                                      "multimodal_model")

    def _load_or_warn(self, path: Path, label: str):
        if not path.exists():
            logger.warning("%s absent : %s -- mode degrade", label, path)
            return None
        return joblib.load(path)

    # ---------------- CLIP (indispensable) ----------------
    def _ensure_clip(self):
        if self._clip_model is not None:
            return
        import open_clip
        logger.info("Chargement CLIP ViT-L/14 sur %s...", self.device)
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self._clip_model = model.to(self.device).eval()
        self._clip_preprocess = preprocess

    def encode_image_clip(self, image) -> np.ndarray:
        """Renvoie l'embedding CLIP normalise L2 (768-d)."""
        self._ensure_clip()
        torch = self.torch
        with torch.no_grad():
            x = self._clip_preprocess(image).unsqueeze(0).to(self.device)
            f = self._clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().squeeze(0).astype(np.float32)

    # ---------------- LMM (optionnels) ----------------
    def _ensure_blip2(self):
        if self._blip2 is not None or not self.load_lmm_flag:
            return
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            logger.info("Chargement BLIP-2...")
            self._blip2_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self._blip2 = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=self.torch.float16 if self.device == "cuda" else self.torch.float32,
            ).to(self.device).eval()
        except Exception as e:
            logger.warning("Echec chargement BLIP-2 : %s", e)
            self.load_lmm_flag = False

    def _ensure_llava(self):
        if self._llava is not None or not self.load_lmm_flag:
            return
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
            logger.info("Chargement LLaVA-1.5-7B...")
            self._llava_proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            kwargs = {"torch_dtype": self.torch.float32}
            if self.device == "cuda":
                kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=self.torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                    "device_map": "auto",
                }
            self._llava = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", **kwargs
            ).eval()
        except Exception as e:
            logger.warning("Echec chargement LLaVA : %s", e)
            self.load_lmm_flag = False

    def lmm_features_image(self, image) -> dict:
        """Renvoie les scores BLIP-2 et LLaVA s'ils sont charges, sinon des None."""
        out = {
            "blip2_score_real": 0.5, "blip2_score_ai": 0.5, "blip2_score_artifact": 0.5,
            "llava_score_real": 0.5, "llava_score_artifact": 0.5, "llava_score_coherence": 0.5,
        }
        if not self.load_lmm_flag:
            return out
        # ... A implementer pour BLIP-2 et LLaVA, voir extract_blip2.py et extract_llava.py
        # Par defaut, on renvoie 0.5 (neutre) pour ne pas perturber le classifieur
        return out

    # ---------------- Texte ----------------
    def _ensure_spacy(self):
        if self._spacy_nlp is not None:
            return
        try:
            import spacy
            self._spacy_nlp = spacy.load("fr_core_news_md")
        except Exception as e:
            logger.warning("spaCy fr_core_news_md indisponible : %s", e)
            self._spacy_nlp = False

    def extract_text_handcrafted(self, text: str) -> dict:
        if not text:
            return {k: 0 for k in [
                "txt_n_chars", "txt_n_words", "txt_n_sentences", "txt_ttr",
                "txt_n_dates", "txt_n_money", "txt_n_loc", "txt_n_persons", "txt_n_emphatic",
            ]}
        words = re.findall(r"\b\w+\b", text.lower())
        n_chars = len(text)
        n_words = len(words)
        n_sentences = max(1, len(re.findall(r"[.!?]+", text)))
        ttr = len(set(words)) / n_words if n_words > 0 else 0.0
        emphatic_set = {
            "totalement", "completement", "absolument", "entierement", "integralement",
            "catastrophe", "desastre", "drame", "horrible", "terrible",
            "tout est detruit", "rien n'a ete epargne", "absolument tout", "inestimable",
        }
        n_emphatic = sum(1 for kw in emphatic_set if kw in text.lower())

        n_dates = n_money = n_loc = n_persons = 0
        self._ensure_spacy()
        if self._spacy_nlp:
            try:
                doc = self._spacy_nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        n_dates += 1
                    elif ent.label_ == "MONEY":
                        n_money += 1
                    elif ent.label_ in ("LOC", "GPE"):
                        n_loc += 1
                    elif ent.label_ in ("PER", "PERSON"):
                        n_persons += 1
            except Exception:
                pass

        return {
            "txt_n_chars": n_chars, "txt_n_words": n_words, "txt_n_sentences": n_sentences,
            "txt_ttr": ttr,
            "txt_n_dates": n_dates, "txt_n_money": n_money, "txt_n_loc": n_loc, "txt_n_persons": n_persons,
            "txt_n_emphatic": n_emphatic,
        }

    def text_features_judge(self, text: str) -> dict:
        """LLM-as-judge sur le texte. Renvoie 5 scores entre 0 et 1.
        Si Mistral n'est pas charge, renvoie des 0.5 neutres."""
        out = {
            "judge_specificity": 0.5, "judge_coherence": 0.5,
            "judge_plausibility": 0.5, "judge_red_flags": 0.5,
            "judge_overall_genuine": 0.5,
        }
        if not self.load_judge_flag:
            return out
        # Utilise extract_text.judge_text si dispo
        try:
            from features.extract_text import judge_text, load_judge
            if self._judge_model is None:
                self._judge_model, self._judge_tokenizer = load_judge(device=self.device)
            res = judge_text(self._judge_model, self._judge_tokenizer, text)
            return {k: (v if v is not None else 0.5) for k, v in res.items()}
        except Exception as e:
            logger.warning("Judge a echoue : %s", e)
            return out

    # ---------------- Coherence inter-champs ----------------
    @staticmethod
    def cross_field_coherence(form: dict[str, Any]) -> dict:
        """Calcule des features de coherence sur le formulaire structure."""
        coh = {
            "cf_delay_days": 0.0,
            "cf_amount_per_m2": 0.0,
            "cf_has_third_party": 0.0,
            "cf_has_witness": 0.0,
            "cf_authorities_called": 0.0,
            "cf_circumstances_short": 0.0,
        }
        try:
            cd = _to_date(form.get("claim_date"))
            dd = _to_date(form.get("discovery_date"))
            if cd and dd:
                coh["cf_delay_days"] = (dd - cd).days
        except Exception:
            pass
        amount = float(form.get("amount") or 0)
        surface = float(form.get("surface") or 0)
        if surface > 0:
            coh["cf_amount_per_m2"] = amount / surface
        coh["cf_has_third_party"] = 1.0 if form.get("third_party") else 0.0
        coh["cf_has_witness"] = 1.0 if form.get("witness") else 0.0
        coh["cf_authorities_called"] = 1.0 if form.get("authorities") else 0.0
        circ = form.get("circumstances") or ""
        coh["cf_circumstances_short"] = 1.0 if len(circ) < 80 else 0.0
        return coh

    # ---------------- Score image seul (utilise par occlusion) ----------------
    def score_image_only(self, image) -> float:
        """Renvoie uniquement le score image (sans calculer le multimodal).
        Utilise pour l'occlusion sensitivity ou les analyses de sensibilite."""
        if self.image_model_pkg is None:
            raise RuntimeError("Aucun modele image charge.")
        clip_emb = self.encode_image_clip(image)
        exif = extract_exif_features(image)
        lmm_feats = self.lmm_features_image(image)
        feat_names = self.image_model_pkg["feature_names"]
        n_clip = sum(1 for n in feat_names if n.startswith("clip_"))
        extras = [n for n in feat_names if not n.startswith("clip_")]
        feature_values = {**exif, **lmm_feats}
        extras_arr = np.array([float(feature_values.get(n, 0.5)) for n in extras], dtype=np.float32)
        X_img = np.concatenate([clip_emb[:n_clip], extras_arr]).reshape(1, -1)
        X_img_s = self.image_model_pkg["scaler"].transform(X_img)
        return float(self.image_model_pkg["model"].predict_proba(X_img_s)[0, 1])

    # ---------------- Explicabilite image ----------------
    def explain_image(self, image, patch_size: int = 56, stride: int = 28) -> dict:
        """Calcule une heatmap d'importance des zones de l'image.

        Returns dict avec keys :
          heatmap : ndarray (H, W) dans [0, 1], plus rouge = plus utilise par le modele
          overlay : PIL.Image avec heatmap superposee
          base_score : score image initial
        """
        from utils.explain import occlusion_heatmap, overlay_heatmap
        base_score = self.score_image_only(image)
        heatmap = occlusion_heatmap(image, self.score_image_only,
                                    patch_size=patch_size, stride=stride)
        overlay = overlay_heatmap(image, heatmap)
        return {"heatmap": heatmap, "overlay": overlay, "base_score": base_score}

    # ---------------- Pipeline complet ----------------
    def predict(self, image, claim_form: dict | None = None) -> dict:
        """Predict. Retourne un dict structure pour l'app Streamlit."""
        if self.image_model_pkg is None:
            raise RuntimeError("Aucun modele image charge. Lance train_image_model.py d'abord.")

        # 1) features image
        clip_emb = self.encode_image_clip(image)
        exif = extract_exif_features(image)
        lmm_feats = self.lmm_features_image(image)

        # 2) Construire X_image en respectant feature_names du modele
        feat_names = self.image_model_pkg["feature_names"]
        n_clip = sum(1 for n in feat_names if n.startswith("clip_"))
        extras = [n for n in feat_names if not n.startswith("clip_")]

        feature_values = {**exif, **lmm_feats}
        extras_arr = np.array([float(feature_values.get(n, 0.5)) for n in extras], dtype=np.float32)
        X_img = np.concatenate([clip_emb[:n_clip], extras_arr]).reshape(1, -1)
        X_img_s = self.image_model_pkg["scaler"].transform(X_img)
        score_image = float(self.image_model_pkg["model"].predict_proba(X_img_s)[0, 1])

        # 3) Features texte si declaration fournie
        text_feats = {}
        score_multimodal = None
        if claim_form is not None:
            circ = claim_form.get("circumstances", "") or ""
            text_feats.update(self.extract_text_handcrafted(circ))
            text_feats.update(self.text_features_judge(circ))
            text_feats.update(self.cross_field_coherence(claim_form))

            if self.multimodal_model_pkg is not None:
                mm_names = self.multimodal_model_pkg["feature_names"]
                row = []
                for n in mm_names:
                    if n == "score_image":
                        row.append(score_image)
                    elif n in text_feats:
                        row.append(float(text_feats[n]))
                    else:
                        row.append(0.0)
                X_mm = np.array(row, dtype=np.float32).reshape(1, -1)
                X_mm_s = self.multimodal_model_pkg["scaler"].transform(X_mm)
                score_multimodal = float(self.multimodal_model_pkg["model"].predict_proba(X_mm_s)[0, 1])

        # 4) Decision
        score_used = score_multimodal if score_multimodal is not None else score_image
        if score_used < 0.3:
            decision = "Légitime"
            decision_color = "green"
        elif score_used < 0.7:
            decision = "À expertiser"
            decision_color = "orange"
        else:
            decision = "Fraude probable"
            decision_color = "red"

        return {
            "score_image": score_image,
            "score_multimodal": score_multimodal,
            "score_used": score_used,
            "decision": decision,
            "decision_color": decision_color,
            "image_features": {**exif, **lmm_feats},
            "text_features": text_feats,
        }


# =============================================================================
# Helpers
# =============================================================================

def _to_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                continue
    return None
