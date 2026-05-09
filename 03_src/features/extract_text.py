"""
Extraction de features pour la declaration textuelle de sinistre.

Deux familles de features :

  A. HANDCRAFTED (rapide, deterministe, interpretable) :
     - longueur (chars, mots, phrases)
     - richesse vocabulaire (TTR : type-token ratio)
     - nombre de dates / nombre / lieux detectes par spaCy
     - presence de mots-cles emphatiques ("totalement", "completement", "tout est detruit"...)
     - score subjectivite simple (presence d'adjectifs emotionnels)

  B. LLM-AS-JUDGE (Mistral-7B-Instruct, en 4-bit) :
     Demande au LLM d'evaluer le texte selon 5 criteres et de renvoyer un JSON :
     - judge_specificity     : niveau de detail concret (0-1)
     - judge_coherence       : coherence interne du recit (0-1)
     - judge_plausibility    : plausibilite globale des faits (0-1)
     - judge_red_flags       : presence de signaux d'alerte (0-1)
     - judge_overall_genuine : verdict global, plus eleve = plus authentique (0-1)

Usage :
    python 03_src/features/extract_text.py --input 01_data/synthetic/claim_texts.parquet \
                                           --out 01_data/processed/claim_texts_with_features.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Mots-cles emphatiques en francais
EMPHATIC_WORDS = {
    "totalement", "completement", "absolument", "entierement", "integralement",
    "catastrophe", "desastre", "drame", "horrible", "terrible", "tout est detruit",
    "rien n'a ete epargne", "absolument tout", "inestimable",
}

JUDGE_PROMPT = """Tu es un enqueteur expert en fraude a l'assurance. On te soumet une declaration de sinistre habitation
ecrite par un assure. Tu dois evaluer la qualite et la credibilite du texte selon 5 criteres, en te basant
UNIQUEMENT sur le texte (pas de jugement moral).

Criteres (chacun note de 0.0 a 1.0) :
- specificity : niveau de detail concret (lieux precis, dates, montants, objets nommes). 1.0 = tres specifique.
- coherence : coherence interne du recit (pas de contradiction, suite logique). 1.0 = parfaitement coherent.
- plausibility : les faits decrits sont-ils plausibles ? 1.0 = totalement plausible.
- red_flags : presence de signaux d'alerte (vocabulaire emphatique, valeurs surevaluees, recit copie-colle,
  contradictions, dates impossibles). 1.0 = beaucoup de red flags.
- overall_genuine : verdict global, en etant prudent. 1.0 = quasi-certainement authentique, 0.0 = quasi-certainement frauduleux.

Reponds UNIQUEMENT avec un JSON valide selon ce schema, sans aucun commentaire :
{"specificity": float, "coherence": float, "plausibility": float, "red_flags": float, "overall_genuine": float}

Texte de la declaration :
\"\"\"
{text}
\"\"\"
"""


# ============================================================
# A. Features handcrafted
# ============================================================

def load_spacy_fr():
    """Charge spaCy fr_core_news_md, telecharge si absent."""
    import spacy
    try:
        return spacy.load("fr_core_news_md")
    except OSError:
        logger.info("Modele spaCy fr_core_news_md absent, telechargement...")
        from spacy.cli import download
        download("fr_core_news_md")
        return spacy.load("fr_core_news_md")


def handcrafted_features(text: str, nlp=None) -> dict:
    """Features handcrafted sur un texte."""
    if not text or not isinstance(text, str):
        return {
            "txt_n_chars": 0, "txt_n_words": 0, "txt_n_sentences": 0,
            "txt_ttr": 0.0,
            "txt_n_dates": 0, "txt_n_money": 0, "txt_n_loc": 0, "txt_n_persons": 0,
            "txt_n_emphatic": 0,
        }

    n_chars = len(text)
    words = re.findall(r"\b\w+\b", text.lower())
    n_words = len(words)
    n_sentences = max(1, len(re.findall(r"[.!?]+", text)))
    ttr = len(set(words)) / n_words if n_words > 0 else 0.0

    feats = {
        "txt_n_chars": n_chars,
        "txt_n_words": n_words,
        "txt_n_sentences": n_sentences,
        "txt_ttr": ttr,
    }

    n_dates = n_money = n_loc = n_persons = 0
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    n_dates += 1
                elif ent.label_ == "MONEY":
                    n_money += 1
                elif ent.label_ in ("LOC", "GPE"):
                    n_loc += 1
                elif ent.label_ in ("PER", "PERSON"):
                    n_persons += 1
        except Exception as e:
            logger.debug("spaCy a echoue : %s", e)
    feats.update({
        "txt_n_dates": n_dates,
        "txt_n_money": n_money,
        "txt_n_loc": n_loc,
        "txt_n_persons": n_persons,
    })

    text_lower = text.lower()
    n_emphatic = sum(1 for kw in EMPHATIC_WORDS if kw in text_lower)
    feats["txt_n_emphatic"] = n_emphatic

    return feats


# ============================================================
# B. LLM-as-judge
# ============================================================

def load_judge(device: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Chargement Mistral-7B (judge), device=%s", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if device == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def judge_text(model, tokenizer, text: str) -> dict:
    # NOTE : on N'UTILISE PAS .format() car le JUDGE_PROMPT contient un schema JSON
    # avec des accolades (ex: {"specificity": float, ...}) que .format interpreterait
    # comme des placeholders et leverait KeyError('"specificity"').
    safe_text = text.replace('"""', "'''")
    prompt = JUDGE_PROMPT.replace("{text}", safe_text)
    msgs = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_judge_json(answer)


def parse_judge_json(text: str) -> dict:
    default = {
        "judge_specificity": None, "judge_coherence": None,
        "judge_plausibility": None, "judge_red_flags": None,
        "judge_overall_genuine": None,
    }
    if not text:
        return default
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return default
    try:
        d = json.loads(match.group(0))
        return {
            "judge_specificity": _f(d.get("specificity")),
            "judge_coherence": _f(d.get("coherence")),
            "judge_plausibility": _f(d.get("plausibility")),
            "judge_red_flags": _f(d.get("red_flags")),
            "judge_overall_genuine": _f(d.get("overall_genuine")),
        }
    except (json.JSONDecodeError, TypeError):
        return default


def _f(v):
    if v is None:
        return None
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


# ============================================================
# Pipeline
# ============================================================

def extract(df: pd.DataFrame, with_judge: bool = True) -> pd.DataFrame:
    nlp = None
    try:
        nlp = load_spacy_fr()
    except Exception as e:
        logger.warning("spaCy indisponible, features NER ignorees : %s", e)

    rows = []
    judge_model = judge_tokenizer = None
    if with_judge:
        try:
            judge_model, judge_tokenizer = load_judge()
        except Exception as e:
            logger.warning("Judge indisponible (%s) : on garde uniquement les features handcrafted", e)
            with_judge = False

    for _, row in tqdm(df.iterrows(), total=len(df), desc="text features"):
        text = row.get("text") or row.get("claim_text") or ""
        feats = handcrafted_features(text, nlp=nlp)
        if with_judge:
            try:
                feats.update(judge_text(judge_model, judge_tokenizer, text))
            except Exception as e:
                logger.warning("Judge a echoue : %s", e)
                feats.update({k: None for k in [
                    "judge_specificity", "judge_coherence", "judge_plausibility",
                    "judge_red_flags", "judge_overall_genuine",
                ]})
        else:
            feats.update({k: None for k in [
                "judge_specificity", "judge_coherence", "judge_plausibility",
                "judge_red_flags", "judge_overall_genuine",
            ]})
        rows.append(feats)

    extra = pd.DataFrame(rows).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Features texte (handcrafted + LLM judge)")
    parser.add_argument("--input", type=Path,
                        default=Path("01_data/synthetic/claim_texts.parquet"))
    parser.add_argument("--out", type=Path,
                        default=Path("01_data/processed/claim_texts_with_features.parquet"))
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip Mistral judge (plus rapide, pour les tests)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    logger.info("Input : %d declarations", len(df))

    out = extract(df, with_judge=not args.no_judge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    logger.info("Sauvegarde : %s", args.out)


if __name__ == "__main__":
    main()
