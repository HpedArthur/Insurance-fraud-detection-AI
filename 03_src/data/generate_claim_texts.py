"""
Generation d'un dataset synthetique de declarations de sinistres habitation.

Avec :
- Sauvegarde incrementale apres chaque texte (zero perte si crash)
- Resume automatique : skip ce qui est deja dans le parquet
- max_new_tokens reduit a 250 pour aller plus vite

Usage :
    python 03_src/data/generate_claim_texts.py --per-class 100
    # Si interrompu, relancer la meme commande pour reprendre
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

CATEGORIES = ["water", "fire", "glass", "vandalism"]

CATEGORY_LABELS_FR = {
    "water": "degat des eaux",
    "fire": "incendie",
    "glass": "bris de vitre",
    "vandalism": "degradation/cambriolage",
}

SYSTEM_GENUINE = """Tu es un assure francais qui declare un sinistre habitation a son assureur.
Ecris une declaration de sinistre realiste, en francais courant, de 4 a 6 phrases.
Sois SPECIFIQUE : date, heure, lieu, montant plausible. Mentionne UN element verifiable.
Style naturel, vocabulaire courant. Reponds uniquement avec la declaration."""

SYSTEM_FRAUD = """Tu es un assure francais qui redige une FAUSSE declaration de sinistre habitation.
Tu veux que ca paraisse plausible MAIS tu commets une imprudence subtile :
date impossible, montant exagere, vocabulaire trop emphatique, contradiction interne, OU
description trop vague d'objets de valeur sans preuve d'achat.
Ecris en francais courant, 4 a 6 phrases. Style naturel, ne pas avouer la fraude.
Reponds uniquement avec la declaration."""

USER_TEMPLATE = """Type de sinistre : {category_fr}.
Contexte : appartement/maison en France.
Genere une declaration {flavor}."""


def load_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Chargement Mistral-7B-Instruct (device=%s)", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if device == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                       device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.eval()
    return model, tokenizer, device


def generate_one(model, tokenizer, system, user, seed, max_new_tokens=250):
    messages = [{"role": "user", "content": system + "\n\n" + user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=0.85, top_p=0.92, repetition_penalty=1.1,
                                  pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def clean_text(text):
    text = re.sub(r"^\s*(\*\*|##\s*|Declaration\s*:\s*)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def append_row(out_path: Path, row: dict):
    """Ajoute une ligne au parquet de maniere atomique (read+append+write)."""
    if out_path.exists():
        df = pd.read_parquet(out_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)


def get_existing_counts(out_path: Path) -> dict:
    """Renvoie le nombre de textes deja generes par (category, label)."""
    if not out_path.exists():
        return {}
    df = pd.read_parquet(out_path)
    counts = df.groupby(["category", "label"]).size().to_dict()
    return counts


def run(out_path: Path, per_class: int, seed_base: int = 42, max_new_tokens: int = 250):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = get_existing_counts(out_path)
    if existing:
        logger.info("Resume detecte. Deja en place : %s", existing)

    model, tokenizer, _ = load_model()

    next_claim_id = 0
    if out_path.exists():
        df_exist = pd.read_parquet(out_path)
        if len(df_exist) > 0:
            next_claim_id = int(df_exist["claim_id"].max()) + 1

    for category in CATEGORIES:
        category_fr = CATEGORY_LABELS_FR[category]
        for label, system, flavor in [
            (0, SYSTEM_GENUINE, "legitime et coherente"),
            (1, SYSTEM_FRAUD, "qui paraitra plausible mais contient des incoherences subtiles"),
        ]:
            target = per_class
            already = existing.get((category, label), 0)
            if already >= target:
                logger.info("[%s/%s] deja complet (%d/%d), skip", category, label, already, target)
                continue

            user = USER_TEMPLATE.format(category_fr=category_fr, flavor=flavor)
            seed = seed_base + (next_claim_id + 1) * 1000
            with tqdm(total=target, initial=already, desc=f"{category}/{['real','fake'][label]}") as pbar:
                generated = already
                attempts = 0
                while generated < target and attempts < target * 3:
                    try:
                        text = generate_one(model, tokenizer, system, user, seed, max_new_tokens)
                        text = clean_text(text)
                        if 80 <= len(text) <= 2000:
                            row = {
                                "claim_id": next_claim_id,
                                "category": category,
                                "label": label,
                                "text": text,
                                "generation_seed": seed,
                            }
                            append_row(out_path, row)
                            next_claim_id += 1
                            generated += 1
                            pbar.update(1)
                    except Exception as e:
                        logger.warning("Echec seed=%d : %s", seed, e)
                    seed += 1
                    attempts += 1

    df = pd.read_parquet(out_path)
    logger.info("Termine : %d declarations totales", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generation declarations synthetiques (avec resume)")
    parser.add_argument("--per-class", type=int, default=100,
                        help="Nombre de declarations par (categorie x label)")
    parser.add_argument("--out", type=Path, default=Path("01_data/synthetic/claim_texts.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=250,
                        help="Longueur max des textes (250 = bon compromis vitesse/qualite)")
    args = parser.parse_args()

    df = run(args.out, per_class=args.per_class, seed_base=args.seed,
             max_new_tokens=args.max_tokens)
    print(f"\n{len(df)} declarations totales")
    print(df.groupby(["category", "label"]).size())


if __name__ == "__main__":
    main()
