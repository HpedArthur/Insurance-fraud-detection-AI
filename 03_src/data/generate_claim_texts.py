"""
Generation d'un dataset synthetique de declarations de sinistres habitation.

On utilise Mistral-7B-Instruct-v0.3 (open-source, en 4-bit pour Colab T4) pour
generer des paires texte/label :
  - label = 0 : declaration legitime (coherente, faits realistes)
  - label = 1 : declaration frauduleuse (incoherences subtiles, exagerations,
                contradictions internes, dates impossibles, vocabulaire generique)

Sortie : 01_data/synthetic/claim_texts.parquet
  Colonnes : claim_id, category, label, text, generation_seed

Usage :
    python 03_src/data/generate_claim_texts.py --per-class 150
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

# Templates de prompts. On varie systeme + scenarios pour eviter du sur-apprentissage.
SYSTEM_GENUINE = """Tu es un assure francais qui declare un sinistre habitation a son assureur.
Ecris une declaration ECRITE de sinistre realiste, en francais courant, de 4 a 8 phrases.

Contraintes :
- Sois SPECIFIQUE : date precise, heure approximative, lieu exact dans le logement, montant
  d'estimation des dommages plausible.
- Mentionne au moins UN element concret verifiable (ex : voisin temoin, intervention pompiers,
  date d'achat d'un meuble endommage).
- Style naturel, pas trop emotionnel, vocabulaire courant.
- Ne mentionne JAMAIS que c'est une declaration frauduleuse ou legitime.
- Reponds uniquement avec la declaration, sans preambule."""

SYSTEM_FRAUD = """Tu es un assure francais qui redige une FAUSSE declaration de sinistre habitation pour
escroquer son assureur. Tu veux que ca paraisse plausible MAIS tu commets quelques imprudences subtiles
qu'un enqueteur experimente pourrait reperer.

Inclure DISCRETEMENT au moins UNE de ces incoherences :
- Date impossible (week-end alors que tu pretends etre au travail, date passee/future absurde)
- Montant exagere par rapport au type de bien
- Vocabulaire trop vague ou trop emphatique ("tout est detruit", "c'est une catastrophe absolue")
- Contradiction interne entre deux phrases (ex : "j'etais absent" puis "j'ai entendu un bruit")
- Description d'objets de valeur sans aucune preuve d'achat
- Recit qui semble copie-colle d'un modele type
- Trop de details sur la valeur, peu sur les circonstances

Ecris en francais courant, 4 a 8 phrases.
Style naturel, sans avouer la fraude.
Ne mentionne JAMAIS que c'est une declaration frauduleuse.
Reponds uniquement avec la declaration, sans preambule."""

USER_TEMPLATE = """Type de sinistre : {category_fr}.
Contexte : appartement/maison en France, declaration aupres de l'assureur.
Tu dois generer une declaration {flavor}."""


def load_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Chargement Mistral-7B-Instruct (device=%s)", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        model = model.to(device)

    model.eval()
    return model, tokenizer, device


def generate_one(model, tokenizer, system: str, user: str, seed: int,
                 max_new_tokens: int = 350) -> str:
    messages = [
        {"role": "user", "content": system + "\n\n" + user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def clean_text(text: str) -> str:
    """Retire markdown / preambules / tags eventuels."""
    text = re.sub(r"^\s*(\*\*|##\s*|Declaration\s*:\s*)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run(out_path: Path, per_class: int, seed_base: int = 42) -> pd.DataFrame:
    model, tokenizer, _ = load_model()

    rows = []
    claim_id = 0

    for category in CATEGORIES:
        category_fr = CATEGORY_LABELS_FR[category]
        for label, system, flavor in [
            (0, SYSTEM_GENUINE, "legitime et coherente"),
            (1, SYSTEM_FRAUD, "qui paraitra plausible mais contient des incoherences subtiles"),
        ]:
            user = USER_TEMPLATE.format(category_fr=category_fr, flavor=flavor)
            with tqdm(total=per_class, desc=f"{category}/{['real', 'fake'][label]}") as pbar:
                generated = 0
                seed = seed_base + claim_id * 1000
                attempts = 0
                while generated < per_class and attempts < per_class * 3:
                    try:
                        text = generate_one(model, tokenizer, system, user, seed)
                        text = clean_text(text)
                        if 80 <= len(text) <= 2000:  # filtre longueur
                            rows.append({
                                "claim_id": claim_id,
                                "category": category,
                                "label": label,
                                "text": text,
                                "generation_seed": seed,
                            })
                            claim_id += 1
                            generated += 1
                            pbar.update(1)
                    except Exception as e:
                        logger.warning("Echec generation seed=%d : %s", seed, e)
                    seed += 1
                    attempts += 1

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Sauvegarde : %s (%d declarations)", out_path, len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generation declarations synthetiques")
    parser.add_argument("--per-class", type=int, default=150,
                        help="Nombre de declarations par (categorie x label)")
    parser.add_argument("--out", type=Path,
                        default=Path("01_data/synthetic/claim_texts.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = run(args.out, per_class=args.per_class, seed_base=args.seed)
    print(f"\n{len(df)} declarations generees")
    print(df.groupby(["category", "label"]).size())


if __name__ == "__main__":
    main()
