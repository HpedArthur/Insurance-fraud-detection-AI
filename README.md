# Détection de fraude à l'assurance habitation par LMM

Pipeline de détection d'images générées par IA dans le cadre de déclarations de sinistre habitation. Projet réalisé dans le cadre du cours *Intelligence Artificielle appliquée à l'Assurance* (ISFA, 2025-2026).

## Contexte

L'arrivée des modèles génératifs grand public abaisse drastiquement le coût de fabrication de fausses preuves photographiques. Ce projet propose un pipeline open-source combinant un encodeur visuel (CLIP), un LMM open-source pour l'analyse sémantique, et un classifieur supervisé léger, afin de discriminer une photo authentique de sinistre d'une photo générée par IA.

## Architecture

```
Photo soumise -> Pré-traitement -> CLIP ViT-L/14 -> LMM (BLIP-2 / LLaVA) -> Classifieur (XGBoost) -> Décision + score
```

Voir `00_PLAN_PROJET.md` pour le détail complet.

## Structure du repo

```
.
├── 00_PLAN_PROJET.md          # Plan maître (cadrage, planning, rôles, risques)
├── 01_data/                   # Données (manifest seulement, images hors Git)
│   ├── raw/                   # Téléchargements bruts
│   ├── processed/             # Dataset consolidé (parquet)
│   ├── external/              # Datasets HuggingFace (CIFAKE, ArtiFact)
│   └── synthetic/             # Images générées par nos soins
├── 02_notebooks/              # Notebooks Jupyter (EDA, modélisation, sensibilité)
├── 03_src/                    # Code Python réutilisable
│   ├── data/                  # Scripts d'acquisition et préparation
│   ├── features/              # Extraction d'embeddings, captions LMM
│   ├── models/                # Entraînement et inférence
│   └── utils/                 # Helpers
├── 04_models/                 # Modèles sérialisés (.pkl, .npy)
├── 05_app/                    # Application Gradio
├── 06_reports/                # Rapport PDF + figures
├── 07_presentation/           # Support de soutenance (PPTX)
├── 08_docs/                   # Notes techniques (stratégie dataset, choix tech)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone <url-du-repo>
cd <nom-du-repo>
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
huggingface-cli login              # token HF gratuit nécessaire
```

## Démarrage rapide

```bash
# Smoke test : télécharger 1000 images CIFAKE et entraîner un classifieur baseline
python 03_src/data/download_cifake.py --n 1000
python 03_src/features/extract_clip_embeddings.py
python 03_src/models/train_baseline.py

# Lancer l'application Streamlit en local
streamlit run 05_app/app.py
```

## Démo en ligne

Lien HuggingFace Spaces : *à compléter en semaine 3*

## Stack technique

Python 3.10+, PyTorch, Transformers, OpenCLIP, scikit-learn, XGBoost, Diffusers, Streamlit, SHAP.

## Licence

Projet pédagogique. Code sous licence MIT. Datasets publics sous leurs licences respectives (voir `08_docs/strategie_dataset.md`).
