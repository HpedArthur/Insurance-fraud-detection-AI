---
title: Insurance Fraud Detection
emoji: V
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Detection de fraude a l'assurance habitation

Demonstrateur du projet ISFA 2025-2026.

L'application prend en entree :
- Une photo prétendant illustrer un sinistre habitation
- Une declaration textuelle structuree (formulaire de gestion de sinistre)

Et produit deux scores :
- **Score image** : probabilite que la photo soit generee par IA (axe principal)
- **Score multimodal** : score affine integrant la coherence textuelle

## Deploiement sur HuggingFace Spaces

1. Creer un Space type `streamlit` sur [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Copier le contenu de `05_app/` a la racine du Space
3. Uploader les modeles entraines `04_models/image_model_lite.joblib` et `04_models/multimodal_model_lite.joblib` dans `04_models/` du Space
4. Le Space construit automatiquement l'environnement et lance l'app

## Lancement local

```bash
pip install -r 05_app/requirements.txt
python -m spacy download fr_core_news_md
streamlit run 05_app/app.py
```
