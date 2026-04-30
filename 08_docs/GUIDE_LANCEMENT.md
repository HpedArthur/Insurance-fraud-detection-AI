# Guide de lancement du projet — pas à pas

Ce guide t'amène de **zéro** à un **projet entièrement fonctionnel** : pipeline qui tourne, modèles entraînés, app Streamlit utilisable. Compte 4 à 6 heures de calcul cumulé sur Colab T4 (gratuit) + ~1 heure de manipulation manuelle.

---

## Vue d'ensemble du pipeline

```
A. Setup environnement (15 min, une fois)
   |
B. Acquisition des données (1h30 sur Colab T4)
   |
C. Extraction des features (1h30 sur Colab T4)
   |
D. Entraînement des modèles (10 min)
   |
E. Calibration des seuils (1 min)
   |
F. Lancer l'app Streamlit (instantané)
   |
G. (Optionnel) Déploiement HuggingFace Spaces (10 min)
```

---

## A — Setup environnement (à faire une fois)

### A.1 Sur ta machine Windows

```powershell
# 1. Aller à la racine du projet
cd "C:\Users\arthu\Documents\Data Mining Fraude à l'assurance\Data Mining Fraude à l'assurance"

# 2. Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Installer toutes les dépendances
pip install -r requirements.txt

# 4. Modèle spaCy français (~50 Mo)
python -m spacy download fr_core_news_md

# 5. Compte HuggingFace gratuit + token
#    -> https://huggingface.co/join (si pas déjà créé)
#    -> https://huggingface.co/settings/tokens (créer un token "read")
huggingface-cli login
# Coller le token quand demandé

# 6. (Optionnel mais recommandé) Clé API Pexels
#    -> https://www.pexels.com/api/ (instantané, gratuit, sans CB)
#    -> Mettre la clé dans un fichier .env à la racine du projet :
#       PEXELS_API_KEY=ta_cle_ici
```

### A.2 Sur Google Colab (pour les étapes lourdes)

1. Aller sur https://colab.research.google.com
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** (gratuit)
3. Monter Google Drive et y déposer le projet :

```python
# Dans une cellule Colab
from google.colab import drive
drive.mount('/content/drive')

# Copier le projet sur Drive ou le cloner depuis GitHub
%cd /content/drive/MyDrive/
!git clone https://github.com/HpedArthur/Insurance-fraud-detection-AI.git
%cd Insurance-fraud-detection-AI

# Installer les dépendances
!pip install -r requirements.txt -q
!python -m spacy download fr_core_news_md -q

# Auth HuggingFace
from huggingface_hub import login
login()  # coller ton token
```

---

## B — Acquisition des données (1h30 sur Colab T4)

### B.1 Étage 1 — Données génériques (CIFAKE + ArtiFact)

```bash
# CIFAKE : 5000 images réelles + 5000 fakes par split (train/test)
python 03_src/data/download_cifake.py --n 5000
# Sortie : 01_data/external/cifake/{train,test}/{real,fake}/*.jpg
# Durée : ~10 min

# ArtiFact subset : 5000 par classe
python 03_src/data/download_artifact.py --n 5000
# Sortie : 01_data/external/artifact/{real,fake}/*.jpg
# Durée : ~15 min
```

Tu peux **vérifier visuellement** en ouvrant `01_data/external/cifake/train/real/` dans l'explorateur de fichiers — tu verras les vraies vignettes CIFAR-10.

### B.2 Étage 2 — Domaine habitation

```bash
# Wikimedia Commons (sans clé API) : ~300 images par catégorie
python 03_src/data/scrape_wikimedia.py --max 300
# Sortie : 01_data/raw/wikimedia/{water,fire,glass,vandalism}/*.jpg
# Durée : ~20 min

# Pexels (nécessite la clé dans .env) : ~200 par catégorie
python 03_src/data/scrape_pexels.py --per-cat 200
# Sortie : 01_data/raw/pexels/{water,fire,glass,vandalism}/*.jpg
# Durée : ~10 min
# Si pas de clé Pexels, skipper cette étape (Wikimedia suffit)

# Génération synthétique SDXL Turbo (impérativement sur GPU) : 200 par catégorie
python 03_src/data/generate_sdxl.py --per-cat 200
# Sortie : 01_data/synthetic/sdxl/{water,fire,glass,vandalism}/*.jpg
# Durée : ~30 min sur Colab T4
```

### B.3 (Optionnel mais recommandé) Vos photos perso

À chacun de prendre 20-30 photos avec le smartphone et les ranger ainsi :

```
01_data/raw/personal/water/      (taches d'eau au plafond, fuites...)
01_data/raw/personal/fire/       (papier brûlé, traces de suie...)
01_data/raw/personal/glass/      (verre cassé, vitre fissurée...)
01_data/raw/personal/vandalism/  (objets renversés, tags...)
```

Format JPG, JPEG ou PNG. Pas besoin de qualité pro, au contraire — des photos smartphone authentiques avec EXIF sont précieuses.

### B.4 Génération des déclarations textuelles synthétiques

```bash
# Mistral-7B-Instruct sur GPU (4-bit) : 150 textes par catégorie x 2 classes
python 03_src/data/generate_claim_texts.py --per-class 150
# Sortie : 01_data/synthetic/claim_texts.parquet (1200 textes)
# Durée : ~30 min sur Colab T4
```

### B.5 Consolidation du dataset

```bash
# Scanne tous les dossiers, calcule EXIF + phash, fait les splits
python 03_src/data/build_dataset.py
# Sortie : 01_data/processed/dataset.parquet
# Durée : ~5 min
```

À la fin tu verras un résumé du type :

```
=== RESUME DATASET ===
Total : 84321 images

Par split :
  generic_train     67000
  generic_test      8400
  generic_val       8400
  domain_test        450
  domain_val         200

Par source :
  source     label
  artifact   0     5000
  artifact   1     5000
  cifake     0     30000
  cifake     1     30000
  pexels     0     800
  personal   0      60
  sdxl       1     800
  wikimedia  0     1200
```

---

## C — Extraction des features (1h30 sur Colab T4)

### C.1 Embeddings CLIP (indispensable)

```bash
python 03_src/features/extract_clip.py --batch 32
# Sortie : 04_models/clip_embeddings.npy
#          01_data/processed/dataset_with_clip.parquet
# Durée : ~30 min sur T4
```

### C.2 (Variante full uniquement) BLIP-2 et LLaVA

```bash
# BLIP-2 — 3 scores VQA par image
python 03_src/features/extract_blip2.py
# Sortie : 01_data/processed/dataset_with_blip2.parquet
# Durée : ~1h sur T4

# LLaVA-1.5-7B (4-bit) — 4 scores structurés par image
python 03_src/features/extract_llava.py
# Sortie : 01_data/processed/dataset_with_lmm.parquet
# Durée : ~2h sur T4
```

Si Colab T4 timeout, utiliser `--limit-to-domain` pour ne traiter que le sous-ensemble habitation (beaucoup plus rapide, suffisant pour la démo).

### C.3 Features texte

```bash
# Handcrafted + Mistral-as-judge sur les déclarations synthétiques
python 03_src/features/extract_text.py
# Sortie : 01_data/processed/claim_texts_with_features.parquet
# Durée : ~1h sur T4
```

Si tu veux aller vite (sans le LLM judge) : ajouter `--no-judge` (le fichier sera produit avec uniquement les features handcrafted, ce qui est suffisant pour la variante lite).

---

## D — Entraînement des modèles (10 min)

```bash
# Modèle image (axe principal du projet)
python 03_src/models/train_image_model.py --variant both
# Entraîne et sauvegarde :
#   04_models/image_model_lite.joblib
#   04_models/image_model_full.joblib
# Affiche les métriques sur val/test/domain_test
# Durée : ~5 min sur CPU

# Modèle multimodal (image + texte)
python 03_src/models/train_multimodal_model.py --variant both
# Sortie : 04_models/multimodal_model_{lite,full}.joblib
# Durée : ~2 min
```

À la fin de chaque entraînement, le terminal affiche les métriques. **Notez les chiffres** : c'est ce qui ira dans le rapport et le PPT.

---

## E — Calibration des seuils (1 min)

```bash
# Pour la variante lite
python 03_src/utils/calibration.py --variant lite

# Pour la variante full
python 03_src/utils/calibration.py --variant full
```

Sortie : `04_models/thresholds_image_{lite,full}.json` — l'app Streamlit les chargera automatiquement.

---

## F — Lancer l'app Streamlit

```bash
# Activer l'env si pas déjà fait
.venv\Scripts\Activate.ps1   # Windows

# Lancer
streamlit run 05_app/app.py
```

→ Ouvre http://localhost:8501 dans ton navigateur. L'app est prête à être utilisée. Test sur 2 cas :

1. **Cas légitime** : photo réelle de dégât (de Wikimedia ou ton smartphone) + déclaration spécifique avec dates, témoin, etc.
2. **Cas frauduleux** : image SDXL générée + déclaration vague et emphatique.

Tu vois pour chaque dossier : score image, score multimodal, décision finale, heatmap d'explication.

---

## G — (Optionnel) Déploiement HuggingFace Spaces

Pour avoir une URL publique à montrer en soutenance.

1. Créer un Space : https://huggingface.co/new-space
2. Type : **Streamlit**
3. Hardware : **CPU basic (gratuit)**
4. Nom : `insurance-fraud-detection`
5. Cloner le Space localement et copier les fichiers de l'app + le modèle lite :

```bash
git clone https://huggingface.co/spaces/HpedArthur/insurance-fraud-detection
cd insurance-fraud-detection

# Copier les fichiers app
cp ../05_app/app.py .
cp ../05_app/requirements.txt .
cp ../05_app/packages.txt .
cp ../05_app/README.md .

# Copier le code source nécessaire
mkdir -p 03_src
cp -r ../03_src/* 03_src/

# Copier les modèles entraînés (variante lite uniquement, pour rester sous 5 Go)
mkdir -p 04_models
cp ../04_models/image_model_lite.joblib 04_models/
cp ../04_models/multimodal_model_lite.joblib 04_models/
cp ../04_models/thresholds_image_lite.json 04_models/

# Pousser
git add .
git commit -m "Deploiement initial"
git push
```

Le Space build automatiquement (~3 min) et publie l'URL `https://huggingface.co/spaces/HpedArthur/insurance-fraud-detection`.

---

## Troubleshooting

### "Aucune image trouvée" lors du build_dataset

→ Tu n'as pas lancé les scripts d'acquisition (étape B). Relance-les.

### "PEXELS_API_KEY manquante"

→ Skipper l'étape Pexels OU créer un fichier `.env` à la racine avec `PEXELS_API_KEY=ta_cle`.

### Colab T4 timeout

→ Réduire les volumes : `--n 1000` au lieu de `--n 5000` pour CIFAKE/ArtiFact, `--per-cat 50` au lieu de `200` pour SDXL, `--per-class 50` pour les textes. Le modèle sera moins bon mais la chaîne tournera.

### "CUDA out of memory" pour BLIP-2 ou LLaVA

→ Utiliser `--limit-to-domain` (calcul uniquement sur le sous-ensemble habitation, ~3 000 images au lieu de ~80 000).

### Streamlit échoue avec "Aucun modèle image chargé"

→ Tu n'as pas lancé `train_image_model.py`. Sans modèle entraîné, l'app affiche le formulaire mais ne peut pas calculer de score.

### "ModuleNotFoundError: open_clip"

→ `pip install open-clip-torch>=2.24` (déjà dans requirements.txt mais peut nécessiter une réinstall si l'env a changé).

---

## Pipeline minimal en 1 heure (si vraiment serré)

Si tu n'as qu'une heure pour avoir une démo qui tourne :

```bash
# A. Setup (10 min)
pip install -r requirements.txt
python -m spacy download fr_core_news_md

# B. Données minimales (30 min)
python 03_src/data/download_cifake.py --n 1000
python 03_src/data/scrape_wikimedia.py --max 100
python 03_src/data/build_dataset.py --no-phash

# C. Features minimales (15 min sur GPU, plus long sur CPU)
python 03_src/features/extract_clip.py --batch 16

# D. Entraînement (3 min)
python 03_src/models/train_image_model.py --variant lite

# E. Calibration (30 sec)
python 03_src/utils/calibration.py --variant lite

# F. App (instantané)
streamlit run 05_app/app.py
```

Tu auras une démo avec le formulaire complet, le scoring image-only, et la heatmap d'explication. Sans BLIP-2/LLaVA/Mistral, sans le score multimodal, mais c'est largement suffisant pour montrer le concept.

---

## Récap des fichiers attendus à la fin

```
01_data/external/cifake/...       (~10 000 images)
01_data/external/artifact/...      (~10 000 images)
01_data/raw/wikimedia/...          (~1 200 images)
01_data/raw/pexels/...             (~800 images)
01_data/raw/personal/...           (~60 images)
01_data/synthetic/sdxl/...         (~800 images)
01_data/synthetic/claim_texts.parquet  (1 200 textes)
01_data/processed/dataset.parquet  (manifest consolidé)
01_data/processed/dataset_with_lmm.parquet  (manifest enrichi LMM)
01_data/processed/claim_texts_with_features.parquet  (textes enrichis)

04_models/clip_embeddings.npy
04_models/image_model_lite.joblib
04_models/image_model_full.joblib
04_models/multimodal_model_lite.joblib
04_models/multimodal_model_full.joblib
04_models/thresholds_image_lite.json
04_models/thresholds_image_full.json
04_models/baseline_metrics.json
04_models/image_model_lite_metrics.json
04_models/image_model_full_metrics.json
04_models/multimodal_model_lite_metrics.json
04_models/multimodal_model_full_metrics.json
```

Une fois tout ça en place, ton projet est complet : data + modèles + app + métriques. Il ne te reste qu'à reporter les chiffres dans le rapport et le PPT, prendre tes photos perso, et préparer la démo de soutenance.

Bon courage, vous y êtes presque.
