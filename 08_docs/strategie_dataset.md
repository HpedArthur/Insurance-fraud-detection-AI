# Stratégie dataset — Note technique

> Risque numéro 1 du projet. À lire avant toute action en semaine 1.

## 1. Le problème honnêtement posé

Construire un classifieur « image authentique vs image générée par IA » dans le contexte assurance habitation suppose deux jeux de données :
- des **photos authentiques de sinistres habitation** ;
- des **photos synthétiques** des mêmes types de sinistres.

La photo authentique de sinistre **réel d'assuré** n'est disponible nulle part en accès ouvert. Les bases comme ISO ClaimSearch, Shift Technology, ou les internes des assureurs sont fermées. Aucune autorité publique ne diffuse de tels jeux de données pour des raisons évidentes de RGPD et de secret commercial.

Il faut accepter ce constat dès le départ et adapter la définition opérationnelle de « image réelle » :

> Dans ce projet, **« image réelle »** désigne une photographie authentique (capturée par un appareil photo réel) représentant un type de scène cohérent avec un sinistre habitation : dégât des eaux, incendie, bris de vitre, dégradation post-cambriolage. Ce ne sont pas des photos issues de dossiers de sinistre réels.

Cette approximation doit être assumée explicitement dans le rapport (section Limites) et dans la soutenance. C'est même un point de maturité.

## 2. Architecture du dataset en deux étages

### Étage 1 — Dataset générique (volume)

Sert à **entraîner et calibrer** le pipeline de détection « réel vs synthétique ». Pas spécifique à l'habitation.

| Source | Description | Volume utilisé | Accès |
|---|---|---|---|
| **CIFAKE** | 60 000 paires CIFAR-10 (réelles) + équivalents Stable Diffusion 1.4 | 60 000 | HuggingFace `dragonintelligence/CIFAKE-image-dataset` |
| **ArtiFact** | Réels (COCO, FFHQ, ImageNet, AFHQ) + synthétiques de 25 générateurs | sous-set 20 000 | HuggingFace `Hemg/AI-Generated-vs-Real-Images-Datasets` |
| **GenImage** (optionnel) | Référentiel ImageNet + 8 générateurs | sous-set 10 000 | https://github.com/GenImage-Dataset/GenImage |

**Total étage 1** : ~80 000 images. Classes équilibrées par construction.

### Étage 2 — Dataset spécifique habitation (qualité métier)

Sert à **valider** la capacité du modèle à transférer au domaine cible.

#### 2.1 Photos authentiques

Sources libres de droits, scraping API :

| Plateforme | API / Méthode | Licence | Mots-clés FR/EN |
|---|---|---|---|
| Pexels | API gratuite (200 req/h) | CC0 / Pexels License | `dégât des eaux`, `incendie maison`, `vitre cassée`, `home damage`, `water leak`, `fire damage interior`, `broken window` |
| Unsplash | API gratuite (50 req/h) | Unsplash License | mêmes mots-clés |
| Pixabay | API gratuite | Pixabay License | mêmes mots-clés |
| Wikimedia Commons | Catégories (`Fire damage`, `Water damage`) | CC variées | filtrer par licence CC0/CC-BY |

**Cible : 1 500 images authentiques** réparties équitablement sur 4 catégories (dégât des eaux / incendie / bris de vitre / vandalisme intérieur), soit ~375 par catégorie.

#### 2.2 Photos synthétiques

Génération **par nous** avec Stable Diffusion XL Turbo (téléchargeable via `diffusers`, gratuit, exécutable sur Colab T4). Avantages :

- contrôle total du prompt, donc équilibrage par catégorie ;
- traçabilité de la chaîne de génération ;
- comparaison équitable aux réelles (pas de biais source).

**Prompts type** (à varier par catégorie) :

```
- "interior of a flooded apartment, water damage on wooden floor, photorealistic, dslr photo, natural light"
- "burned kitchen after fire, soot on walls, charred furniture, insurance claim photo"
- "broken window in a living room, glass shards on floor, after burglary"
- "vandalized apartment interior, overturned furniture, graffiti on wall"
```

**Cible : 1 500 images synthétiques**, ~375 par catégorie. Génération en batch sur Colab : ~30 minutes.

**Diversification critique** : générer en variant les seeds, les samplers (DPM++ 2M, Euler), et inclure **au moins 200 images d'un autre générateur** (Flux Schnell ou SD 1.5) pour évaluer la généralisation cross-générateur — c'est la question piège type que l'enseignant peut poser.

#### 2.3 Splits

Le dataset spécifique habitation (3 000 images) est utilisé en **validation et test seulement** pour la version finale du modèle. Stratification par catégorie ET par classe.

```
Étage 1 (générique, ~80 000) :
  - train : 80 %
  - val : 10 %
  - test : 10 %

Étage 2 (habitation, ~3 000) :
  - val_domain : 30 %
  - test_domain : 70 %  (jamais touché jusqu'à l'évaluation finale)
```

## 3. Métadonnées capturées par image

Stockées dans le `dataset.parquet` final :

| Colonne | Description |
|---|---|
| `image_path` | Chemin relatif |
| `label` | 0 = réelle, 1 = synthétique |
| `category` | water / fire / glass / vandalism / generic |
| `source` | pexels / unsplash / pixabay / wikimedia / sdxl_turbo / flux / cifake / artifact |
| `generator_model` | nom modèle si synthétique, sinon NaN |
| `prompt` | prompt si généré par nous |
| `width`, `height` | dimensions originales |
| `exif_present` | bool — présence de métadonnées EXIF |
| `exif_camera` | modèle d'appareil si présent |
| `exif_datetime` | date EXIF si présente |
| `phash` | hash perceptuel (64 bits) — pour détection de doublons |
| `split` | train / val / test / val_domain / test_domain |

## 4. Volumétrie totale et stockage

- ~83 000 images, taille moyenne 200 Ko après resize 512px → **~17 Go**.
- Trop pour Git LFS gratuit (1 Go). Stockage : **HuggingFace Datasets** (gratuit, illimité pour datasets publics) ou Google Drive partagé.
- Le repo Git ne contient **que le code et le manifest parquet** des chemins. Les images vivent ailleurs.

## 5. Biais et limites à documenter

À mentionner explicitement dans la section Limites du rapport :

1. **Biais source réelle** : nos images « réelles » sont des photos pro/semi-pro de Pexels/Unsplash. Une photo prise par un assuré au smartphone aura une distribution différente (qualité, cadrage, EXIF smartphone vs reflex).
2. **Biais source synthétique** : le modèle peut apprendre la signature spécifique de SDXL Turbo plutôt qu'une notion générale de « synthétique ». L'évaluation cross-générateur quantifie ce risque.
3. **Biais catégoriel** : seulement 4 types de dégâts couverts ; un sinistre rare (foudre, dégât animal) sera mal classé.
4. **Pas de cas-limites adversariaux** : retouche partielle d'image authentique, image générée puis re-photographiée à l'écran, etc. Out of scope.
5. **Pas de validation par expert métier** : aucun gestionnaire de sinistre n'a annoté nos images. Le label « réelle » est une approximation contextuelle.

## 6. Calendrier de constitution (semaine 1)

| Jour | Action | Sortie |
|---|---|---|
| J+1 | Téléchargement CIFAKE (HF datasets) | 60k images, manifest |
| J+2 | Téléchargement ArtiFact subset | +20k, manifest concaténé |
| J+3 | Scraping Pexels + Unsplash + Pixabay | ~1 500 réelles habitation |
| J+4 | Tri manuel rapide (15 min/personne) | exclusion des images hors sujet |
| J+4-5 | Génération SDXL Turbo sur Colab (4h) | 1 500 synthétiques + 200 cross-gen |
| J+6 | Calcul EXIF + phash + dé-duplication | dataset.parquet final |
| J+6 | EDA : distribution catégories, doublons, fuites split | notebook EDA |

## 7. Quick-start technique

À exécuter dès J+1 sur la machine de chacun pour valider que tout fonctionne :

```bash
pip install -r requirements.txt
huggingface-cli login   # créer un compte HF gratuit
python 03_src/data/download_cifake.py --n 1000  # smoke test sur 1000 images
```

Si ça passe, le pipeline est prêt à être étendu.
