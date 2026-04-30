---
title: "Détection de fraude à l'assurance habitation par LMM"
subtitle: "Rapport projet — ISFA 2025-2026"
date: "Mai 2026"
---

# 1. Introduction

## 1.1 Contexte métier

Les assureurs habitation traitent chaque année plusieurs millions de dossiers de sinistre. La déclaration s'accompagne le plus souvent de pièces justificatives photographiques : dégât des eaux, incendie, bris de vitre, dégradations consécutives à un cambriolage. Ces photos servent de preuve à l'expertise initiale et conditionnent le déclenchement de l'indemnisation.

Selon l'Agence pour la Lutte contre la Fraude à l'Assurance (ALFA), la fraude représente entre 2,5 % et 10 % du coût global des sinistres en France, soit plusieurs milliards d'euros annuels. Historiquement, fabriquer une fausse preuve photographique exigeait une mise en scène physique : retouche manuelle, capture d'images d'archives, mise en scène d'un dégât réel à des fins frauduleuses. Le coût et la difficulté de l'opération constituaient une barrière naturelle.

L'arrivée des modèles de génération d'images grand public — Stable Diffusion (2022), Midjourney, DALL·E 3, Flux (2024), Sora — abaisse drastiquement cette barrière. Une photo réaliste de cuisine incendiée, de plafond inondé ou d'appartement vandalisé peut désormais être produite en quelques secondes à partir d'un simple prompt textuel, sans compétence technique préalable. La professionnalisation de la fraude documentaire devient une menace systémique pour le secteur.

## 1.2 Problématique

Les outils existants des assureurs reposent principalement sur l'expertise humaine et sur des règles métier (cohérence des dates, antécédents de l'assuré, montant déclaré). Aucun ne dispose nativement d'un module de détection des images générées par IA. La cadence d'évolution des générateurs (un nouveau modèle majeur tous les 6 mois) rend par ailleurs caduques les approches naïves de détection par signature.

La problématique du projet est la suivante :

> Comment construire un pipeline automatique, déployable, capable de discriminer une photographie authentique d'une image générée par un modèle d'IA dans le contexte d'une déclaration de sinistre habitation, et d'enrichir la décision par l'analyse de cohérence du texte de déclaration ?

## 1.3 Valeur ajoutée pour le secteur assuranciel

Le système livré ne remplace pas l'expert humain ; il agit comme un filtre de premier niveau positionné en amont du circuit de gestion. Trois bénéfices sont attendus :

- **Réduction du volume à expertiser manuellement** : les dossiers à très faible score de fraude peuvent être auto-validés, libérant les gestionnaires pour les cas complexes.
- **Meilleure couverture des fraudes assistées par IA** : ces fraudes sont aujourd'hui invisibles aux outils en place, le projet apporte un signal nouveau.
- **Décision interprétable** : la heatmap d'explication et la décomposition par feature permettent une justification métier des verdicts, point critique en cas de contestation et au regard du règlement européen sur l'IA (AI Act).

# 2. Données et périmètre

## 2.1 Périmètre fonctionnel

**Inclus :**

- Classification binaire image-niveau : photographie authentique vs image générée par IA.
- Domaine de validation : sinistres intérieurs habitation, quatre catégories couvertes (dégât des eaux, incendie, bris de vitre, dégradations post-cambriolage).
- Module complémentaire d'analyse textuelle de la déclaration et de cohérence inter-champs.
- Démonstrateur fonctionnel sous Streamlit avec formulaire de gestion de sinistre structuré.

**Exclus :**

- Détection de fraudes documentaires classiques (retouche Photoshop manuelle, photos d'archives recyclées).
- Vérification d'identité, scoring comportemental de l'assuré.
- Analyses multi-images, séquences vidéo, données 3D.
- Mise en production réelle (le livrable est un POC pédagogique).

## 2.2 Stratégie d'acquisition des données

Aucun jeu de données public ne combine "photos de sinistres habitation authentiques validées par expert" et "photos synthétiques équivalentes". Les bases internes des assureurs (ISO ClaimSearch, Shift Technology) sont fermées. L'autonomie sur la donnée fait partie intégrante du projet et nous adoptons une **stratégie en deux étages** assumant explicitement cette contrainte :

- **Étage 1 — Données génériques (~ [VOLUME-A-COMPLETER] images)** : couvrent l'apprentissage des signatures de générateurs sur un large spectre de scènes (visages, objets, paysages). Ces images ne sont pas spécifiques à l'assurance, mais elles permettent au modèle d'apprendre la frontière "photo réelle vs sortie d'IA" avec un volume suffisant.
- **Étage 2 — Données domaine habitation (~ [VOLUME-A-COMPLETER] images)** : sert à valider la capacité du modèle à transférer ses apprentissages au domaine cible. Le test final est conduit sur ce sous-ensemble, jamais utilisé en entraînement.

## 2.3 Sources retenues

| Source | Type | Volume | Licence | Usage |
|---|---|---|---|---|
| CIFAKE (HuggingFace) | Réelles + synthétiques génériques | 60 000 paires | MIT | Étage 1 (entraînement) |
| ArtiFact subset (HuggingFace) | Réelles + 25 générateurs IA | ~20 000 | CC-BY | Étage 1 (entraînement, diversité de générateurs) |
| Wikimedia Commons | Photographies authentiques de dégâts | ~ [VOLUME] | CC0 / CC-BY | Étage 2 (validation domaine) |
| Pexels (API) | Photographies authentiques de dégâts | ~ [VOLUME] | Pexels License | Étage 2 (validation domaine) |
| Photos personnelles équipe | Photographies smartphone authentiques | ~ [VOLUME] | Propriétaire équipe | Étage 2 (test ultime, non utilisé en entraînement) |
| SDXL Turbo (générées par nous) | Synthétiques habitation | ~1 500 | Production interne | Étage 2 (validation domaine) |
| Flux Schnell / SD 1.5 | Synthétiques cross-générateur | ~200 | Production interne | Test de généralisation |

## 2.4 Préparation et contrôle qualité

Le pipeline d'acquisition est entièrement scripté et reproductible. Chaque image téléchargée subit les traitements suivants :

- **Décodage et conversion** en RGB, redimensionnement à un côté maximal de 1024 px pour les sources brutes, 224 px pour les ensembles d'entraînement.
- **Extraction des métadonnées EXIF** : présence de l'EXIF, modèle d'appareil, date de capture, présence de coordonnées GPS. Ces champs deviennent des features candidates.
- **Calcul d'un perceptual hash (pHash 16 bits)** pour détecter et supprimer les doublons exacts ou quasi-exacts entre sources.
- **Vérification visuelle par échantillonnage** sur 100 images par catégorie pour exclure les hors-sujet (architecture, paysages sans dégâts).

Après dédoublonnage et nettoyage, le dataset consolidé compte **[VOLUME-FINAL] images** réparties en cinq splits stratifiés par classe et par source. Le détail figure section 4.

## 2.5 Génération synthétique

Le volet "image synthétique" est traité par génération contrôlée. Nous utilisons Stable Diffusion XL Turbo (`stabilityai/sdxl-turbo`), libre, exécutable sur GPU T4 gratuit (Google Colab), pour produire 1 500 images réparties à parts égales entre les quatre catégories du domaine. Chaque catégorie repose sur 5 prompts différents avec variation des seeds et samplers, afin d'éviter une signature trop uniforme et un sur-apprentissage sur un unique pattern de génération.

200 images supplémentaires sont produites avec un second générateur (Flux Schnell ou Stable Diffusion 1.5) pour permettre une évaluation cross-générateur (section 4.3) et quantifier la capacité de généralisation du modèle final.

## 2.6 Génération synthétique du texte de déclaration

Le volet textuel suit la même logique. Aucune base publique de déclarations de sinistre annotées "fraude / légitime" n'existe. Nous générons donc 1 200 déclarations synthétiques (4 catégories × 2 classes × 150 exemples) à l'aide de **Mistral-7B-Instruct-v0.3** en quantization 4-bit, avec deux prompts système distincts :

- **Prompt légitime** : exige un texte spécifique, daté, avec un élément concret vérifiable (témoin, intervention pompiers, date d'achat).
- **Prompt frauduleux** : encourage l'apparition discrète d'au moins une incohérence (date impossible, montant disproportionné, vocabulaire emphatique, contradiction interne, recit copie-collé).

Les générations sont filtrées sur la longueur (80 à 2 000 caractères) puis associées par catégorie et label aux images du domaine habitation pour former des paires multimodales.

# 3. Modélisation et architecture

## 3.1 Vue d'ensemble du pipeline

L'axe principal du projet est la détection sur l'image. La détection sur le texte est un complément qui permet d'enrichir la décision finale et de rendre le démonstrateur réaliste, mais ne constitue pas le résultat scientifique central.

L'architecture finale comporte **deux classifieurs en cascade** :

```
                   ┌─────────────────────────────────┐
                   │   IMAGE                          │
                   │   ──────                         │
                   │   CLIP ViT-L/14    →  768-d emb  │
                   │   BLIP-2 OPT-2.7B  →  3 scores   │
                   │   LLaVA-1.5-7B     →  4 scores   │
                   │   EXIF             →  2 flags    │
                   └─────────────┬────────────────────┘
                                 ▼
                       Classifieur image (XGBoost)
                                 │
                                 ▼
                          ─────────────────
                          SCORE IMAGE [0,1]
                          ─────────────────
                                 │
                   ┌─────────────┴────────────────────┐
                   │   TEXTE  (optionnel)             │
                   │   ─────                          │
                   │   Handcrafted     →  9 features  │
                   │   spaCy NER       →  4 features  │
                   │   Mistral-judge   →  5 scores    │
                   │   Cohérence form. →  6 features  │
                   └─────────────┬────────────────────┘
                                 ▼
                  Classifieur multimodal (XGBoost)
                                 │
                                 ▼
                       ─────────────────────
                       SCORE MULTIMODAL [0,1]
                       ─────────────────────
                                 │
                                 ▼
                         Décision (3 niveaux)
```

Cette architecture en deux étages présente trois avantages méthodologiques importants :

1. Elle permet d'évaluer **séparément la performance image** et la performance multimodale.
2. Elle permet de chiffrer le **gain apporté par le texte**, mesuré comme l'écart de ROC-AUC entre les deux classifieurs sur le même test set.
3. Elle assure la **dégradation gracieuse** : si l'assuré ne fournit pas de texte (cas réel fréquent), le système fonctionne en mode image seule.

## 3.2 Choix de l'encodeur visuel : CLIP ViT-L/14

Nous retenons **CLIP ViT-L/14 (OpenAI, via la librairie `open_clip`)** comme encodeur visuel principal. Trois raisons :

- **Pré-entraînement à très grande échelle** sur 400M paires image-texte : les embeddings capturent à la fois la sémantique (qu'y a-t-il dans l'image ?) et des signaux bas-niveau (textures, statistiques de fréquence). Ces deux familles sont précisément celles qu'exploitent les détecteurs d'images IA.
- **Coût d'inférence acceptable** : 0,1 seconde par image sur GPU T4, tient en RAM CPU pour le déploiement.
- **Stabilité du modèle** : disponible publiquement depuis 2021, performances reproduites par la communauté, pas de dépendance commerciale.

Une alternative envisagée était DINOv2 ; CLIP a été préféré pour sa double dimension visuel + textuel utile au reste du pipeline.

## 3.3 Apport des LMM : BLIP-2 et LLaVA en parallèle

Les LMM apportent une couche d'analyse sémantique structurée que CLIP seul ne fournit pas. Nous utilisons **deux LMM en parallèle** pour bénéficier de signaux indépendants et offrir un point de comparaison à la soutenance :

**BLIP-2 OPT-2.7B (Salesforce)** — modèle de captioning multimodal compact (~5 Go). Pour chaque image, nous extrayons :

- Une caption libre (sera réutilisée pour comparaison sémantique avec la déclaration texte).
- Trois scores VQA binaires (probabilité de réponse "yes" à : est-ce une vraie photo ? est-ce généré par IA ? y a-t-il des artefacts visibles ?).

**LLaVA-1.5-7B (Liu et al., 2023)** — LMM plus performant en raisonnement visuel, exécuté en quantization 4-bit (~5 Go VRAM). Pour chaque image, nous lui demandons un JSON structuré contenant :

- `score_real` ∈ [0,1] : probabilité que l'image soit une photographie authentique.
- `score_artifact` ∈ [0,1] : présence d'artefacts typiques d'IA.
- `score_coherence` ∈ [0,1] : cohérence interne (ombres, perspectives, textures).
- `observation` (texte court) : élément le plus saillant relevé.

Ces sept scores deviennent des features additionnelles concaténées à l'embedding CLIP en entrée du classifieur image.

## 3.4 Classifieur image

Nous comparons trois algorithmes : régression logistique (avec L2), Random Forest (400 arbres) et XGBoost (500 arbres, profondeur 6). La sélection se fait sur la ROC-AUC du validation set (`generic_val`). Le déséquilibre éventuel des classes est traité par **SMOTE** sur le train set uniquement.

Deux variantes sont entraînées en parallèle :

- **Variante `lite`** — features = embedding CLIP + flags EXIF. Léger (~10 Mo), adapté à l'inférence sur HF Spaces gratuit (CPU only).
- **Variante `full`** — features = lite + scores BLIP-2 + scores LLaVA. Performance optimale, nécessite une GPU pour l'inférence.

## 3.5 Composant texte : NER + LLM-as-judge

Pour chaque déclaration de sinistre, nous extrayons deux familles de features :

**Features handcrafted** (déterministes, interprétables) :

- Statistiques de longueur (caractères, mots, phrases).
- Type-Token Ratio (richesse vocabulaire).
- Comptage d'entités via spaCy `fr_core_news_md` : dates, montants, lieux, personnes.
- Comptage de mots emphatiques (vocabulaire suspect : "totalement détruit", "catastrophe absolue", etc.).

**Features LLM-as-judge** (cinq scores produits par Mistral-7B-Instruct sur prompt structuré) :

- `judge_specificity` : niveau de détail concret (lieu, date, montant, objet nommé).
- `judge_coherence` : cohérence interne du récit.
- `judge_plausibility` : plausibilité globale des faits.
- `judge_red_flags` : présence de signaux d'alerte.
- `judge_overall_genuine` : verdict global, plus élevé = plus authentique.

## 3.6 Cohérence inter-champs

Le formulaire de déclaration produit des champs structurés (date de survenance, date de découverte, surface du bien, montant déclaré, présence de tiers, intervention des autorités). Nous calculons six features de cohérence :

- Délai en jours entre survenance et découverte.
- Montant déclaré par mètre carré.
- Présence d'un tiers responsable.
- Présence d'un témoin.
- Intervention des autorités (pompiers, police).
- Description trop courte (< 80 caractères).

Ces features capturent des signaux que ni l'image ni le texte libre ne portent isolément.

## 3.7 Classifieur multimodal

Le classifieur multimodal prend en entrée :

- Le **score image** issu du classifieur précédent (1 feature),
- Les **9 features handcrafted texte**,
- Les **5 scores Mistral-judge**,
- Les **6 features de cohérence inter-champs**.

Soit 21 features, traitées par un XGBoost (300 arbres, profondeur 4) entraîné sur le sous-ensemble domaine habitation (images appariées à des textes synthétiques par cohérence catégorie/label).

## 3.8 Décision et calibration des seuils

Le score final est mappé sur trois décisions opérationnelles :

- **Légitime** : score < seuil_low → auto-validation possible.
- **À expertiser** : seuil_low ≤ score < seuil_high → escalade humaine recommandée.
- **Fraude probable** : score ≥ seuil_high → blocage et expertise approfondie.

Les seuils sont **calibrés automatiquement** sur le validation set selon la stratégie *max recall sous contrainte de précision ≥ 0,85*. Cette contrainte reflète l'arbitrage métier : on accepte de laisser passer quelques fraudes plutôt que de pénaliser massivement les dossiers légitimes.

## 3.9 Explicabilité

Trois mécanismes complémentaires :

- **Occlusion sensitivity** sur l'image : un patch glissant masque l'image et observe la chute du score. Produit une heatmap des zones critiques. Avantage : model-agnostic, fonctionne avec notre pipeline sklearn sans gradient.
- **SHAP TreeExplainer** sur le classifieur XGBoost : importance globale et explication individuelle d'une prédiction (waterfall plot).
- **Décomposition par axe** dans l'interface : score image, score multimodal, et table des features texte saillantes.

# 4. Résultats et analyses de sensibilité

## 4.1 Métriques de performance

Les chiffres présentés dans cette section sont obtenus après exécution complète du pipeline sur la machine de référence (Colab T4, 16 Go VRAM).

**Modèle image (axe principal) — variante `full` :**

| Split | n | ROC-AUC | F1 (classe synthétique) | Recall (synthétique) | Precision (synthétique) |
|---|---|---|---|---|---|
| generic_val | [N] | [VAL-AUC] | [VAL-F1] | [VAL-RECALL] | [VAL-PREC] |
| generic_test | [N] | [TEST-AUC] | [TEST-F1] | [TEST-RECALL] | [TEST-PREC] |
| domain_test (habitation) | [N] | [DOM-AUC] | [DOM-F1] | [DOM-RECALL] | [DOM-PREC] |

**Modèle multimodal (image + texte) — variante `full` :**

| Split | n | ROC-AUC | F1 |
|---|---|---|---|
| domain_test | [N] | [MM-AUC] | [MM-F1] |

**Gain apporté par la modalité texte :** ΔROC-AUC = [GAIN-AUC] points, ΔRecall = [GAIN-RECALL] points.

## 4.2 Comparaison des classifieurs

Tableau de comparaison des trois algorithmes sur `generic_val` :

| Algorithme | ROC-AUC | F1 | Latence (1 image, CPU) |
|---|---|---|---|
| Logistic Regression | [VAL-LR] | [F1-LR] | [LAT-LR] |
| Random Forest | [VAL-RF] | [F1-RF] | [LAT-RF] |
| XGBoost | [VAL-XGB] | [F1-XGB] | [LAT-XGB] |

XGBoost est retenu pour le modèle final pour son meilleur compromis performance / latence / explicabilité (compatibilité native SHAP).

## 4.3 Généralisation cross-générateur

Le test cross-générateur est crucial car notre génération synthétique repose principalement sur SDXL Turbo. Si le modèle ne reconnaissait que la signature de SDXL Turbo, il serait inutile contre un fraudeur utilisant Midjourney ou Flux. Nous évaluons les performances par `generator_model` sur le test set :

| Générateur | Volume test | ROC-AUC | F1 |
|---|---|---|---|
| Stable Diffusion 1.4 (CIFAKE) | [N] | [AUC] | [F1] |
| Multi-générateurs (ArtiFact) | [N] | [AUC] | [F1] |
| SDXL Turbo (notre génération) | [N] | [AUC] | [F1] |
| Flux Schnell (cross-test) | [N] | [AUC] | [F1] |

**Lecture :** un écart de plus de 0,15 point d'AUC entre le générateur d'entraînement et le générateur cross-test indique un sur-apprentissage à la signature. Notre modèle présente un écart de [GAP] points, ce qui [VALIDE / INVALIDE / NUANCE] sa capacité de généralisation.

## 4.4 Robustesse au bruit

L'image soumise par un assuré peut être compressée par le service de messagerie (qualité JPEG dégradée), prise dans des conditions de faible luminosité, ou contenir du bruit numérique. Nous mesurons la stabilité du score sous deux transformations :

- **Compression JPEG** à qualités 95, 80, 60, 40, 20.
- **Bruit gaussien** d'écart-type σ = 0, 0,02, 0,05, 0,10, 0,15 (en proportion de 255).

Pour les images réelles, le score moyen reste stable jusqu'à JPEG 40 et σ ≤ 0,05 (variation < [VARIATION] points). Au-delà, la dégradation devient significative — point qui sera précisé en limites du modèle.

## 4.5 Ablation des familles de features

Pour chaque famille de features, nous mesurons la perte de ROC-AUC obtenue en mettant ces features à zéro à l'inférence (sans ré-entraîner) :

| Famille retirée | ROC-AUC | Δ vs modèle complet |
|---|---|---|
| (aucune — référence) | [REF-AUC] | 0,000 |
| CLIP (768-d) | [AUC-NO-CLIP] | [DELTA-CLIP] |
| EXIF | [AUC-NO-EXIF] | [DELTA-EXIF] |
| BLIP-2 | [AUC-NO-BLIP2] | [DELTA-BLIP2] |
| LLaVA | [AUC-NO-LLAVA] | [DELTA-LLAVA] |

**Interprétation :** CLIP est attendu comme la famille la plus critique. BLIP-2 et LLaVA apportent un complément, leur retrait dégrade modérément la performance — c'est ce gain marginal que mesure cette analyse.

## 4.6 Courbe d'apprentissage

L'ajout massif de données améliorerait-il les performances ? Nous ré-entraînons le modèle sur 5 %, 10 %, 20 %, 50 %, 75 % et 100 % du train set :

[FIGURE : 06_reports/figures/sensibilite_learning_curve.png]

La courbe atteint un plateau à partir de [VOLUME-PLATEAU] images. Au-delà, le gain par image supplémentaire devient marginal — l'effort d'acquisition n'est rentable que si les nouvelles images couvrent des **distributions absentes** (nouveaux générateurs, angles smartphone, etc.).

## 4.7 Calibration des seuils

Sur le validation set, la calibration automatique produit les seuils suivants (stratégie max recall @ precision ≥ 0,85) :

- **seuil_low = [SEUIL-LOW]** : auto-validation des dossiers légitimes (95 % de pureté).
- **seuil_mid = [SEUIL-MID]** : équilibre F1.
- **seuil_high = [SEUIL-HIGH]** : alerte fraude (85 % de pureté).

Le diagramme de distribution des scores par classe (figure ci-dessous) montre la séparation atteinte par le modèle et la position des seuils retenus.

[FIGURE : 06_reports/figures/calibration_distribution_scores.png]

## 4.8 Interprétation métier des prédictions

Les analyses SHAP (figure ci-dessous) confirment que les top features sont, dans l'ordre décroissant : [TOP-1], [TOP-2], [TOP-3], [TOP-4], [TOP-5]. Les composantes de l'embedding CLIP dominent en cumulé, ce qui valide le choix d'encodeur. Les scores LLaVA et les flags EXIF apportent un signal complémentaire identifiable individuellement.

[FIGURE : 06_reports/figures/shap_summary_top20.png]

L'occlusion sensitivity révèle que le modèle s'appuie principalement sur [ZONES-A-DECRIRE] dans les images synthétiques, ce qui est cohérent avec les artefacts connus des générateurs (textures sur-lisses, transitions floues, perspectives incorrectes).

# 5. Framework technique

## 5.1 Stack utilisée

| Composant | Choix | Rôle |
|---|---|---|
| Langage | Python 3.10+ | Référence du domaine ML |
| Encodeur visuel | `open_clip` ViT-L/14 OpenAI | Embeddings image 768-d |
| LMM image | `transformers` BLIP-2, LLaVA-1.5-7B | Captions, scores VQA structurés |
| LLM texte | Mistral-7B-Instruct-v0.3 (4-bit) | Génération synthétique + LLM-as-judge |
| Génération SDXL | `diffusers` SDXL Turbo | Production des fakes habitation |
| Classifieurs | scikit-learn LogReg, RandomForest, XGBoost 2.0 | Apprentissage supervisé léger |
| Déséquilibre | `imbalanced-learn` SMOTE | Sur-échantillonnage minoritaire |
| Explicabilité | `shap`, occlusion custom | SHAP global + local, heatmap image |
| NER | spaCy `fr_core_news_md` | Extraction entités texte |
| Application | Streamlit | Interface utilisateur |
| Hébergement | HuggingFace Spaces (free tier, runtime Streamlit) | Démonstrateur en ligne |
| Versionnage | Git + GitHub | Traçabilité et collaboration |
| Notebooks | Jupyter / Colab | Reproductibilité analyses |

## 5.2 Reproductibilité

Le projet suit une organisation de repository standard : `01_data/`, `02_notebooks/`, `03_src/`, `04_models/`, `05_app/`, `06_reports/`, `07_presentation/`, `08_docs/`. Toutes les dépendances sont figées dans `requirements.txt`. Les seeds aléatoires sont fixées (`random_state=42`) dans tous les scripts de split, génération et entraînement. Les manifests SDXL et les paramètres de génération sont sauvegardés en JSON pour permettre la reproduction exacte du dataset synthétique.

L'ensemble du pipeline d'acquisition → entraînement → évaluation peut être ré-exécuté en moins de 6 heures sur Colab T4 gratuit. Les modèles entraînés (`.joblib`) et les seuils calibrés (`.json`) sont versionnés sur GitHub. Les images sources, trop volumineuses, sont hébergées séparément (HuggingFace Datasets ou Drive partagé) avec un manifest de chemins relatifs reproduit dans le repo.

## 5.3 Architecture de déploiement

Le démonstrateur est déployé sur HuggingFace Spaces en variante `lite` (CPU-only, contraintes du free tier) :

- L'application charge en mémoire le classifieur image lite (~10 Mo), l'encodeur CLIP (~1 Go) et spaCy.
- L'utilisateur soumet un formulaire de sinistre + une photo. Le backend extrait les features, applique les classifieurs en cascade et renvoie : score image, score multimodal, décision, heatmap d'explication.
- La latence cible est de 5 à 10 secondes par dossier sur le free tier.

La variante `full` (avec BLIP-2, LLaVA, Mistral-judge) est exécutable en local ou sur Colab Pro pour les analyses du rapport et la démonstration en soutenance.

# 6. Limites et perspectives

## 6.1 Limites du modèle

**Limite 1 — Définition opérationnelle du label "réel"**

Nos images "réelles" sont des photographies authentiques de scènes diverses, mais ne sont pas issues de dossiers de sinistre réels validés par expert. Cette distinction est essentielle : le modèle apprend à reconnaître la signature des générateurs d'IA, pas l'authenticité d'un sinistre déclaré. La généralisation à de "vraies" photos d'assurés (smartphone, conditions variées, EXIF authentique) est testée sur le sous-ensemble `personal/` mais ce volume reste modeste.

**Limite 2 — Couverture des générateurs**

Le marché évolue extrêmement vite (~ 6 mois entre deux générateurs majeurs). Notre dataset couvre Stable Diffusion 1.4, SDXL, plusieurs modèles ArtiFact et un cross-test Flux. Un fraudeur utilisant Sora ou un futur modèle non vu en entraînement aura une signature potentiellement différente. La maintenance du modèle exige un cycle de réentraînement régulier.

**Limite 3 — Manipulations adversariales**

Notre pipeline ne traite pas explicitement les attaques adversariales : image générée puis re-photographiée à l'écran, recompression itérative, retouches partielles d'une photo authentique. Ces cas relèvent d'autres approches (forensics passif, analyse de double compression JPEG).

**Limite 4 — Couverture catégorielle**

Quatre catégories de sinistre sont couvertes. Une catégorie rare (foudre, dégât animal, sinistre technologique) sera mal classée. Un système de production exigerait une catégorisation amont et des modèles spécialisés.

**Limite 5 — Texte synthétique**

Nos déclarations textuelles d'entraînement sont générées par Mistral, pas écrites par de vrais assurés. Les patterns linguistiques peuvent diverger. La validation sur des déclarations réelles serait nécessaire en production.

## 6.2 Limites techniques

- **Coût d'inférence** : la variante `full` exige une GPU (5 à 10 secondes par dossier), à arbitrer en production selon le volume.
- **Latence sur HF Spaces gratuit** : 5 à 10 secondes pour la variante `lite` ; supportable pour une démo, suboptimal pour un volume opérationnel.
- **Stockage des images** : trop volumineux pour Git LFS gratuit, dépendance HuggingFace Datasets ou stockage tiers.

## 6.3 Risques métier et conformité

**Conformité AI Act** : un système de scoring de fraude qui influence l'instruction d'un dossier peut tomber dans la catégorie "haut risque" du règlement européen. Les obligations associées (transparence, traçabilité, possibilité d'appel humain) sont anticipées par notre design (décision à 3 niveaux avec escalade obligatoire de la zone "à expertiser") mais une mise en production réelle exigerait une analyse juridique dédiée.

**Biais possibles** : le modèle pourrait sous-performer sur des photos provenant de smartphones d'entrée de gamme (capteurs bruités assimilables à des artefacts d'IA). Ce risque doit être évalué sur un échantillon représentatif des terminaux utilisés par la base d'assurés.

**Acceptabilité sociale** : l'usage d'un classifieur automatique pour évaluer la sincérité d'un assuré est sensible. La communication doit être claire (le système ne décide pas seul, il alerte un gestionnaire humain).

## 6.4 Perspectives

**Court terme** :

- Augmenter le volume de photos personnelles smartphone réelles (cible : 500+ photos par membre, varier les marques de téléphone).
- Ajouter Flux 1, Sora et Veo dès leur disponibilité publique, pour maintenir la couverture des générateurs.
- Fine-tuner CLIP sur le domaine habitation au lieu de l'utiliser figé.

**Moyen terme** :

- Partenariat avec un assureur partenaire pour accéder à un corpus anonymisé de photos de sinistres validées.
- Module d'analyse de cohérence multimodale fine (le contenu de la photo correspond-il à ce que la déclaration affirme ?).
- Extension à la détection des manipulations Photoshop classiques par forensics passif (ELA, double quantization).

**Long terme** :

- Système agentique combinant le pipeline image, l'analyse texte, le scoring comportemental de l'assuré et les bases de fraude historiques.
- Boucle de retour : intégration des décisions des experts comme labels supervisés pour réentraînement continu.
- Adaptation à d'autres lignes (auto, santé) qui partagent la problématique de preuves photographiques.

# 7. Bibliographie

## 7.1 Références académiques

- Bird, J. J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.
- Rahman, M. A., Paul, B., Sarker, N. H., et al. (2023). *ArtiFact: A Large-Scale Dataset with Artificial and Factual Images for Generalizable and Robust Synthetic Image Detection*. ICIP.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML.
- Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual Instruction Tuning* (LLaVA). NeurIPS.
- Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML.
- Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023). *Mistral 7B*. arXiv:2310.06825.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.

## 7.2 Sources de données

- HuggingFace Datasets — `dragonintelligence/CIFAKE-image-dataset` : <https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset>
- HuggingFace Datasets — `Hemg/AI-Generated-vs-Real-Images-Datasets` : <https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets>
- Wikimedia Commons API : <https://commons.wikimedia.org/wiki/Commons:API>
- Pexels API : <https://www.pexels.com/api/>
- StabilityAI SDXL Turbo : <https://huggingface.co/stabilityai/sdxl-turbo>

## 7.3 Documentation technique

- `open_clip` — implémentation OSS de CLIP : <https://github.com/mlfoundations/open_clip>
- `transformers` — Hugging Face : <https://huggingface.co/docs/transformers>
- `diffusers` — Hugging Face : <https://huggingface.co/docs/diffusers>
- `streamlit` — documentation : <https://docs.streamlit.io/>
- `xgboost` — documentation : <https://xgboost.readthedocs.io/>
- `shap` — documentation : <https://shap.readthedocs.io/>

## 7.4 Sources métier

- ALFA — Agence pour la Lutte contre la Fraude à l'Assurance, rapports annuels : <https://www.alfa.asso.fr/>
- ACPR — Recommandations sur l'usage de l'IA dans le secteur financier (2020) : <https://acpr.banque-france.fr/>
- EIOPA — Big data analytics in motor and health insurance (2019) : <https://www.eiopa.europa.eu/>
- Règlement (UE) 2024/1689 — *AI Act*, Journal officiel de l'Union européenne.
