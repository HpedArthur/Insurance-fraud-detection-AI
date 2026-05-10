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

- **Étage 1 — Données génériques (~ 29 900 images)** : couvrent l'apprentissage des signatures de générateurs sur un large spectre de scènes (visages, objets, paysages). Ces images ne sont pas spécifiques à l'assurance, mais elles permettent au modèle d'apprendre la frontière "photo réelle vs sortie d'IA" avec un volume suffisant.
- **Étage 2 — Données domaine habitation (~ 1 170 images)** : sert à valider la capacité du modèle à transférer ses apprentissages au domaine cible. Le test final est conduit sur ce sous-ensemble, jamais utilisé en entraînement.

## 2.3 Sources retenues

| Source | Type | Volume | Licence | Usage |
|---|---|---|---|---|
| CIFAKE (HuggingFace) | Réelles + synthétiques génériques | 60 000 paires | MIT | Étage 1 (entraînement) |
| ArtiFact subset (HuggingFace) | Réelles + 25 générateurs IA | ~20 000 | CC-BY | Étage 1 (entraînement, diversité de générateurs) |
| Wikimedia Commons | Photographies authentiques de dégâts | ~ 800 | CC0 / CC-BY | Étage 2 (validation domaine) |
| Pexels (API) | Photographies authentiques de dégâts | non utilisé | Pexels License | Source identifiée mais non exploitée (rate limit) |
| Photos personnelles équipe | Photographies smartphone authentiques | à compléter | Propriétaire équipe | Étage 2 (test ultime, non utilisé en entraînement) |
| SDXL Turbo (générées par nous) | Synthétiques habitation | ~1 500 | Production interne | Étage 2 (validation domaine + test cross-générateur) |

## 2.4 Préparation et contrôle qualité

Le pipeline d'acquisition est entièrement scripté et reproductible. Chaque image téléchargée subit les traitements suivants :

- **Décodage et conversion** en RGB, redimensionnement à un côté maximal de 1024 px pour les sources brutes, 224 px pour les ensembles d'entraînement.
- **Extraction des métadonnées EXIF** : présence de l'EXIF, modèle d'appareil, date de capture, présence de coordonnées GPS. Ces champs deviennent des features candidates.
- **Calcul d'un perceptual hash (pHash 16 bits)** pour détecter et supprimer les doublons exacts ou quasi-exacts entre sources.
- **Vérification visuelle par échantillonnage** sur 100 images par catégorie pour exclure les hors-sujet (architecture, paysages sans dégâts).

Après dédoublonnage et nettoyage, le dataset consolidé compte **31 062 images** réparties en cinq splits stratifiés par classe et par source : `generic_train` (≈ 24 000), `generic_val` (≈ 2 990), `generic_test` (≈ 2 990), `domain_train` (≈ 760), `domain_val` (≈ 175), `domain_test` (≈ 820). Le détail figure section 4.

## 2.5 Génération synthétique

Le volet "image synthétique" est traité par génération contrôlée. Nous utilisons Stable Diffusion XL Turbo (`stabilityai/sdxl-turbo`), libre, exécutable sur GPU T4 gratuit (Google Colab), pour produire 1 500 images réparties à parts égales entre les quatre catégories du domaine. Chaque catégorie repose sur 5 prompts différents avec variation des seeds et samplers, afin d'éviter une signature trop uniforme et un sur-apprentissage sur un unique pattern de génération.

L'évaluation cross-générateur (section 4.3) ne s'appuie pas sur des générations additionnelles produites par nos soins, mais sur les **groupes de générateurs déjà présents dans les datasets externes** : Stable Diffusion 1.4 dans CIFAKE, mélange multi-générateurs ("various") dans ArtiFact, comparés à notre génération SDXL Turbo. Cette stratégie est plus économique en ressources GPU et reste pertinente puisqu'elle compare un générateur ancien (SD 1.4) à un générateur moderne (SDXL Turbo). L'extension à Flux Schnell ou Sora est listée en perspective (section 6.4).

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

Empiriquement, c'est la **régression logistique** qui obtient la meilleure ROC-AUC sur le validation set (cf. section 4.2). Ce résultat est cohérent avec la nature des features : un embedding CLIP de 768 dimensions est conçu pour être linéairement séparable (apprentissage contrastif), un classifieur linéaire suffit donc à exploiter l'information. Les modèles à arbres souffrent ici de la haute dimensionnalité et de l'absence d'une feature scalaire dominante.

Deux variantes sont entraînées en parallèle :

- **Variante `lite`** — features = embedding CLIP + flags EXIF (770 features au total). Léger (~10 Mo), adapté à l'inférence sur HF Spaces gratuit (CPU only).
- **Variante `full`** — features = lite + scores BLIP-2 (3) + scores LLaVA (3) (776 features au total). Les scores BLIP-2 et LLaVA ne sont calculés que sur le sous-ensemble domain (1 170 images), ils sont imputés par défaut neutre (0,5) sur les images génériques pour permettre l'entraînement sur le dataset complet. Cette imputation explique que le gain de la variante full sur la variante lite reste modeste sur generic_val.

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
- Les **9 features handcrafted texte** (longueurs, TTR, comptage entités, mots emphatiques),
- Les **5 scores Mistral-judge** (variante `full` uniquement).

Soit **10 features pour le multimodal `lite`** et **15 features pour le multimodal `full`**. Les 6 features de cohérence inter-champs (`cf_*` décrites section 3.6) sont calculées au moment de l'inférence sur le formulaire Streamlit mais ne sont pas utilisées en entraînement (les paires synthétiques image+texte ne contiennent pas de formulaire structuré).

Le classifieur est un XGBoost (300 arbres, profondeur 4, learning rate 0,05) entraîné sur le sous-ensemble domaine habitation (1 170 paires images appariées à des textes synthétiques par cohérence catégorie/label, voir limite méthodologique section 6.1).

## 3.8 Décision et calibration des seuils

Le score final est mappé sur trois décisions opérationnelles :

- **Légitime** : score < seuil_low → auto-validation possible.
- **À expertiser** : seuil_low ≤ score < seuil_high → escalade humaine recommandée.
- **Fraude probable** : score ≥ seuil_high → blocage et expertise approfondie.

Les seuils sont **calibrés automatiquement** sur le validation set selon la stratégie *max recall sous contrainte de précision ≥ 0,85*. Cette contrainte reflète l'arbitrage métier : on accepte de laisser passer quelques fraudes plutôt que de pénaliser massivement les dossiers légitimes.

## 3.9 Explicabilité

Trois mécanismes complémentaires :

- **Occlusion sensitivity** sur l'image : un patch glissant (taille 48 px, stride 24) masque l'image et observe la chute du score. Produit une heatmap des zones critiques. Avantage : model-agnostic, fonctionne avec notre pipeline sklearn sans accès au gradient.
- **SHAP** sur le classifieur image : `LinearExplainer` pour la régression logistique retenue (sinon `TreeExplainer` si XGBoost). Fournit l'importance globale (summary plot et bar plot) et l'explication individuelle d'une prédiction (waterfall plot).
- **Décomposition par axe** dans l'interface Streamlit : affichage simultané du score image, du score multimodal et de la table des features texte saillantes pour permettre à l'expert d'identifier rapidement la source du verdict.

# 4. Résultats et analyses de sensibilité

## 4.1 Métriques de performance

Les chiffres présentés dans cette section sont obtenus après exécution complète du pipeline sur la machine de référence (Colab T4, 16 Go VRAM).

**Modèle image (axe principal) — variante `full` (algorithme retenu : régression logistique) :**

| Split | n | ROC-AUC | F1 (synthétique) | Recall (synthétique) | Precision (synthétique) |
|---|---|---|---|---|---|
| generic_val | 2 989 | 0,763 | 0,724 | 0,75 | 0,70 |
| generic_test | 2 990 | 0,737 | 0,686 | 0,68 | 0,69 |
| domain_test | 819 | 0,414 | 0,273 | 0,19 | 0,49 |

L'effondrement de la performance sur `domain_test` (AUC 0,414) traduit le décalage entre les générateurs vus en entraînement (Stable Diffusion 1.4 dans CIFAKE) et le générateur cible du domaine (SDXL Turbo). Cette chute motive l'introduction de la modalité texte.

**Modèle multimodal (image + texte) — variante `full` (XGBoost sur paires image+texte) :**

| Split | n | ROC-AUC | F1 (synthétique) | Recall | Precision |
|---|---|---|---|---|---|
| domain_val | 176 | 0,964 | 0,937 | 0,93 | 0,95 |
| domain_test | 585 | 0,932 | 0,905 | 0,92 | 0,89 |

**Gain apporté par la modalité texte sur `domain_test` :** ΔROC-AUC = +0,541 points (passage de 0,391 à 0,932), ΔF1 = +0,654 points (passage de 0,251 à 0,905). Cette amplitude valide l'hypothèse de complémentarité des modalités : le texte rattrape la majorité des fraudes que l'image seule ne détecte plus sur SDXL Turbo.

## 4.2 Comparaison des classifieurs

Tableau de comparaison des trois algorithmes sur `generic_val` (variante `full`, 770 features) :

| Algorithme | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0,763 | 0,724 | 0,717 |
| Random Forest | 0,676 | 0,696 | 0,687 |
| XGBoost | 0,699 | 0,694 | 0,685 |

Contrairement à l'intuition initiale (XGBoost favori), c'est la **régression logistique** qui obtient le meilleur ROC-AUC sur ce setup. Cela s'explique par la nature des features : un embedding CLIP de 768 dimensions est déjà construit pour être linéairement séparable (modèle contrastif), et un classifieur linéaire suffit. Les modèles à arbres souffrent de la haute dimensionnalité et de l'absence de feature scalaire dominante.

La régression logistique est donc retenue pour le modèle image final. Pour le modèle multimodal (21 features seulement, dont des scores hétérogènes), XGBoost reste préféré car les interactions non-linéaires entre score image, features texte et features de cohérence apportent un gain mesurable.

## 4.3 Généralisation cross-générateur

Le test cross-générateur est crucial car notre génération synthétique repose principalement sur SDXL Turbo. Si le modèle ne reconnaissait que la signature de SDXL Turbo, il serait inutile contre un fraudeur utilisant Midjourney ou Flux. Nous évaluons les performances par `generator_model` sur le test set :

Comme chaque générateur produit uniquement des images d'une classe (label=1 pour les générateurs IA, label=0 pour les sources réelles), nous reportons un **taux de détection** plutôt qu'une AUC : pour les générateurs IA c'est le `recall_fake` (% d'images correctement classées comme synthétiques au seuil 0,5), pour le groupe `real` c'est le complément du taux de fausses alarmes (`1 − fpr_real`).

| Générateur | Volume test | Score moyen | Taux de détection correcte |
|---|---|---|---|
| Stable Diffusion 1.4 (CIFAKE, vu en train) | 980 | 0,669 | **0,879** (recall fakes) |
| ArtiFact "various" (multi-générateurs) | 510 | 0,444 | 0,296 (recall fakes) |
| Wikimedia + photos réelles (`real`) | 1 759 | 0,406 | 0,684 (1 − fpr) |
| **SDXL Turbo (notre génération domaine)** | **560** | **0,331** | **0,184** (recall fakes) |

**Lecture :** le modèle image-seul atteint **88 % de détection sur SD 1.4** (générateur ancien présent en training) mais s'effondre à **18 % sur SDXL Turbo**, soit un écart de 0,70 point. Cette chute infirme catégoriquement une généralisation simple par signature pour les générateurs récents. Elle confirme deux points : (1) la signature visuelle spécifique de SDXL Turbo (surface lisse, palette saturée) est très différente de SD 1.4 (artefacts d'aliasing typiques des premiers modèles de diffusion) ; (2) sans signal complémentaire, un détecteur de génération anciennes ne protège pas contre la fraude moderne. La modalité texte introduite dans le pipeline multimodal compense cette limite (recall remonté à 0,92 sur domain_test, section 4.1).

## 4.4 Robustesse au bruit

L'image soumise par un assuré peut être compressée par le service de messagerie (qualité JPEG dégradée), prise dans des conditions de faible luminosité, ou contenir du bruit numérique. Nous mesurons la stabilité du score sous deux transformations :

- **Compression JPEG** à qualités 95, 80, 60, 40, 20.
- **Bruit gaussien** d'écart-type σ = 0, 0,02, 0,05, 0,10, 0,15 (en proportion de 255).

**Compression JPEG** : le score moyen oscille entre 0,47 et 0,60 pour les images réelles, et entre 0,30 et 0,47 pour les synthétiques entre les qualités 95 et 40. À qualité 20, les deux distributions convergent autour de 0,47 — le modèle ne discrimine quasiment plus. Cette fragilité est attendue : la compression JPEG supprime les hautes fréquences qui portent une partie de la signature des générateurs.

**Bruit gaussien** : pour les images réelles, le score reste stable autour de 0,5 quel que soit σ. Pour les images synthétiques, le score chute de 0,35 à 0,10 quand σ passe de 0,02 à 0,15. Ce comportement est paradoxal au premier regard mais révèle un mécanisme important : notre détecteur exploite principalement la **lissité anormale** de SDXL Turbo. Ajouter du bruit gaussien camoufle cette lissité et fait basculer la prédiction vers "réelle". Un fraudeur sophistiqué pourrait donc contourner la détection en post-traitant ses fakes avec un bruit léger (σ ≈ 0,05).

[FIGURE : 06_reports/figures/sensibilite_robustesse.png]

Cette double fragilité (compression et bruit) est documentée dans la littérature (Frank et al., 2020 ; Wang et al., 2020) et appelle deux contre-mesures pour une mise en production : (1) augmentation de données pendant le training avec compression JPEG aléatoire et bruit gaussien aléatoire, (2) ensemble avec un détecteur basé fréquences (FFT, DCT) plus robuste aux perturbations bas-niveau.

## 4.5 Ablation des familles de features

Pour chaque famille de features, nous mesurons la perte de ROC-AUC obtenue en mettant ces features à zéro à l'inférence (sans ré-entraîner) :

| Famille retirée | ROC-AUC sur generic_test | Δ vs modèle complet |
|---|---|---|
| (aucune — référence) | 0,736 | 0,000 |
| CLIP (768-d) | 0,500 | **−0,236** |
| EXIF (2 flags) | 0,736 | 0,000 |

**Interprétation :** sans CLIP, l'AUC s'effondre au niveau du hasard (0,500). CLIP est donc la pierre angulaire absolue du modèle image. Le retrait des deux flags EXIF n'a en revanche **aucun effet mesurable**. Cette absence d'apport s'explique par la prévalence des images sans EXIF dans notre dataset : CIFAKE et ArtiFact sont des dérivés de CIFAR-10 (32×32 px sans métadonnées), Wikimedia compresse fréquemment au téléversement, SDXL Turbo ne produit aucun EXIF. La feature `exif_present` finit donc binaire-presque-constante et n'apporte pas d'information discriminante exploitable par le classifieur.

**Limite de cette analyse :** les ablations BLIP-2 et LLaVA n'ont pas été conduites séparément dans ce notebook car le modèle évalué est `image_lite` qui n'inclut pas ces features. Sur la variante `full` du modèle image, BLIP-2 et LLaVA contribuent peu car leurs scores ne varient que sur le sous-ensemble domain_* (1 170 images sur 31 062), insuffisant pour faire émerger un signal robuste à l'échelle du training generic. C'est uniquement dans le **modèle multimodal**, entraîné sur les paires domaine, que les features LMM jouent pleinement leur rôle.

## 4.6 Courbe d'apprentissage

L'ajout massif de données améliorerait-il les performances ? Nous ré-entraînons le modèle sur 5 %, 10 %, 20 %, 50 %, 75 % et 100 % du train set :

| Fraction | n_train | ROC-AUC test |
|---|---|---|
| 5 % | 1 195 | 0,672 |
| 10 % | 2 391 | 0,699 |
| 20 % | 4 782 | 0,697 |
| 50 % | 11 956 | 0,700 |
| 75 % | 17 934 | 0,698 |
| 100 % | 23 913 | 0,694 |

[FIGURE : 06_reports/figures/sensibilite_learning_curve.png]

La courbe atteint un plateau dès **2 400 images** d'entraînement. Au-delà, le gain par image supplémentaire est nul (variation < 0,005 sur l'AUC) et l'AUC redescend même légèrement à 100 % du train set, suggérant un soupçon de sur-ajustement aux conditions spécifiques de CIFAKE/ArtiFact. L'effort d'acquisition n'est rentable que si les nouvelles images couvrent des **distributions absentes** (nouveaux générateurs, angles smartphone, conditions de luminosité particulières) — ajouter plus du même type d'images ne déplace plus le modèle. La piste prioritaire pour améliorer la performance est donc un changement d'**architecture** (fine-tuning de CLIP, classifieur non-linéaire profond, features fréquentielles), pas un changement de volume.

## 4.7 Calibration des seuils

Sur le validation set du modèle multimodal `lite` (n = 176 paires domain_val), la calibration automatique produit les seuils suivants (stratégie max recall sous contrainte precision ≥ 0,85) :

- **seuil_low = 0,051** : auto-validation des dossiers légitimes (95 % de pureté côté légitime).
- **seuil_mid = 0,206** : seuil de bascule "à expertiser → fraude probable" (precision 0,853, recall 0,967, F1 0,906).
- **seuil_high = 0,256** : alerte fraude (85 % de pureté côté fraude).

Pour la variante `full` du multimodal, les seuils calibrés sont décalés vers le bas (mid = 0,114) car les features judge_* du LLM-as-judge produisent des scores plus tranchés. La performance opérationnelle est très proche : precision 0,853, recall 0,967, F1 0,906 également.

Le diagramme de distribution des scores par classe (figure ci-dessous, illustration sur le modèle image lite calibré sur generic_val) montre la séparation atteinte et la position des seuils retenus.

[FIGURE : 06_reports/figures/calibration_distribution_scores.png]
[FIGURE : 06_reports/figures/calibration_roc_pr.png]

## 4.8 Interprétation métier des prédictions

Les analyses SHAP (figure ci-dessous) montrent que les top features sont, dans l'ordre décroissant d'importance moyenne |SHAP| : `clip_417` (0,70), `clip_313` (0,69), `clip_625` (0,60), `clip_393` (0,59), `clip_84` (0,58). Aucune dimension CLIP ne domine isolément ; chaque feature pèse moins de 1 point en moyenne. Surtout, le bar plot révèle que la **somme de l'importance des 756 autres dimensions CLIP atteint 103,82**, soit deux ordres de grandeur au-dessus des features individuelles. Cette répartition est typique d'un modèle linéaire sur représentation dense : la décision est portée par le motif global de l'embedding plutôt que par une signature spécifique localisée. Les flags EXIF n'apparaissent pas dans le top 20, cohérent avec l'ablation (section 4.5) qui montre que leur contribution est nulle.

[FIGURE : 06_reports/figures/shap_summary_top20.png]
[FIGURE : 06_reports/figures/shap_bar_importance.png]

L'analyse SHAP *waterfall* sur un cas individuel à proba 0,999 (vraie fraude correctement détectée) montre que le score logit f(x) = 7,46 résulte d'une superposition de petites contributions : Feature 281 (+3,42), Feature 79 (−3,22), Feature 477 (+2,50), Feature 393 (−2,01), et 759 autres features cumulant +5,58. Aucune dimension à elle seule n'explique la décision, ce qui complique la traduction en termes métier mais reste cohérent avec un détecteur exploitant la statistique globale de l'image.

[FIGURE : 06_reports/figures/shap_waterfall_cas_fraude.png]

L'occlusion sensitivity (heatmap d'importance par patch glissant) révèle que le modèle s'appuie sur des zones étendues plutôt que sur des artefacts ponctuels : **régions centrales de l'image** où sont concentrés les objets et la texture des matériaux (sols, murs, mobilier), et **zones de transition** entre objets où les générateurs produisent souvent des bords flous ou incohérents. Sur les exemples illustrés (figure ci-dessous), l'image réelle d'architecture vernaculaire reçoit une heatmap diffuse mais centrée sur les toitures et la végétation, tandis que l'image synthétique de cuisine reçoit une heatmap fortement focalisée sur les surfaces réfléchissantes du sol stratifié, qui présentent une régularité atypique caractéristique de SDXL Turbo.

[FIGURE : 06_reports/figures/occlusion_heatmap_exemples.png]

Il est honnête de noter que les deux exemples illustrés correspondent à des **erreurs du modèle image-seul** : l'image réelle est classée à tort comme synthétique (score 0,78) et l'image synthétique est classée à tort comme réelle (score 0,30). Cette observation visuelle confirme la limite de l'image-seul sur SDXL Turbo et l'apport décisif du multimodal — qui, lui, classifie correctement ces deux cas grâce au signal textuel.

# 5. Framework technique

## 5.1 Stack utilisée

| Composant | Choix | Rôle |
|---|---|---|
| Langage | Python 3.10+ | Référence du domaine ML |
| Encodeur visuel | `open_clip` ViT-L/14 OpenAI | Embeddings image 768-d |
| LMM image | `transformers` BLIP-2, LLaVA-1.5-7B | Captions, scores VQA structurés |
| LLM texte | Mistral-7B-Instruct-v0.3 (4-bit) | Génération synthétique + LLM-as-judge |
| Génération SDXL | `diffusers` SDXL Turbo | Production des fakes habitation |
| Classifieur image | scikit-learn LogReg (L2) | Apprentissage supervisé léger sur embeddings CLIP |
| Classifieur multimodal | XGBoost 2.0 | Combinaison non-linéaire score image + features texte |
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

Nos images "réelles" sont des photographies authentiques de scènes diverses, mais ne sont pas issues de dossiers de sinistre réels validés par expert. Cette distinction est essentielle : le modèle apprend à reconnaître la signature des générateurs d'IA, pas l'authenticité d'un sinistre déclaré. La généralisation à de "vraies" photos d'assurés (smartphone, conditions variées, EXIF authentique) reste à valider, le sous-ensemble `personal/` n'ayant pas pu être collecté à un volume représentatif dans le temps imparti.

**Limite 2 — Couverture des générateurs et chute sur SDXL Turbo**

Le marché évolue extrêmement vite (~ 6 mois entre deux générateurs majeurs). Notre analyse cross-générateur (section 4.3) **quantifie** précisément cette limite : le modèle image-seul atteint 88 % de recall sur Stable Diffusion 1.4 (présent en training via CIFAKE) mais s'effondre à **18 % sur SDXL Turbo**. Un fraudeur utilisant Sora, Flux 1, ou un futur modèle aura une signature potentiellement très différente. La maintenance du modèle exige un cycle de réentraînement régulier (idéalement trimestriel) sur les nouveaux générateurs majeurs.

**Limite 3 — Saturation rapide de l'apprentissage**

La courbe d'apprentissage (section 4.6) montre un plateau à partir de 2 400 images d'entraînement. Au-delà, l'AUC reste bloquée autour de 0,70. Cette saturation rapide indique que l'amélioration ne viendra pas de plus de données du même type, mais d'un changement d'**architecture** : fine-tuning de CLIP sur le domaine, classifieur non-linéaire profond, ou ajout de features de fréquence (FFT, DCT) sensibles aux artefacts spécifiques des modèles de diffusion modernes.

**Limite 4 — Robustesse aux manipulations adversariales**

Notre analyse de robustesse (section 4.4) **mesure** une double fragilité du modèle image : (a) sous compression JPEG agressive (qualité < 40), les distributions de scores réels et synthétiques convergent ; (b) sous bruit gaussien, les images SDXL Turbo voient leur score chuter, révélant que le détecteur exploite principalement la lissité anormale de SDXL et qu'un bruit léger suffit à camoufler ce signal. Ces vulnérabilités sont documentées dans la littérature (Frank et al., 2020 ; Wang et al., 2020). Un fraudeur sophistiqué pourrait contourner le détecteur en post-traitant ses fakes (recompression itérative, ajout de bruit, capture d'écran). Ces attaques relèvent d'autres approches (forensics passif, analyse de double compression JPEG, ensembles de détecteurs hétérogènes).

**Limite 5 — Couverture catégorielle**

Quatre catégories de sinistre sont couvertes (eau, feu, vitre, vandalisme). Une catégorie rare (foudre, dégât animal, sinistre technologique) sera mal classée. Un système de production exigerait une catégorisation amont et des modèles spécialisés par catégorie.

**Limite 6 — Appariement synthétique image+texte**

Notre dataset multimodal apparie chaque image domaine avec un texte Mistral aléatoire de **même catégorie ET même label**. Cette stratégie simule un fraudeur cohérent qui aligne sa fausse image et son faux texte — c'est un cas favorable. Sur des données réelles non-corrélées (vraie photo + faux texte, ou inversement), le gain multimodal pourrait être plus modeste. Les chiffres rapportés section 4.1 (AUC 0,93) constituent donc un majorant des performances attendues en production.

**Limite 7 — Texte synthétique**

Nos déclarations textuelles d'entraînement sont générées par Mistral avec une consigne explicite d'introduire des incohérences subtiles dans les versions frauduleuses. Les patterns linguistiques peuvent diverger des vraies déclarations d'assurés (vocabulaire, niveau de langue, tournures régionales). La validation sur un corpus annoté de vraies déclarations sinistre/fraude serait nécessaire avant mise en production.

**Limite 8 — EXIF non discriminant**

L'ablation des features (section 4.5) montre que les flags EXIF (présence, GPS) n'apportent **aucun gain mesurable**. Cela s'explique par la prévalence d'images sans EXIF dans notre dataset (CIFAKE/ArtiFact dérivés de CIFAR-10, Wikimedia compressé au téléversement, SDXL ne produisant aucun EXIF). En production, l'EXIF dynamique (date de prise, modèle d'appareil) en cohérence avec les champs déclaratifs serait probablement plus utile que la simple présence/absence.

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
