# Plan projet — Détection de fraude à l'assurance habitation par LMM

**Cours** : Intelligence Artificielle appliquée à l'Assurance — ML, Deep Learning & IA Générative (ISFA, 2025-2026)
**Équipe** : 3 membres
**Soutenance** : 15 minutes (10 min présentation + 5 min Q/R) — **cible interne : semaine du 25 mai 2026**
**Statut sujet** : validé

---

## 1. Cadrage affiné

### 1.1 Problématique métier

Les assureurs habitation reçoivent chaque jour des dossiers de sinistre accompagnés de photos (dégâts des eaux, incendie, vandalisme, bris de glace). L'arrivée des modèles génératifs grand public (Stable Diffusion, Midjourney, DALL·E, Flux) abaisse drastiquement le coût de fabrication d'une fausse preuve photographique. Une fraude qui exigeait hier une mise en scène physique se résume aujourd'hui à un prompt textuel.

L'enjeu pour l'assureur est double :
- **Financier** : limiter les indemnisations indûment versées (la fraude représente entre 2,5 % et 10 % du coût des sinistres selon les estimations ALFA).
- **Opérationnel** : ne pas allonger les délais d'instruction des dossiers légitimes ni multiplier les expertises terrain.

### 1.2 Hypothèse de travail

L'**axe principal** du projet est la détection des images générées par IA. La détection sur le texte de déclaration est un **complément** qui enrichit la décision et rend la démo réaliste, mais ne porte pas la valeur ajoutée scientifique du projet.

Le pipeline cible produit deux scores distincts :

1. **Score image (principal)** — Calculé par un classifieur supervisé sur des features extraites de la photo seule : embedding CLIP ViT-L/14, captions et scores VQA de BLIP-2, scores structurés de LLaVA-1.5, métadonnées EXIF. C'est ce score qui est utilisé pour mesurer la qualité scientifique du modèle (ROC-AUC, F1, recall sur la classe « image générée »).

2. **Score multimodal (enrichi)** — Combine le score image avec des features texte (handcrafted + Mistral-7B en mode *LLM-as-judge*) extraites de la déclaration de l'assuré. Permet d'arbitrer les cas frontières et apporte la dimension métier réaliste (cohérence entre ce qui est décrit et ce qui est montré).

Le système prend en entrée :
- **Une photo** prétendant illustrer le sinistre (objet de l'analyse principale),
- **Un texte de déclaration** décrivant le sinistre (date, lieu, circonstances, montant estimé) — optionnel.

et produit en sortie :
- Un **score image** dans [0, 1],
- Un **score multimodal** dans [0, 1] (si déclaration fournie),
- Une **décision** (Légitime / À expertiser / Fraude probable) basée sur le score multimodal s'il est disponible, sinon sur le score image,
- Une **explication interprétable** : heatmap GradCAM sur l'image, mise en évidence des signaux texte saillants, et tableau de contribution par feature (SHAP).

### 1.3 Périmètre

**Dans le périmètre :**
- Classification binaire image : authentique vs générée par IA (axe principal).
- Scoring de cohérence du texte de déclaration (axe complémentaire).
- Fusion des deux signaux en un score de fraude unique.
- Domaine : dégâts intérieurs habitation (dégât des eaux, incendie, bris de vitre, dégradation post-cambriolage).
- Démonstrateur web Streamlit avec formulaire de déclaration complet (upload image + saisie texte structurée).

**Hors périmètre :**
- Détection de fraude « classique » (déclaration mensongère sur une photo authentique, manipulation type Photoshop subtile sans IA générative).
- Vérification d'identité, scoring de l'assuré.
- Analyse multi-image / vidéo / 3D.
- Production réelle (le livrable est un POC, pas un système certifié).

### 1.4 Valeur ajoutée pour le secteur

Le pipeline livré démontre la faisabilité d'un **filtre automatique de premier niveau** : il ne remplace pas un expert mais réduit le volume de dossiers à inspecter manuellement. La métrique cible côté métier est le **rappel sur la classe fraude** (ne rien laisser passer), avec un seuil ajustable selon l'appétit au risque de l'assureur.

---

## 2. Stratégie données

Voir le document détaillé `08_docs/strategie_dataset.md`. Synthèse :

| Source | Type | Volume cible | Licence |
|---|---|---|---|
| CIFAKE (HuggingFace) | Réel + synthétique génériques | 60 000 paires | MIT |
| ArtiFact (subset) | Réel + synthétique multi-générateurs | 20 000 images | CC-BY |
| Pexels / Unsplash / Pixabay (scraping ciblé) | Réel — dégâts habitation | 1 500 images | CC0 |
| Stable Diffusion XL Turbo (génération locale Colab) | Synthétique — dégâts habitation | 1 500 images | générées par nous |
| Total dataset spécifique habitation (val/test) | — | ~3 000 | — |

**Approche en deux étages** :
1. **Pré-entraînement / sélection de features** sur le grand dataset générique (CIFAKE + ArtiFact) pour apprendre la signature des générateurs.
2. **Évaluation et fine-tuning léger** sur le petit dataset spécifique habitation pour mesurer la capacité de transfert au domaine cible.

C'est cette articulation qui répond honnêtement au problème de la rareté des photos de sinistres réelles — point sur lequel l'enseignant interrogera presque certainement.

---

## 3. Architecture technique

### 3.1 Pipeline d'inférence — deux scores

L'architecture est volontairement structurée pour produire **un score image autonome** (le résultat scientifique principal du projet) auquel un **score texte** vient se greffer pour enrichir la décision finale.

```
   ┌───────────────────────────────────────────────────────────────────────┐
   │  ENTRÉE : Photo (obligatoire) + Déclaration texte (optionnelle)        │
   └───────────────────────────────────────────────────────────────────────┘
                          │
       ┌──────────────────┴──────────────────┐
       │                                     │
       ▼                                     ▼
   ──────────────────────────       ──────────────────────────
    BRANCHE IMAGE (principale)        BRANCHE TEXTE (complémentaire)
   ──────────────────────────       ──────────────────────────
       │                                     │
   [I-1] Pré-traitement                  [T-1] Pré-traitement
        (resize, EXIF, phash)                  (cleaning, NER spaCy)
       │                                     │
       ▼                                     ▼
   [I-2] CLIP ViT-L/14                  [T-2] Features handcrafted
        embedding 768-d                       (longueur, TTR, dates, montants,
       │                                       mots emphatiques)
       ▼                                     │
   [I-3a] BLIP-2                              ▼
        caption + 3 scores VQA           [T-3] Mistral-7B LLM-as-judge
       │                                       5 scores : specificity, coherence,
       ▼                                       plausibility, red_flags, overall
   [I-3b] LLaVA-1.5 (4-bit)                  │
        4 scores structurés                   │
       │                                     │
       ▼                                     │
   [I-4] Classifieur image XGBoost #1        │
        ────────────────────────              │
         SCORE IMAGE [0, 1]                  │
        ────────────────────────              │
       │                                     │
       └─────────────────┬───────────────────┘
                         │
                         ▼
   [M-1] Classifieur multimodal XGBoost #2
        (entrée = features image + features texte + score image)
        ──────────────────────────────────
          SCORE MULTIMODAL [0, 1]
        ──────────────────────────────────
                         │
                         ▼
   [D] Décision avec seuils ajustables
       Si déclaration fournie : on utilise SCORE MULTIMODAL.
       Sinon : on utilise SCORE IMAGE.
       Légitime (< 0.3) / À expertiser (0.3 - 0.7) / Fraude probable (> 0.7)
                         │
                         ▼
   [E] Explication
       - GradCAM sur l'image (zones suspectes)
       - SHAP sur le classifieur (contribution de chaque feature)
       - Saillance texte (mots ayant pesé sur le verdict du LLM-judge)
```

**Pourquoi deux classifieurs distincts ?**

- Permet d'évaluer scientifiquement la performance image seule (cible principale du cours).
- Permet de chiffrer le **gain** apporté par le texte (différence de ROC-AUC entre les deux modèles), un point fort à mettre en avant en soutenance.
- Permet de servir une démo dégradée si l'assuré ne fournit pas de texte.

### 3.2 Stack technique

| Composant | Choix retenu | Justification |
|---|---|---|
| Langage | Python 3.10+ | Standard du domaine |
| Vision encoder | CLIP ViT-L/14 (OpenAI, via `open_clip`) | Embeddings de référence, gratuit, rapide |
| LMM image | BLIP-2 OPT-2.7B + LLaVA-1.5-7B en 4-bit (les deux pour comparaison et redondance) | Open-source, exécutables sur Colab T4 |
| LLM texte | Mistral-7B-Instruct-v0.3 en 4-bit | Génération de déclarations synthétiques + LLM-as-judge sur le texte de l'assuré |
| Extraction entités texte | spaCy `fr_core_news_md` | NER (date, montant, lieu) pour features handcrafted |
| Génération synthétique image | `diffusers` + SDXL Turbo | Gratuit, rapide en local |
| Classifieur | scikit-learn (LogReg, RandomForest) + XGBoost | Léger, interprétable, reproductible |
| Déséquilibre | `imbalanced-learn` (SMOTE, class_weight) | Standard pédagogique |
| Explicabilité | SHAP (sur classifieur) + GradCAM (sur CLIP) via `pytorch-grad-cam` | Exigence rapport |
| App web | Streamlit | Maîtrise existante au sein de l'équipe, écosystème mature, layout plus flexible que Gradio pour une UI métier |
| Hébergement | HuggingFace Spaces (free tier, runtime Streamlit) | Démo live pour la soutenance, gratuit, déploiement git-push |
| Versionnage | Git + GitHub | Livrable obligatoire |
| Notebooks | Jupyter / Google Colab | Reproductibilité |

**Note sur le choix Streamlit** : la note de cadrage cite explicitement Streamlit / Gradio / HuggingFace Spaces comme cibles de déploiement valides. Streamlit a été retenu pour deux raisons : (1) maîtrise déjà acquise par un membre de l'équipe, ce qui réduit le risque sur la semaine 3 ; (2) plus de flexibilité de mise en page pour construire une UI proche d'un outil métier de gestionnaire de sinistre (sidebar, multi-pages, widgets de seuil de décision). HF Spaces supporte nativement Streamlit (`runtime: streamlit` dans le `README.md` du Space).

---

## 4. Planning 4 semaines

Aujourd'hui : **jeudi 30 avril 2026**. Soutenance cible : **lundi 25 mai 2026** (à confirmer avec l'enseignant).

### Semaine 1 — du 30 avril au 6 mai : Fondations & données

| Jour | Action | Responsable | Livrable |
|---|---|---|---|
| J+0 (jeu 30/04) | Création repo GitHub privé, push squelette | Membre C | Repo initialisé |
| J+1 | Chacun installe l'environnement, fait tourner un notebook hello-world CLIP | Tous | 3 envs OK |
| J+2 | Téléchargement CIFAKE + ArtiFact (subset), EDA initiale | Membre A | `02_notebooks/01_eda_datasets_generiques.ipynb` |
| J+3 | Scraping Pexels/Unsplash : 4 catégories (eau, feu, vitre, vandalisme) | Membre A | `01_data/raw/pexels/` |
| J+4-5 | Génération SDXL : 1 500 images sur les 4 catégories | Membre B | `01_data/synthetic/sdxl/` |
| J+6 | Construction dataset final (train/val/test) avec stratification | Membre A | `01_data/processed/dataset.parquet` |
| J+6 | Point d'étape équipe (1h) | Tous | Décisions tracées |

**Critère sortie semaine 1** : un fichier `dataset.parquet` chargeable, équilibré en classes par catégorie, avec splits train/val/test stratifiés.

### Semaine 2 — du 7 au 13 mai : Modélisation

| Jour | Action | Responsable | Livrable |
|---|---|---|---|
| J+7 | Extraction embeddings CLIP sur tout le dataset | Membre B | `04_models/clip_embeddings.npy` |
| J+8 | Baseline : LogReg sur embeddings CLIP — ROC-AUC, F1, matrice de confusion | Membre B | `02_notebooks/02_baseline_clip_logreg.ipynb` |
| J+9 | Ajout XGBoost + RandomForest, comparaison | Membre B | `02_notebooks/03_modeles_compares.ipynb` |
| J+10 | Inférence LMM (BLIP-2) sur sous-échantillon : captions + score VQA | Membre B | `02_notebooks/04_features_lmm.ipynb` |
| J+11 | Modèle final : embeddings CLIP + features LMM + EXIF | Membre B | `04_models/final_model.pkl` |
| J+12 | SHAP + GradCAM pour interprétabilité | Membre A | `02_notebooks/05_explicabilite.ipynb` |
| J+13 | Point d'étape, gel des features | Tous | Métriques figées |

**Critère sortie semaine 2** : un modèle `final_model.pkl` avec ROC-AUC > 0.85 sur le split test du dataset générique, et > 0.70 sur le dataset spécifique habitation (cible réaliste).

### Semaine 3 — du 14 au 20 mai : Application & déploiement

| Jour | Action | Responsable | Livrable |
|---|---|---|---|
| J+14 | Wrapper d'inférence : `predict(image, texte) -> (label, score, explication)` | Membre B | `03_src/inference.py` |
| J+15-16 | App Streamlit : upload image + texte, affichage prédiction + heatmap GradCAM, slider de seuil de décision | Membre C | `05_app/app.py` |
| J+17 | Déploiement HuggingFace Spaces (runtime Streamlit), test bout-en-bout | Membre C | URL Space publique |
| J+18 | Analyse de sensibilité : variation seuil, robustesse au bruit, cas adverses | Membre A | `02_notebooks/06_sensibilite.ipynb` |
| J+19 | Génération des figures finales pour le rapport (matrices, courbes ROC, exemples) | Membre A | `06_reports/figures/` |
| J+20 | Point d'étape, gel des résultats | Tous | Tous chiffres figés |

**Critère sortie semaine 3** : démo Gradio publique sur HF Spaces, traitant une image en moins de 10 secondes.

### Semaine 4 — du 21 au 27 mai : Livrables & soutenance

| Jour | Action | Responsable | Livrable |
|---|---|---|---|
| J+21-22 | Rédaction rapport (10-15 pages PDF, plan ISFA imposé) | Membre C (lead) + tous | `06_reports/rapport_final.pdf` |
| J+23-24 | Construction PPT (10-12 slides max, visuel) | Membre A (lead) + tous | `07_presentation/soutenance.pptx` |
| J+25 | Répétition orale 1 (10 min chrono, chacun parle) | Tous | Notes corrections |
| J+26 | Répétition orale 2 + scénario démo live | Tous | Démo sécurisée |
| J+27 | Buffer / corrections / soutenance | Tous | — |

---

## 5. Répartition des rôles

L'enseignant peut interroger chaque membre sur n'importe quelle partie : **chacun doit comprendre l'ensemble**. La répartition ci-dessous est une responsabilité de pilotage, pas un cloisonnement.

| Membre | Pilotage principal | Doit aussi savoir expliquer |
|---|---|---|
| **A — Data & analyse** | Acquisition, nettoyage, EDA, stratification, sensibilité | Architecture du modèle, choix du LMM |
| **B — Modélisation** | Extraction features, entraînement, évaluation, explicabilité | Provenance et biais du dataset |
| **C — MLOps & livrables** | Squelette repo, app Gradio, déploiement HF, rapport, PPT | Justification métier et limites |

---

## 6. Métriques de succès

### 6.1 Performance modèle

**Modèle image (cible principale)** :

| Métrique | Cible générique (CIFAKE+ArtiFact) | Cible domaine (habitation) | Justification |
|---|---|---|---|
| ROC-AUC | > 0.90 | > 0.75 | Synthèse standard, robuste au déséquilibre |
| F1 (classe fraude) | > 0.85 | > 0.70 | Compromis precision/recall |
| Recall (classe fraude) | > 0.85 | > 0.80 | Métrique métier prioritaire |
| Latence inférence (1 image) | < 10 s sur HF Spaces (Streamlit) | — | Convient à un usage opérationnel |

**Modèle multimodal (cible secondaire)** :

| Métrique | Cible | Justification |
|---|---|---|
| Gain ROC-AUC vs image seule | > +3 points | Montre la valeur ajoutée du texte |
| Recall classe fraude | > +2 points vs image seule | Le texte doit principalement améliorer la détection des fraudes que l'image rate |
| Faux positifs (légitimes mal classés) | ≤ niveau image seule | La fusion ne doit pas dégrader |

### 6.2 Livrables (consignes ISFA)

- Rapport PDF 10-15 pages (plan imposé en 7 sections).
- PPT synthétique 10-12 slides.
- Notebook Jupyter exécutable + repo GitHub public ou partagé.
- Application déployée fonctionnelle (Gradio sur HF Spaces).

---

## 7. Risques et mitigations

| # | Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Dataset spécifique habitation trop petit ou biaisé | Élevée | Élevé | Étage 1 sur dataset générique pour pré-train + acceptation explicite de la limite dans le rapport |
| R2 | LLaVA / BLIP-2 trop lourd pour Colab Free | Moyenne | Moyen | Quantization 4-bit, batch size 1, fallback CLIP-seul si nécessaire |
| R3 | HuggingFace Space tombe en panne le jour J | Faible | Élevé | Démo locale `streamlit run 05_app/app.py` en backup + capture vidéo de la démo |
| R4 | Sur-apprentissage sur la signature SDXL (le modèle n'apprend que « SDXL vs photo ») | Élevée | Moyen | Évaluation cross-générateur (Midjourney, Flux) pour mesurer la généralisation |
| R5 | Indisponibilité d'un membre (maladie, charge pro) | Moyenne | Moyen | Documentation systématique en cellules markdown, code commenté, aucune dépendance personne-machine locale critique |
| R6 | Manque de temps pour la soutenance | Moyenne | Élevé | Buffer de 2 jours en S4, répétitions chronométrées en S4 |

---

## 8. Bibliographie de démarrage (à étoffer)

- Bird, J. J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.
- Rahman, M. A., et al. (2023). *ArtiFact: A Large-Scale Dataset with Artificial and Factual Images for Generalizable and Robust Synthetic Image Detection*. ICIP.
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML.
- Liu, H., et al. (2023). *Visual Instruction Tuning* (LLaVA). NeurIPS.
- Li, J., et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training* (BLIP-2). ICML.
- ALFA — Agence pour la Lutte contre la Fraude à l'Assurance, rapports annuels.
- ACPR — recommandations sur l'usage de l'IA dans le secteur financier.

---

## 9. Point d'attention soutenance

Sur 20 points, **8 points sont alloués à la présentation orale** (dont la qualité des réponses aux questions). Conséquences pratiques :

- Construire le PPT autour d'**une seule histoire** : « voici le problème métier, voici notre approche, voici la démo, voici les limites assumées ».
- La **démo live** est explicitement encouragée par la note de cadrage. Elle doit être courte (90 secondes max) et tester un cas légitime + un cas fraude évident.
- Anticiper les questions probables et préparer des réponses chiffrées :
  - Pourquoi CLIP plutôt que ResNet/EfficientNet ?
  - Pourquoi ce LMM open-source plutôt qu'un autre ?
  - Comment vous généralisez à un générateur que vous n'avez jamais vu ?
  - Quel taux de faux positifs vous accepteriez en production ?
  - Coût d'inférence à grande échelle (1M de dossiers / mois) ?
  - Problèmes d'éthique / RGPD sur les photos d'assurés ?
