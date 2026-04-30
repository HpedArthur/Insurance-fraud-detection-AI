# Guide Git — Initialisation et premier push

Ce guide t'amène de zéro à un repo GitHub fonctionnel partagé avec les 2 autres membres de l'équipe. Compte 15 minutes en tout.

## Pré-requis

- Git installé sur ta machine ([git-scm.com](https://git-scm.com/download/win))
- Un compte GitHub
- Un Personal Access Token (PAT) GitHub avec scope `repo` — voir étape 4 si tu n'en as pas

## Étape 1 — Nettoyer l'ébauche `.git` existante

Un dossier `.git` partiellement créé existe déjà dans le projet (lié à une tentative côté serveur). Il faut le supprimer avant de réinitialiser proprement.

Ouvre **PowerShell** dans le dossier du projet :

```powershell
cd "C:\Users\arthu\Documents\Data Mining Fraude à l'assurance\Data Mining Fraude à l'assurance"

# Supprimer l'ébauche .git
Remove-Item -Recurse -Force .git
```

## Étape 2 — Initialiser le repo local

```powershell
git init -b main
git config user.email "amospidiohoungbedji.aph@gmail.com"
git config user.name "HPA"

# Vérifier que le .gitignore exclut bien les gros fichiers
git status
```

Tu dois voir listés : `00_PLAN_PROJET.md`, `00_PLAN_PROJET.pdf`, `README.md`, `requirements.txt`, `08_docs/...`, etc. **Tu ne dois pas voir** de fichiers dans `01_data/` (sauf les `.gitkeep`).

## Étape 3 — Premier commit

```powershell
git add .
git commit -m "Initialisation du projet : plan, structure repo, stratégie dataset"
```

Vérifie :

```powershell
git log --oneline
```

## Étape 4 — Créer un Personal Access Token GitHub (si tu n'en as pas)

1. Aller sur https://github.com/settings/tokens
2. Cliquer **Generate new token (classic)**
3. Note : `Projet ISFA Data Mining`
4. Expiration : 90 jours
5. Scope : cocher uniquement `repo`
6. Cliquer **Generate token**, **copier la valeur** (tu ne la reverras plus)

## Étape 5 — Créer le repo distant sur GitHub

Deux options.

### Option A — Via l'interface web (plus simple)

1. Aller sur https://github.com/new
2. **Repository name** : `data-mining-fraude-assurance` (ou ce que vous voulez)
3. **Description** : `Détection de fraude à l'assurance habitation par LMM — Projet ISFA 2025-2026`
4. **Visibility** : **Private** (recommandé pour un projet pédagogique non finalisé)
5. **NE PAS** cocher "Add a README" / "Add .gitignore" — on a déjà tout en local
6. Cliquer **Create repository**

GitHub t'affiche l'URL du repo, du type `https://github.com/<ton-user>/data-mining-fraude-assurance.git`.

### Option B — Via GitHub CLI (si installé)

```powershell
gh repo create data-mining-fraude-assurance --private --source=. --remote=origin --description "Détection de fraude à l'assurance habitation par LMM — Projet ISFA 2025-2026"
```

## Étape 6 — Lier le local au distant et pousser

Si tu as utilisé l'**Option A** :

```powershell
# Remplace <ton-user> par ton username GitHub
git remote add origin https://github.com/<ton-user>/data-mining-fraude-assurance.git
git push -u origin main
```

Au moment du push, Git te demande tes identifiants :
- **Username** : ton username GitHub
- **Password** : **colle ton Personal Access Token** (pas ton mot de passe GitHub habituel)

Si tu utilises Git Credential Manager (installé par défaut avec Git for Windows récent), une fenêtre OAuth s'ouvrira directement.

## Étape 7 — Inviter les 2 autres membres

Sur la page du repo : **Settings → Collaborators → Add people**, ajouter les deux usernames GitHub des membres de l'équipe avec le rôle **Write**.

## Étape 8 — Vérification finale

Sur la page du repo, tu dois voir :

- `README.md` rendu en page d'accueil
- L'arborescence complète `01_data/` ... `08_docs/`
- Le PDF `00_PLAN_PROJET.pdf` consultable directement dans GitHub (ouvre-le pour vérifier qu'il n'est pas corrompu)

## Workflow équipe à partir de maintenant

Branche stable = `main`. Chacun travaille sur une branche feature dédiée.

**Convention de nommage de branches** :

```
feat/<initiales>/<sujet>          # nouvelle fonctionnalité
fix/<initiales>/<sujet>           # correction
docs/<initiales>/<sujet>          # documentation
data/<initiales>/<sujet>          # ajout/préparation données
```

Exemples : `feat/hpa/streamlit-app`, `data/hpa/cifake-download`, `docs/hpa/rapport-section-3`.

**Cycle type pour une contribution** :

```powershell
# Récupérer les dernières modifs
git checkout main
git pull

# Créer une branche
git checkout -b feat/hpa/streamlit-app

# ... faire des modifs, commit ...
git add .
git commit -m "feat(app): squelette Streamlit avec upload image"

# Pousser la branche
git push -u origin feat/hpa/streamlit-app
```

Puis sur GitHub : **Compare & pull request** → demander une review à un autre membre → merger dans `main`.

## Commandes utiles

| Action | Commande |
|---|---|
| Voir le statut | `git status` |
| Voir les commits | `git log --oneline --graph --all` |
| Annuler des modifs non commitées | `git restore <fichier>` |
| Annuler le dernier commit (en gardant les modifs) | `git reset --soft HEAD~1` |
| Voir les diff non commités | `git diff` |
| Récupérer les modifs distantes | `git pull` |
| Lister les branches | `git branch -a` |
| Changer de branche | `git checkout <branche>` |

## Si tu bloques

- Erreur d'authentification au push → ton PAT est invalide ou expiré, regénère-en un.
- Erreur "remote already exists" → `git remote remove origin` puis recommence à l'étape 6.
- Erreur "non-fast-forward" au premier push → tu as oublié de ne pas cocher "Add README" sur GitHub. Solution : `git pull origin main --rebase --allow-unrelated-histories` puis retenter.
