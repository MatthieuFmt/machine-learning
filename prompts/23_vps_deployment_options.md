# Prompt 23 — Options de déploiement (VPS / Docker / GitHub Actions)

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/22_scheduler_local.md`

## Objectif
Présenter à l'utilisateur 3 options de déploiement détaillées avec leurs avantages/inconvénients, et produire les fichiers de configuration pour l'option choisie. **Pas de provisioning automatique** — l'utilisateur exécute lui-même les étapes serveur.

## Definition of Done (testable)
- [ ] `docs/deployment_options.md` présente 3 options :
  1. **VPS classique (Ubuntu 22.04 + systemd + cron)**
  2. **Docker (Dockerfile + docker-compose.yml)**
  3. **GitHub Actions (cron workflow)**
- [ ] Pour chaque option : prérequis, coût mensuel estimé, étapes de déploiement détaillées, avantages, inconvénients.
- [ ] **Question à l'utilisateur** : quelle option il choisit. Attendre réponse explicite.
- [ ] Selon l'option choisie, créer les fichiers :
  - **Option 1 (VPS)** :
    - `deploy/systemd/trading-bot.service` : unit file systemd
    - `deploy/cron.example` : exemple de cron
    - `deploy/install_vps.sh` : script d'installation idempotent (Python 3.12, venv, dépendances, clone, .env)
    - `docs/vps_setup.md` : guide pas-à-pas
  - **Option 2 (Docker)** :
    - `Dockerfile` (multi-stage Python 3.12-slim)
    - `docker-compose.yml`
    - `.dockerignore`
    - `docs/docker_setup.md`
  - **Option 3 (GitHub Actions)** :
    - `.github/workflows/run_signals.yml` (cron trigger selon TF)
    - Documentation des secrets GitHub à configurer (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.)
    - `docs/github_actions_setup.md`
- [ ] `JOURNAL.md` mis à jour avec l'option retenue.

## NE PAS FAIRE
- Ne PAS exécuter `apt install`, `docker build`, etc. — uniquement produire les fichiers.
- Ne PAS commit le `.env`.
- Ne PAS choisir l'option à la place de l'utilisateur. Demander.
- Ne PAS produire les 3 implémentations — seulement celle choisie.

## Étapes

### Étape 1 — Présenter les 3 options

**Option 1 — VPS classique**

| Aspect | Détail |
|---|---|
| Coût | 4-10 €/mois (Hetzner CX11, OVH VPS Starter) |
| Prérequis | Compte VPS, accès SSH, clé SSH configurée |
| Avantages | Contrôle total, persistance des modèles, logs locaux |
| Inconvénients | Setup manuel (Python, venv, cron), maintenance OS |

**Option 2 — Docker**

| Aspect | Détail |
|---|---|
| Coût | Idem VPS si auto-hébergé, ou gratuit en local |
| Prérequis | Docker installé sur le serveur cible |
| Avantages | Environnement reproductible, isolation, déploiement facile |
| Inconvénients | Overhead léger, nécessite Docker sur le serveur |

**Option 3 — GitHub Actions**

| Aspect | Détail |
|---|---|
| Coût | Gratuit (2000 min/mois public, 3000 min/mois free privé) |
| Prérequis | Repo GitHub, secrets configurés |
| Avantages | Pas de serveur à maintenir, secrets gérés, logs intégrés |
| Inconvénients | Limites de minutes, état non persistant (recharger snapshot chaque run), pas de TF M1/M5 (granularité cron limitée) |

### Étape 2 — Recommandation
- Si TF = D1 ou H4 : **GitHub Actions** (gratuit, simple)
- Si TF = H1 : **VPS** ou **Docker** (mieux pour fréquence élevée)
- Si volonté de contrôle total : **VPS**

### Étape 3 — Demander à l'utilisateur

```
Trois options de déploiement disponibles :
1. VPS classique (Hetzner/OVH, ~5€/mois)
2. Docker (auto-hébergé ou cloud)
3. GitHub Actions (gratuit, recommandé pour D1/H4)

Quelle option choisis-tu ?
```

### Étape 4 — Produire les fichiers de l'option choisie

#### Option 1 — VPS

`deploy/systemd/trading-bot.service` :
```ini
[Unit]
Description=Trading Bot ML
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/opt/trading-bot
EnvironmentFile=/opt/trading-bot/.env
ExecStart=/opt/trading-bot/venv/bin/python scripts/run_scheduler_local.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

`deploy/install_vps.sh` :
```bash
#!/usr/bin/env bash
set -euo pipefail

apt update && apt install -y python3.12 python3.12-venv git
useradd -m -s /bin/bash botuser || true

cd /opt
git clone <REPO_URL> trading-bot
cd trading-bot

sudo -u botuser python3.12 -m venv venv
sudo -u botuser ./venv/bin/pip install -r requirements.txt

cp .env.example .env
echo "→ Édite /opt/trading-bot/.env avec ton token Telegram et chat_id."
echo "→ Puis : systemctl enable --now trading-bot"
```

#### Option 2 — Docker

`Dockerfile` :
```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/run_scheduler_local.py"]
```

`docker-compose.yml` :
```yaml
services:
  bot:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
```

#### Option 3 — GitHub Actions

`.github/workflows/run_signals.yml` :
```yaml
name: Trading Signals
on:
  schedule:
    - cron: '0 22 * * *'  # D1 : 22:00 UTC (à adapter selon TF)
  workflow_dispatch:

jobs:
  signals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - name: Run once
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          CAPITAL_EUR: ${{ vars.CAPITAL_EUR }}
          RISK_PER_TRADE: ${{ vars.RISK_PER_TRADE }}
        run: python scripts/run_once.py
```

> Note GitHub Actions : la persistance des modèles entre runs nécessite un release artifact ou un commit auto. Documenter ce point.

## Critères go/no-go
- **GO prompt 24** si : l'utilisateur a déployé avec succès (manuellement) ET reçoit des signaux en conditions réelles.
- **NO-GO** : revenir à ce prompt si erreur de déploiement (logs systemd / docker logs / Actions logs).
