#!/bin/bash
# Hook de démarrage — installe les dépendances pour que tests/linters tournent
# dans les sessions Claude Code on the web. Idempotent, non interactif, résilient.
set -uo pipefail

# Ne tourner qu'en environnement distant (Claude Code on the web).
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

PIP="python -m pip install --quiet --disable-pip-version-check --ignore-installed"

# Dépendances projet. --ignore-installed contourne les conflits avec les paquets
# système Debian (ex. pyparsing). pandas-ta (beta, Python 3.12+) n'est PAS sur
# PyPI en Python 3.11 → on l'exclut puis on tente en best-effort sans bloquer.
grep -vi '^pandas-ta' requirements.txt > /tmp/_req_core.txt 2>/dev/null || cp requirements.txt /tmp/_req_core.txt
$PIP -r /tmp/_req_core.txt || echo "[session-start] WARN: certaines deps cœur ont échoué."

# Outils de dev (pytest non listé dans requirements.txt).
$PIP pytest ruff mypy || echo "[session-start] WARN: install outils dev partielle."

# pandas-ta best-effort (utilisé seulement par app/features/regime.py).
$PIP pandas-ta >/dev/null 2>&1 || echo "[session-start] INFO: pandas-ta indisponible (tests regime ignorés)."

# Rendre le package app importable sans installation.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
fi

echo "[session-start] terminé."
exit 0
