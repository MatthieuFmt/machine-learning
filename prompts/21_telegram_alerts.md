# Prompt 21 — Bot Telegram d'alertes

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/20_signal_engine.md`

## Objectif
Construire le bot Telegram qui envoie une alerte formatée à chaque nouveau signal produit par le moteur live. Inclure dans le message un **prompt copier-coller** à fournir à une IA externe pour validation manuelle.

## Definition of Done (testable)
- [ ] `app/live/telegram_bot.py` contient :
  - `TelegramAlerter(token: str, chat_id: str)` (init)
  - `send_signal(signal: Signal) -> bool` (retourne True si envoi OK)
  - `send_error(error_msg: str) -> bool` (pour erreurs critiques au prompt 24)
  - `format_signal(signal: Signal) -> str` (markdown formaté)
- [ ] Utilise `requests` uniquement (pas de dépendance lourde type `python-telegram-bot`).
- [ ] `.env.example` créé à la racine :
  ```
  CAPITAL_EUR=10000
  RISK_PER_TRADE=0.02
  TELEGRAM_BOT_TOKEN=
  TELEGRAM_CHAT_ID=
  DATA_ROOT=data/raw
  MODEL_SNAPSHOT=models/snapshots/latest.pkl
  ```
- [ ] `.env` est dans `.gitignore`.
- [ ] Format du message conforme à la section 7 de `prompts/00_constitution.md`.
- [ ] `scripts/run_send_signals.py` lit `signals/today.json` et envoie un message par signal.
- [ ] `tests/integration/test_telegram_format.py` : ≥ 3 tests (format LONG, format SHORT, format avec event éco proche). Pas d'envoi réel — mock du `requests.post`.
- [ ] Documentation : `docs/telegram_setup.md` avec :
  1. Comment créer un bot via `@BotFather`
  2. Comment récupérer le `chat_id`
  3. Comment remplir `.env`
  4. Comment tester l'envoi
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS commit le `.env` avec les tokens (vérifier `.gitignore`).
- Ne PAS faire d'envoi réel dans les tests.
- Ne PAS dépendre de `python-telegram-bot` (lib lourde, on a juste besoin de POST).
- Ne PAS oublier de gérer les erreurs HTTP (timeout, 429 rate limit, 403 chat blocked).
- Ne PAS ignorer le header `retry_after` d'un 429 (Telegram BAN si on spam après 429).
- Ne PAS quitter brutalement (Ctrl+C) sans flush — installer SIGTERM/SIGINT handler.

## Étapes

### Étape 1 — Dépendance
Ajouter `requests>=2.31` à `requirements.txt` si absent.

### Étape 2 — `TelegramAlerter` (validation token + 429 + graceful shutdown)
```python
import os
import signal
import sys
import time

import requests

from app.core.logging import get_logger
from app.live.signal_engine import Signal

logger = get_logger(__name__)


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str, validate_on_init: bool = True):
        self.token = token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        if validate_on_init:
            self._validate_token()
        # Installer handler pour graceful shutdown
        signal.signal(signal.SIGTERM, self._on_shutdown)
        signal.signal(signal.SIGINT, self._on_shutdown)

    def _validate_token(self) -> None:
        """Appel getMe au démarrage. Raise si HTTP 401 (token invalide)."""
        r = requests.get(f"{self.base}/getMe", timeout=10)
        if r.status_code == 401:
            raise RuntimeError("Telegram token invalide (HTTP 401). Vérifie .env.")
        r.raise_for_status()

    def _on_shutdown(self, signum, frame):
        logger.info("Telegram shutdown initié (SIGTERM/SIGINT). Flush logs.")
        try:
            self.send_error("🛑 Bot arrêté proprement.")
        except Exception:
            pass
        sys.exit(0)

    def send_signal(self, signal_obj: Signal) -> bool:
        return self._post("sendMessage", {
            "chat_id": self.chat_id,
            "text": self.format_signal(signal_obj),
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        })

    def send_error(self, msg: str) -> bool:
        return self._post("sendMessage", {
            "chat_id": self.chat_id,
            "text": f"⚠️ {msg}",
        })

    def _post(self, endpoint: str, payload: dict, max_429_retries: int = 5) -> bool:
        for attempt in range(max_429_retries):
            try:
                r = requests.post(f"{self.base}/{endpoint}", json=payload, timeout=15)
                if r.status_code == 429:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 5))
                    logger.warning(f"Telegram 429 : attente {retry_after}s")
                    time.sleep(retry_after)
                    continue
                return r.status_code == 200
            except requests.RequestException as e:
                logger.error(f"Telegram POST échoué : {e}")
                return False
        return False

    def format_signal(self, s: Signal) -> str:
        capital = os.getenv("CAPITAL_EUR", "10000")
        risk = float(os.getenv("RISK_PER_TRADE", "0.02")) * float(capital)

        emoji = "🟢 LONG" if s.direction == "LONG" else "🔴 SHORT"
        return f"""🎯 *{s.asset}* {s.tf} — {emoji}
─────────────────────────
📅 {s.timestamp_utc}
💰 Capital : {capital} € | Risk : {risk:.0f} €

🎯 Entrée : `{s.entry_price:.5f}`
🛡️ Stop Loss : `{s.stop_loss:.5f}`
🏁 Take Profit : `{s.take_profit:.5f}` (R:R = {s.risk_reward:.1f}:1)
📦 Taille : {s.size_lots} lots ({s.size_units:.0f} unités)

📊 Stratégie : {s.strategy_name}
📈 Régime : {s.regime}
🎲 Confiance modèle : {s.confidence:.1%}

─────────────────────────
🔍 *PROMPT À COPIER POUR VALIDATION IA :*
─────────────────────────
{self._build_validation_prompt(s)}
"""

    def _build_validation_prompt(self, s: Signal) -> str:
        return f"""```
Contexte de marché : {s.asset} {s.tf} à {s.timestamp_utc}
Direction proposée : {s.direction}
Entrée : {s.entry_price}
Stop Loss : {s.stop_loss}
Take Profit : {s.take_profit}
R:R : {s.risk_reward}:1

Stratégie : {s.strategy_name}
Régime : {s.regime}
Confiance modèle : {s.confidence:.1%}

Données complémentaires :
- ATR : {s.meta_data.get('atr', 'N/A')}
- RSI(14) : {s.meta_data.get('rsi', 'N/A')}
- Prochain event éco High Impact : dans {s.meta_data.get('next_high_impact_event_hours', 'N/A')}h

Question : Est-ce que ce trade est à prendre ? Réponds OUI ou NON avec
une justification en 2 phrases maximum.
```"""
```

### Étape 3 — Tests avec mock
```python
def test_format_signal_long(monkeypatch):
    monkeypatch.setenv("CAPITAL_EUR", "10000")
    monkeypatch.setenv("RISK_PER_TRADE", "0.02")
    signal = Signal(asset="US30", direction="LONG", ...)
    alerter = TelegramAlerter(token="x", chat_id="y")
    msg = alerter.format_signal(signal)
    assert "US30" in msg
    assert "LONG" in msg
    assert "Stop Loss" in msg
    assert "PROMPT À COPIER" in msg
```

### Étape 4 — Documentation `docs/telegram_setup.md`
Étapes détaillées en français pour créer le bot.

## Critères go/no-go
- **GO prompt 22** si : tests passent, l'utilisateur a testé l'envoi sur son chat avec succès.
- **NO-GO** : si l'envoi réel échoue, vérifier token, chat_id, droits du bot dans le chat.
