# Prompt 22 — Scheduler local pour tests

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/21_telegram_alerts.md`

## Objectif
Construire un scheduler local qui exécute le pipeline `update_data → compute_signals → send_telegram` à intervalle régulier, pour valider le bout-en-bout avant le déploiement VPS (prompt 23).

## Definition of Done (testable)
- [ ] `app/live/runner.py` contient `run_once()` qui :
  1. (Optionnel, désactivable) appelle un hook `data_updater.update()` pour rafraîchir les CSV
  2. Appelle `compute_today_signals()`
  3. Pour chaque signal, appelle `TelegramAlerter.send_signal()`
  4. Logge dans `logs/runner.jsonl` (1 entrée JSON par run) : timestamp, nb signaux, erreurs
- [ ] `scripts/run_scheduler_local.py` utilise `APScheduler` pour planifier `run_once()` selon le TF retenu :
  - D1 : 1×/jour à 22:00 UTC (après cloture US)
  - H4 : 6×/jour à H+5 minutes (00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC)
  - H1 : 24×/jour à HH:05 UTC
- [ ] `scripts/run_once.py --dry-run` pour test ponctuel manuel.
- [ ] Le scheduler tourne en mode bloquant. Stop : Ctrl+C ou kill du processus.
- [ ] Documentation : section "Test local" dans `docs/telegram_setup.md`.
- [ ] `tests/integration/test_runner.py` : test `run_once()` avec mocks (data, model, telegram). Vérifie qu'un signal mock déclenche un envoi mock.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS planifier moins fréquemment que TF (ex: H4 = 6×/jour, pas 1×).
- Ne PAS oublier la déduplication : un même signal sur la même barre ne doit pas être renvoyé. Maintenir un cache `logs/sent_signals.jsonl`.
- Ne PAS faire d'envoi réel dans les tests.
- Ne PAS coupler le scheduler à un service externe (Celery, Redis, etc.) — APScheduler local suffit.

## Étapes

### Étape 1 — Dépendance
Ajouter `apscheduler>=3.10` à `requirements.txt`.

### Étape 2 — `run_once`
```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import get_logger
from app.live.signal_engine import compute_today_signals
from app.live.telegram_bot import TelegramAlerter

logger = get_logger(__name__)


def run_once(dry_run: bool = False) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    out: dict = {"started_at": started_at, "signals": [], "errors": []}

    try:
        signals = compute_today_signals(
            data_root=Path(os.getenv("DATA_ROOT", "data/raw")),
            model_snapshot=Path(os.getenv("MODEL_SNAPSHOT", "models/snapshots/latest.pkl")),
        )
    except Exception as e:
        out["errors"].append(f"signal_engine: {e}")
        signals = []

    sent_log = Path("logs/sent_signals.jsonl")
    sent_keys = _load_sent_keys(sent_log)
    new_signals = [s for s in signals if _signal_key(s) not in sent_keys]
    out["n_signals"] = len(signals)
    out["n_new"] = len(new_signals)

    if not dry_run and new_signals:
        alerter = TelegramAlerter(
            token=os.environ["TELEGRAM_BOT_TOKEN"],
            chat_id=os.environ["TELEGRAM_CHAT_ID"],
        )
        for sig in new_signals:
            ok = alerter.send_signal(sig)
            if ok:
                _append_sent(sent_log, sig)
                out["signals"].append({"asset": sig.asset, "direction": sig.direction})
            else:
                out["errors"].append(f"send failed: {sig.asset}")

    Path("logs").mkdir(exist_ok=True)
    with open("logs/runner.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    return out


def _signal_key(s) -> str:
    return f"{s.asset}|{s.tf}|{s.timestamp_utc}|{s.direction}"


def _load_sent_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}


def _append_sent(path: Path, s) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": _signal_key(s), "at": datetime.now(timezone.utc).isoformat()}) + "\n")
```

### Étape 3 — Scheduler APScheduler
```python
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config.timeframe import PRIMARY_TF
from app.live.runner import run_once


def schedule_for_tf(tf: str):
    sched = BlockingScheduler(timezone="UTC")
    if tf == "D1":
        sched.add_job(run_once, CronTrigger(hour=22, minute=0))
    elif tf == "H4":
        for h in [0, 4, 8, 12, 16, 20]:
            sched.add_job(run_once, CronTrigger(hour=h, minute=5))
    elif tf == "H1":
        sched.add_job(run_once, CronTrigger(minute=5))
    return sched


if __name__ == "__main__":
    sched = schedule_for_tf(PRIMARY_TF)
    print(f"Scheduler démarré pour TF={PRIMARY_TF}. Ctrl+C pour stopper.")
    sched.start()
```

### Étape 4 — Tests
Mocker `compute_today_signals` pour retourner 2 signaux. Mocker `TelegramAlerter.send_signal` pour retourner True. Vérifier que `run_once()` retourne `{"n_signals": 2, "n_new": 2, "signals": [...], "errors": []}` et que les logs sont écrits.

## Critères go/no-go
- **GO prompt 23** si : l'utilisateur a fait tourner le scheduler localement pendant au moins 24h (D1) ou 8h (H4/H1) sans crash.
- **NO-GO** : revenir à ce prompt si déduplication ou logs cassés.
