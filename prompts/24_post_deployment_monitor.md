# Prompt 24 — Monitoring post-déploiement

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/23_vps_deployment_options.md`

## Objectif
Mettre en place le monitoring post-déploiement : logs structurés, alertes Telegram en cas d'erreur, KPI hebdomadaire (Sharpe live vs backtest, drift detection).

## Definition of Done (testable)
- [ ] **Logs structurés** : `app/core/logging.py` produit du JSON Lines dans `logs/app.jsonl` (rotation hebdomadaire). Champs : timestamp, level, module, message, context.
- [ ] **Alerte erreur Telegram** : tout `ERROR` level déclenche `TelegramAlerter.send_error()`. Throttle : max 1 alerte par 10 minutes pour le même message.
- [ ] **KPI hebdomadaire** : `scripts/run_weekly_report.py` produit un rapport markdown :
  - Trades de la semaine (entry, SL, TP, PnL si feedback fourni)
  - Sharpe rolling 30j
  - Comparaison Sharpe live vs Sharpe backtest (drift indicator)
  - Top 3 erreurs de la semaine
  - Envoi via Telegram
- [ ] **Drift detection** : `app/analysis/drift.py` calcule, à chaque retrain (prompt 19 H18), si Sharpe rolling **30 trades** live < Sharpe backtest × 0.5 → alerte "drift suspecté". **Minimum 30 trades requis** avant le calcul (sinon estimation trop bruitée).
- [ ] **Auto-rollback** : si Sharpe live < 0 sur **30 trades consécutifs** → le moteur charge automatiquement le snapshot précédent (cf. prompt 19 `_check_and_rollback`) et envoie alerte Telegram.
- [ ] Le rapport hebdomadaire inclut **le ratio Sharpe live / Sharpe backtest par sleeve** (drift granulaire, pas seulement portfolio).
- [ ] **Health check** : `scripts/run_health_check.py` vérifie :
  - Dernier signal envoyé < 48h pour D1 (< 8h pour H4)
  - Modèle snapshot existe et n'a pas plus de 7 mois
  - Données CSV à jour (dernière barre < 48h)
  - **Ratio ERROR/INFO logs < 5 %** sur les dernières 24h (sinon → alerte "instabilité système")
  - **Nombre de WARNING > 50/jour** → alerte "anomalie volumétrie logs"
  - Si un check échoue → alerte Telegram
- [ ] **Feedback manuel des trades** : `scripts/log_trade_outcome.py --signal-id <id> --outcome <hit_tp|hit_sl|timeout> --pnl-eur <X>` permet d'enregistrer manuellement le résultat. Stocké dans `logs/trade_outcomes.jsonl`.
- [ ] Documentation : `docs/monitoring.md`.
- [ ] `JOURNAL.md` mis à jour avec la routine de monitoring (1×/semaine vérifier le rapport, 1×/mois revoir le drift).

## NE PAS FAIRE
- Ne PAS calculer le Sharpe live à partir de 5 trades — minimum 20 trades requis pour une estimation décente. Sinon, afficher "N trades, Sharpe non significatif".
- Ne PAS spammer Telegram en cas de boucle d'erreur — throttling obligatoire.
- Ne PAS modifier le code du modèle en réaction au drift (= data snooping). Drift = signal pour re-évaluer hors prod.
- Ne PAS oublier que le feedback des trades est MANUEL (l'utilisateur copie le résultat).

## Étapes

### Étape 1 — Logging JSON Lines
```python
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }
        if hasattr(record, "context"):
            log["context"] = record.context
        return json.dumps(log)
```

### Étape 2 — Alert Handler
```python
class TelegramErrorHandler(logging.Handler):
    def __init__(self, alerter, throttle_seconds: int = 600):
        super().__init__(level=logging.ERROR)
        self.alerter = alerter
        self.throttle = throttle_seconds
        self.last_sent: dict[str, float] = {}

    def emit(self, record):
        msg = record.getMessage()
        now = time.time()
        if now - self.last_sent.get(msg, 0) < self.throttle:
            return
        self.alerter.send_error(f"[{record.levelname}] {record.module}: {msg}")
        self.last_sent[msg] = now
```

### Étape 3 — Health check (avec count des erreurs)
```python
import json
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path


def count_log_levels_24h(log_path: Path = Path("logs/app.jsonl")) -> Counter:
    if not log_path.exists():
        return Counter()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    counts: Counter = Counter()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            ts = datetime.fromisoformat(entry["ts"])
            if ts >= cutoff:
                counts[entry["level"]] += 1
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return counts


def run_health_check() -> list[str]:
    issues: list[str] = []
    last_signal_age = get_last_signal_age_hours()
    if PRIMARY_TF == "D1" and last_signal_age > 48:
        issues.append(f"Dernier signal il y a {last_signal_age}h")

    model_age = get_model_age_days()
    if model_age > 210:
        issues.append(f"Modèle pas réentraîné depuis {model_age} jours")

    for asset in PORTFOLIO_ASSETS:
        df = load_asset(asset, PRIMARY_TF)
        data_age = (datetime.now(timezone.utc) - df.index[-1]).total_seconds() / 3600
        if data_age > 48:
            issues.append(f"Données {asset} : dernière barre il y a {data_age:.0f}h")

    log_counts = count_log_levels_24h()
    n_error = log_counts.get("ERROR", 0)
    n_info = log_counts.get("INFO", 1)
    if n_info > 0 and (n_error / n_info) > 0.05:
        issues.append(f"Ratio ERROR/INFO {n_error}/{n_info} = {n_error/n_info:.1%} > 5%")
    if log_counts.get("WARNING", 0) > 50:
        issues.append(f"{log_counts['WARNING']} WARNING en 24h (> 50, anormal)")

    return issues
```

### Étape 4 — Drift detection + auto-rollback
```python
def detect_drift(live_returns: pd.Series, backtest_sharpe: float, min_trades: int = 30) -> bool:
    """Minimum 30 trades requis pour une estimation décente."""
    if len(live_returns) < min_trades:
        return False
    live_sharpe = sharpe_ratio(live_returns)
    return live_sharpe < 0.5 * backtest_sharpe


def maybe_auto_rollback(live_returns: pd.Series, min_trades: int = 30) -> bool:
    """Si Sharpe < 0 sur 30 trades consécutifs récents → rollback."""
    if len(live_returns) < min_trades:
        return False
    recent = live_returns.tail(min_trades)
    if sharpe_ratio(recent) < 0:
        from app.pipelines.walk_forward import WalkForwardEngine
        engine = WalkForwardEngine.load_current()
        engine.force_rollback(reason="Sharpe live < 0 sur 30 trades")
        return True
    return False
```

### Étape 5 — Rapport hebdomadaire
Template markdown envoyé via Telegram (court — Telegram limite à 4096 chars).

## Critères go/no-go
- **GO** (fin de la roadmap) si : monitoring opérationnel, premier rapport hebdo envoyé avec succès, premier health check OK.
- **NO-GO** : revenir à ce prompt si rapport non envoyé ou health check non fonctionnel.

---

## 🎉 Fin de la roadmap

À ce stade, le bot tourne en production avec :
- Signal engine validé statistiquement (Sharpe ≥ 1, DSR > 0, DD < 15%, WR > 30%, ≥ 30 trades/an)
- Walk-forward continu de re-training
- Alertes Telegram formatées avec prompt de validation IA
- Monitoring + alertes erreurs + rapport hebdomadaire
- Déploiement automatisé (VPS / Docker / GitHub Actions)

**Prochaines étapes hors-scope** :
- Connexion broker live (XTB API ou MT5) pour exécution automatique
- A/B testing de variantes du modèle en paper trading
- Ajout de nouveaux actifs / nouvelles stratégies au fil de l'eau (chaque ajout = nouvelle hypothèse documentée)

`JOURNAL.md` devient ton suivi continu : une entrée par session, par hypothèse, par incident.
