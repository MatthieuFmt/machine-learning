# Prompt 19 — H18 : Pipeline walk-forward continu

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (✅ GO validation finale)
3. `prompts/18_validation_finale.md`
4. `docs/v3_roadmap.md` section "H18"

## Objectif
**Question H18** : Un pipeline de réentraînement périodique automatique (6 mois) maintient-il le Sharpe ≥ 1.0 sur du walk-forward continu ?

Ce prompt produit le moteur de re-training automatique qui sera utilisé en production (prompt 24 monitoring).

## Definition of Done (testable)
- [ ] `app/pipelines/walk_forward.py` contient `WalkForwardEngine(config)` qui :
  - Tous les 6 mois (01/01 et 01/07), réentraîne les RF méta-labeling sur la fenêtre expansive ≤ date
  - Grid search des params Donchian sur train
  - Recalibre le seuil méta-labeling (plancher 0.50)
  - Recalcule la matrice de corrélation pour la pondération
  - **Sauvegarde l'état dans `models/snapshots/<YYYY-MM-DD>.pkl`** avec metadata complète :
    - `timestamp_utc` (ISO 8601)
    - `git_hash` (depuis `subprocess.run(["git", "rev-parse", "HEAD"])`)
    - `config_version` (lu depuis `ProductionConfig.version`)
    - `train_metrics` (Sharpe train, n_samples, classes_balance)
    - `data_hashes` (SHA256 des CSV d'input pour reproductibilité)
- [ ] **Rollback automatique** : si Sharpe segment < 0 sur **2 segments consécutifs**, le moteur charge automatiquement le snapshot N-2 et alerte via Telegram (`alerter.send_error("Rollback auto vers <date>")`).
- [ ] `models/snapshots/` est dans `.gitignore`.
- [ ] Symlink/fichier `models/snapshots/latest.pkl` pointe vers le dernier snapshot valide (utilisé par le prompt 20).
- [ ] `scripts/run_h18_walk_forward.py` simule le walk-forward sur 2024-2025 et calcule les métriques agrégées.
- [ ] `tests/integration/test_walk_forward_engine.py` : test que le réentraînement à t+6M utilise bien toutes les données ≤ t+6M (et pas plus).
- [ ] Rapport `docs/v3_hypothesis_18.md` avec :
  - Sharpe walk-forward continu
  - Décomposition par tranche 6M
  - Liste des changements de config à chaque rebalance (params Donchian, seuil RF)
  - Comparaison vs validation finale prompt 18 (config figée)
- [ ] `JOURNAL.md` mis à jour : si Sharpe walk-forward ≥ 1.0 → **✅ GO déploiement Phase 4**.

## NE PAS FAIRE
- Ne PAS réentraîner plus fréquemment que 6 mois (overfit + coût compute).
- Ne PAS oublier de purger les données train (purge 1 % de la fenêtre).
- Ne PAS sauvegarder les snapshots dans git (les mettre dans `models/snapshots/` + `.gitignore`).
- Ne PAS lancer Python automatiquement.

## Étapes

### Étape 1 — Architecture
```python
@dataclass
class WalkForwardConfig:
    retrain_months: int = 6
    retrain_dates: list[str]  # ['2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01']
    purge_pct: float = 0.01
    snapshot_dir: Path = Path("models/snapshots")


class WalkForwardEngine:
    def __init__(self, config: WalkForwardConfig, pipeline_config: ProductionConfig):
        self.config = config
        self.pipeline = pipeline_config

    def run(self) -> pd.Series:
        """Simule le walk-forward continu. Retourne l'equity curve."""
        equity = pd.Series(dtype=float)
        for retrain_date in self.config.retrain_dates:
            self._retrain(end_date=retrain_date)
            equity_segment = self._predict_segment(
                start=retrain_date,
                end=self._next_retrain_date(retrain_date),
            )
            equity = pd.concat([equity, equity_segment])
        return equity
```

### Étape 2 — Re-training avec metadata
```python
import hashlib
import pickle
import subprocess
from datetime import datetime, timezone

def _git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"

def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _retrain(self, end_date: str) -> None:
    df_train_dict = {a: load_asset(a, tf).loc[:end_date] for a in self.assets}
    # Grid search Donchian, train RF meta, correlation matrix...
    model = build_model_state(df_train_dict, ...)
    train_metrics = compute_train_metrics(model, df_train_dict)
    data_hashes = {a: _hash_file(Path(f"data/raw/{a}/D1.csv")) for a in self.assets}

    snapshot = {
        "model": model,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "end_date": end_date,
            "git_hash": _git_hash(),
            "config_version": self.production_config.version,
            "train_metrics": train_metrics,
            "data_hashes": data_hashes,
        },
    }
    snapshot_path = self.config.snapshot_dir / f"{end_date}.pkl"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)
    # Update latest symlink/file
    latest = self.config.snapshot_dir / "latest.pkl"
    if latest.exists():
        latest.unlink()
    latest.write_bytes(snapshot_path.read_bytes())
```

### Étape 2 bis — Rollback automatique
```python
def _check_and_rollback(self, segment_sharpes: list[float]) -> None:
    """Si Sharpe < 0 sur 2 segments consécutifs, rollback au snapshot N-2."""
    if len(segment_sharpes) < 2:
        return
    if segment_sharpes[-1] < 0 and segment_sharpes[-2] < 0:
        snapshots = sorted(self.config.snapshot_dir.glob("[0-9]*.pkl"))
        if len(snapshots) >= 3:
            rollback_to = snapshots[-3]  # N-2 dans l'historique des snapshots
            (self.config.snapshot_dir / "latest.pkl").write_bytes(rollback_to.read_bytes())
            self._alert_rollback(rollback_to)


def _alert_rollback(self, snapshot: Path) -> None:
    from app.live.telegram_bot import TelegramAlerter
    alerter = TelegramAlerter(token=os.environ["TELEGRAM_BOT_TOKEN"],
                              chat_id=os.environ["TELEGRAM_CHAT_ID"])
    alerter.send_error(f"⚠️ Rollback auto vers {snapshot.name} (Sharpe < 0 sur 2 segments)")
```

### Étape 3 — Test d'intégration
Vérifier que `_retrain(end_date="2024-01-01")` n'utilise aucune donnée > 2024-01-01.

### Étape 4 — Comparaison

| Métrique | Validation finale (config figée) | Walk-forward continu |
|---|---|---|
| Sharpe | 1.3 | ? |
| Max DD | 12 % | ? |
| Trades/an | 42 | ? |

## Critères go/no-go
- **GO prompt 20** si : Sharpe walk-forward continu ≥ 1.0 ET DD ≤ 25 % (cf. v3_roadmap.md H18).
- **NO-GO** : revenir à prompt 18 et reconsidérer. Le walk-forward est plus dur que la validation figée — si on échoue ici, on n'est pas prêt pour la prod.
