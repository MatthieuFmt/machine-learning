# Prompt 20 — Moteur de signaux live

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (✅ GO H18 walk-forward)
3. `prompts/19_h18_walk_forward_continu.md`

## Objectif
Construire le moteur qui, **étant donné les données actualisées du jour**, calcule les signaux du portfolio retenu et produit un fichier `signals/today.json` lisible par le bot Telegram (prompt 21).

## Definition of Done (testable)
- [ ] `app/live/signal_engine.py` contient `compute_today_signals(data_root: Path, model_snapshot: Path) -> list[Signal]` qui :
  - Charge les CSV de tous les actifs du portfolio retenu
  - Vérifie que la dernière barre est récente (< 48h pour D1, < 8h pour H4)
  - Charge le modèle snapshotté le plus récent
  - Calcule les signaux du dernier jour pour chaque sleeve
  - Applique le filtre régime, la pondération portfolio
  - Retourne une liste de `Signal(asset, direction, entry_price, sl, tp, size_lots, size_units, confidence, regime, strategy_name)`
- [ ] `Signal` est une dataclass typée :
  ```python
  @dataclass
  class Signal:
      asset: str
      tf: str
      direction: Literal["LONG", "SHORT"]
      entry_price: float
      stop_loss: float
      take_profit: float
      risk_reward: float
      size_lots: float
      size_units: float
      confidence: float
      regime: Literal["Trending", "Ranging"]
      strategy_name: str
      timestamp_utc: str
      meta_data: dict  # context : ATR, RSI, prochain event éco, etc.
  ```
- [ ] `app/live/sizing.py` calcule la taille à partir de `CAPITAL_EUR=10000`, `RISK_PER_TRADE=0.02`, distance au SL, et `pip_value_eur` de l'asset config.
- [ ] `scripts/run_today_signals.py --output signals/today.json` produit le fichier JSON.
- [ ] Si aucun signal → `signals/today.json = []` (liste vide), pas d'erreur.
- [ ] Tests d'intégration : `tests/integration/test_signal_engine.py` avec données synthétiques.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS scraper des données depuis une API live ici — on lit `data/raw/`. La mise à jour des CSV est externe (l'utilisateur ou un autre script).
- Ne PAS exécuter d'ordre — uniquement produire un signal.
- Ne PAS écrire dans le fichier JSON si la dernière barre est trop vieille — lever `StaleDataError`.
- Ne PAS oublier les coûts dans le calcul du TP (TP = entry + 2 × (entry - SL) après spread).
- Ne PAS appeler `load_asset()` sans retry (Règle 11 constitution).
- Ne PAS recompute ATR/ADX à chaque appel — utiliser `functools.lru_cache` ou disk cache.
- Ne PAS produire de signal pendant un weekend ou jour férié XTB (`is_market_open` de `app/config/calendar.py`).
- Ne PAS hardcoder `models/snapshots/latest.pkl` — passer via `os.getenv("MODEL_SNAPSHOT")`.

## Étapes

### Étape 1 — Dataclass `Signal`
Comme défini ci-dessus.

### Étape 2 — `compute_today_signals` (avec retry, cache, calendar)
```python
from datetime import datetime, timezone
from functools import lru_cache

from app.core.retry import retry_with_backoff
from app.config.calendar import is_market_open
from app.config.models import ProductionConfig


@retry_with_backoff(max_attempts=3, exceptions=(OSError,))
def load_model(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=32)
def _cached_atr(asset: str, tf: str, last_ts: pd.Timestamp, period: int = 14) -> float:
    """Cache par (asset, tf, last_ts) — invalide automatiquement à nouvelle barre."""
    df = load_asset(asset, tf)
    return float(atr(df, period).iloc[-1])


def compute_today_signals(data_root: Path, model_snapshot: Path) -> list[Signal]:
    signals: list[Signal] = []
    portfolio_config: ProductionConfig = load_production_config()  # typée frozen

    # Vérifier que le marché est ouvert pour AU MOINS un asset
    now = datetime.now(timezone.utc)
    any_open = any(is_market_open(s.asset, now) for s in portfolio_config.sleeves)
    if not any_open:
        logger.info("Tous les marchés sont fermés (weekend/holiday). Aucun signal.")
        return []

    snapshot = load_model(model_snapshot)
    model = snapshot["model"]

    for sleeve in portfolio_config.sleeves:
        df = load_asset(sleeve.asset, sleeve.tf, data_root)
        check_staleness(df, sleeve.tf)

        sig_series = sleeve.strategy.generate_signals(df)
        last_signal = sig_series.iloc[-1]

        if last_signal == 0:
            continue

        # Régime filter
        if portfolio_config.regime_filter:
            regime = classify_regime(df).iloc[-1]
            if regime == 0:  # Ranging
                continue

        # Meta-labeling
        if portfolio_config.meta_labeling:
            features = compute_meta_features(df, sleeve)
            proba = model[sleeve.asset].predict_proba(features.iloc[-1:].values)[0, 1]
            if proba < portfolio_config.meta_thresholds[sleeve.asset]:
                continue
        else:
            proba = float("nan")

        # Compute entry/SL/TP/size
        entry = float(df["close"].iloc[-1])
        atr_val = atr(df, 14).iloc[-1]
        if last_signal == 1:
            sl = entry - 1.5 * atr_val
            tp = entry + 2 * (entry - sl)
            direction = "LONG"
        else:
            sl = entry + 1.5 * atr_val
            tp = entry - 2 * (sl - entry)
            direction = "SHORT"

        size_lots, size_units = compute_size(sleeve.asset, entry, sl)

        signals.append(Signal(
            asset=sleeve.asset, tf=sleeve.tf, direction=direction,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            risk_reward=2.0,
            size_lots=size_lots, size_units=size_units,
            confidence=proba,
            regime="Trending" if portfolio_config.regime_filter else "N/A",
            strategy_name=sleeve.strategy.name,
            timestamp_utc=df.index[-1].isoformat(),
            meta_data={
                "atr": float(atr_val),
                "rsi": float(rsi(df["close"], 14).iloc[-1]),
                "next_high_impact_event_hours": ...,
            },
        ))

    return signals
```

### Étape 3 — Sizing
```python
def compute_size(asset: str, entry: float, sl: float) -> tuple[float, float]:
    cfg = ASSET_CONFIG[asset]
    capital = float(os.getenv("CAPITAL_EUR", "10000"))
    risk_pct = float(os.getenv("RISK_PER_TRADE", "0.02"))
    risk_eur = capital * risk_pct

    distance_points = abs(entry - sl)
    units = risk_eur / (distance_points * cfg.point_value_eur)
    lots = units / cfg.contract_size
    return round(lots, 2), round(units, 0)
```

### Étape 4 — Test intégration
Données synthétiques : un actif US30 D1 avec une cassure Donchian le dernier jour → vérifier qu'un signal LONG est produit avec SL/TP cohérents.

## Critères go/no-go
- **GO prompt 21** si : `scripts/run_today_signals.py` produit un fichier JSON valide sur des données réelles (test manuel utilisateur).
- **NO-GO** : revenir à ce prompt si l'intégration model snapshot / config production est défectueuse.
