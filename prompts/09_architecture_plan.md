# Prompt 09 — Plan d'architecture H08 : Portfolio equal-risk multi-actif

> **Mode** : Architect (planification uniquement, pas d'implémentation)
> **Cible** : passage en mode Code pour exécution

---

## 1. Constat de départ — Situation réelle H06/H07

| Prompt | Hypothèse | Verdict | Sleeves GO |
|--------|-----------|---------|------------|
| 07 | H06 Donchian multi-actif | 🔴 NO-GO | 0/6 |
| 08 | H07 Stratégies alt | 🔴 NO-GO | 0/4 |

**Aucune sleeve n'est validée GO.** Le portfolio equal-risk sera construit avec les meilleures configurations « best effort » identifiées — le Sharpe portfolio sera très probablement négatif. L'infrastructure reste nécessaire pour les prompts suivants (H09 régime, H10-H12 méta-labeling, H13-H15 portfolio avancé).

---

## 2. Fichiers à créer

| # | Fichier | Responsabilité |
|---|---------|---------------|
| 1 | [`app/portfolio/__init__.py`](app/portfolio/__init__.py) | Exports publics |
| 2 | [`app/portfolio/constructor.py`](app/portfolio/constructor.py) | `build_equal_risk_portfolio()`, `vol_target_weights()`, `equal_risk_allocation()` |
| 3 | [`app/portfolio/correlation.py`](app/portfolio/correlation.py) | `filter_correlated_sleeves()` + tie-breaker |
| 4 | [`scripts/run_h08_portfolio.py`](scripts/run_h08_portfolio.py) | Orchestrateur : charge sleeves → portfolio → validate_edge |
| 5 | [`tests/unit/test_portfolio_constructor.py`](tests/unit/test_portfolio_constructor.py) | ≥ 5 tests unitaires |

## 3. Fichiers à modifier

| # | Fichier | Modification |
|---|---------|-------------|
| 6 | [`app/config/instruments.py`](app/config/instruments.py) | Étendre `AssetConfig` : `spread_bid_ask`, `slippage_min`, `slippage_max`, `point_value_eur`, `contract_size` |
| 7 | [`app/backtest/simulator.py`](app/backtest/simulator.py) | Ajouter `apply_costs()` avec slippage stochastique + corriger imports cassés `learning_machine_learning.*` |

## 4. Fichiers à créer (docs)

| # | Fichier | Contenu |
|---|---------|---------|
| 8 | [`docs/v3_hypothesis_08.md`](docs/v3_hypothesis_08.md) | Rapport H08 (sleeves inclus/exclus, Sharpe portfolio, DD, comparaison sleeves individuels) |

---

## 5. Architecture détaillée des modules

### 5.1 [`app/portfolio/constructor.py`](app/portfolio/constructor.py)

```python
"""Construction de portefeuille equal-risk avec volatility targeting.

Trois fonctions pures, vectorisées pandas — zéro boucle Python row-by-row.
"""

def vol_target_weights(
    returns: dict[str, pd.Series],
    target_vol: float = 0.10,
    leverage_max: float = 2.0,
    window: int = 60,
) -> pd.DataFrame:
    """Calcule les poids de vol targeting par sleeve.

    Pour chaque sleeve :
      sigma_daily = returns.rolling(window).std()
      sigma_annual = sigma_daily * sqrt(252)
      weight = (target_vol / sigma_annual).clip(upper=leverage_max)

    Returns:
        pd.DataFrame indexé par date, colonnes = noms de sleeves.
        Poids > 1 = levier. Poids = 0 quand sigma_daily == 0.
    """
    # Union des index de dates pour alignement
    all_dates = sorted(set().union(*(r.index for r in returns.values())))
    date_idx = pd.DatetimeIndex(all_dates).sort_values()

    weights = pd.DataFrame(0.0, index=date_idx, columns=list(returns.keys()))
    annual_factor = np.sqrt(252)

    for name, r in returns.items():
        aligned = r.reindex(date_idx, fill_value=0.0)
        sigma_daily = aligned.rolling(window, min_periods=window).std()
        # Éviter division par zéro
        sigma_annual = sigma_daily * annual_factor
        mask = sigma_annual > 1e-10
        weights.loc[mask, name] = (target_vol / sigma_annual.loc[mask]).clip(upper=leverage_max)
        # sigma_annual == 0 → weight reste 0.0

    return weights.fillna(0.0)


def equal_risk_allocation(
    returns: dict[str, pd.Series],
    target_vol: float = 0.10,
    leverage_max: float = 2.0,
    window: int = 60,
    active_sleeves: list[str] | None = None,
) -> pd.DataFrame:
    """Allocation equal-risk : chaque sleeve active reçoit 1/N du capital,
    puis vol targeting appliqué.

    Args:
        active_sleeves: Si None, toutes les sleeves sont actives.
            Sinon, uniquement les sleeves listées (post-filtre corrélation).

    Returns:
        pd.DataFrame — poids par sleeve, somme ≤ len(active_sleeves).
    """
    if active_sleeves is not None:
        filtered = {k: v for k, v in returns.items() if k in active_sleeves}
    else:
        filtered = returns

    if not filtered:
        return pd.DataFrame()

    w = vol_target_weights(filtered, target_vol, leverage_max, window)
    n = len(filtered)
    return w / n


def build_equal_risk_portfolio(
    returns: dict[str, pd.Series],
    target_vol: float = 0.10,
    leverage_max: float = 2.0,
    correlation_cap: float = 0.7,
    window_vol: int = 60,
    window_corr: int = 60,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Pipeline complet : filtre corrélation → equal risk → equity portfolio.

    Returns:
        (portfolio_equity: pd.Series, weights: pd.DataFrame, active_sleeves: list[str])
    """
    from app.portfolio.correlation import filter_correlated_sleeves

    active = filter_correlated_sleeves(returns, cap=correlation_cap, window=window_corr)
    weights = equal_risk_allocation(
        returns, target_vol, leverage_max, window_vol, active_sleeves=active,
    )

    # Reconstruire le DataFrame de retours aligné
    returns_df = pd.DataFrame(returns).fillna(0.0)
    _, weights_aligned = returns_df.align(weights, axis=0, join="outer")
    weights_aligned = weights_aligned.fillna(0.0)

    portfolio_returns = (weights_aligned * returns_df.fillna(0.0)).sum(axis=1)
    equity = (1.0 + portfolio_returns).cumprod()
    equity.iloc[0] = 1.0  # Base 1.0 pour pct_change()

    return equity, weights_aligned, active
```

**Edge cases traités** :
- `sigma_daily == 0` → `sigma_annual == 0` → division évitée via `mask > 1e-10`, weight reste 0
- Sleeve sans trades → returns = série de zéros → sigma = 0 → poids = 0 → pas d'impact
- Index désalignés entre sleeves → `pd.DataFrame(returns).fillna(0.0)` + `align(axis=0, join="outer")`
- `active_sleeves` vide après filtre → retourne DataFrame vide, equity = série [1.0]
- `leverage_max` dépassé → `clip(upper=leverage_max)`

---

### 5.2 [`app/portfolio/correlation.py`](app/portfolio/correlation.py)

```python
"""Filtre de corrélation entre sleeves — désactive les sleeves redondantes."""

def _rolling_sharpe(
    returns: pd.Series,
    window: int = 126,  # ~6 mois de trading days
) -> pd.Series:
    """Sharpe rolling sur fenêtre glissante."""
    roll_mean = returns.rolling(window, min_periods=window).mean()
    roll_std = returns.rolling(window, min_periods=window).std()
    mask = roll_std > 1e-10
    sr = pd.Series(0.0, index=returns.index)
    sr[mask] = roll_mean[mask] / roll_std[mask] * np.sqrt(252)
    return sr


def filter_correlated_sleeves(
    returns: dict[str, pd.Series],
    cap: float = 0.7,
    window: int = 60,
    sharpe_window: int = 126,
) -> list[str]:
    """Désactive les sleeves dont la corrélation rolling > cap.

    Règle : si ρ_ij > cap pour deux sleeves, on garde celle avec le meilleur
    Sharpe rolling 6M. Tie-breaker (prompt 09 §4 bis) :
      1. Moins de trades (proxy via coûts cumulés — non disponible ici → comptage
         des jours de retour non-nuls)
      2. Ordre alphabétique de la clé sleeve (déterministe)

    Returns:
        Liste ordonnée des noms de sleeves conservées.
    """
    names = list(returns.keys())
    if len(names) <= 1:
        return names

    # Matrice de corrélation rolling (moyenne sur la période)
    returns_df = pd.DataFrame(returns).fillna(0.0)
    corr_matrix = returns_df.rolling(window, min_periods=window).corr().groupby(level=1).mean()

    # Sharpe rolling 6M par sleeve (moyen)
    sharpe_avg: dict[str, float] = {}
    for name in names:
        sr = _rolling_sharpe(returns[name], window=sharpe_window)
        sharpe_avg[name] = float(sr.mean()) if len(sr) > 0 else 0.0

    # Comptage des jours avec retour non-nul (proxy activité)
    active_days: dict[str, int] = {
        name: int((returns[name].abs() > 1e-10).sum()) for name in names
    }

    # Tri initial par Sharpe décroissant
    sorted_names = sorted(names, key=lambda n: sharpe_avg.get(n, 0.0), reverse=True)
    active: set[str] = set()

    for name in sorted_names:
        conflict = False
        for kept in active:
            rho = float(corr_matrix.loc[name, kept]) if name in corr_matrix.index and kept in corr_matrix.columns else 0.0
            if abs(rho) > cap:
                # Tie-breaker si Sharpe identique à 1e-4 près
                if abs(sharpe_avg[name] - sharpe_avg[kept]) < 1e-4:
                    # 1. Moins de trades → garder celle avec le moins de jours actifs
                    if active_days[name] < active_days[kept]:
                        # name a moins de jours actifs → on garde name, on vire kept
                        active.discard(kept)
                        active.add(name)
                    # Sinon on garde kept (moins de jours actifs OU égal → ordre alpha)
                    # Si égalité jours actifs → ordre alphabétique [name < kept → name gagne]
                    elif active_days[name] == active_days[kept] and name < kept:
                        active.discard(kept)
                        active.add(name)
                conflict = True
                break
        if not conflict:
            active.add(name)

    # Retour dans l'ordre d'entrée original (déterministe)
    return [n for n in names if n in active]
```

**Edge cases** :
- Moins de 2 sleeves → pas de filtre, retour direct
- Fenêtre insuffisante pour corrélation → `min_periods=window` → NaN ignoré
- ρ == cap exactement → `>` strict (on garde les deux si ρ == 0.7)
- Toutes les sleeves filtrées → garder la meilleure (première de `sorted_names`) — impossible avec la logique car la première est toujours ajoutée
- ρ négatif > cap en valeur absolue → `abs(rho) > cap` capture les corrélations négatives fortes aussi

---

### 5.3 [`scripts/run_h08_portfolio.py`](scripts/run_h08_portfolio.py)

Sleeves candidates (best effort — aucune GO, mais infrastructure validée) :

| Sleeve Key | Stratégie | Actif | Params | Source |
|------------|-----------|-------|--------|--------|
| `donchian_US30` | Donchian | US30 | N=100, M=10 | H06 (meilleur Sharpe train) |
| `donchian_XAUUSD` | Donchian | XAUUSD | N=100, M=20 | H06 (Sharpe test +1.46) |
| `donchian_GER30` | Donchian | GER30 | N=50, M=10 | H06 |
| `donchian_US500` | Donchian | US500 | N=50, M=50 | H06 |
| `donchian_XAGUSD` | Donchian | XAGUSD | N=20, M=10 | H06 |
| `dual_ma_US30` | Dual MA | US30 | fast=10, slow=50 | H07 (seul Sharpe test > 0) |
| `keltner_US30` | Keltner | US30 | period=20, mult=2.0 | H07 (overfit val mais diversifiant) |

**Flow** :
```
1. set_global_seeds() + check_unlocked()
2. Pour chaque sleeve candidate :
   a. load_asset(asset, "D1")
   b. split train/val/test
   c. strategy.generate_signals(df_test)
   d. run_deterministic_backtest(...)
   e. Extraire daily_returns[période test]
   f. Stocker dans returns[ sleeve_key ]
3. equity_portfolio, weights, active = build_equal_risk_portfolio(returns)
4. Concaténer tous les trades en un DataFrame unique (colonne 'pnl')
5. report = validate_edge(equity_portfolio, all_trades, n_trials=8)
6. read_oos(prompt="09", hypothesis="H08", sharpe=..., n_trades=...)
7. Sauvegarder predictions/h08_portfolio.json
```

**Détail extraction des retours quotidiens** (réutilisation du helper H07) :
```python
def _sleeve_daily_returns(
    df_test: pd.DataFrame,
    signals: pd.Series,
    config: AssetConfig,
) -> pd.Series:
    """Backtest une sleeve sur le test set, retourne les retours quotidiens."""
    result = run_deterministic_backtest(
        df=df_test, signals=signals,
        tp_pips=config.tp_points,
        sl_pips=config.sl_points,
        window_hours=config.window_hours,
        commission_pips=config.spread_pips,
        slippage_pips=config.slippage_pips,
        pip_size=config.pip_size,
    )
    trades = result.get("trades", [])
    if not trades:
        return pd.Series(0.0, index=df_test.index)
    # Reconstruire equity → retours quotidiens
    pnls = np.array([t["pips_net"] for t in trades])
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])
    equity = pd.Series(np.cumsum(pnls), index=exit_times).sort_index()
    equity = equity.reindex(df_test.index, method="ffill").fillna(0.0)
    # Convertir pips → rendement en %
    capital = 10_000.0
    equity_eur = capital + equity * config.pip_value_eur
    daily = equity_eur.resample("D").last().ffill()
    return daily.pct_change().fillna(0.0)
```

---

### 5.4 [`tests/unit/test_portfolio_constructor.py`](tests/unit/test_portfolio_constructor.py)

Tests requis (≥ 5) :

| # | Test | Vérification |
|---|------|-------------|
| 1 | `test_equal_risk_two_sleeves` | 2 sleeves, poids ~ 0.5/n après vol targeting |
| 2 | `test_correlation_cap_filters_sleeve` | ρ=0.9 entre 2 sleeves → une seule gardée |
| 3 | `test_vol_targeting_clip` | σ très faible → poids clip à leverage_max |
| 4 | `test_leverage_cap` | target_vol / sigma > 2.0 → poids = 2.0 exactement |
| 5 | `test_rebalance_zero_sigma` | Sleeve sans variance → poids = 0, pas d'erreur |
| 6 | `test_filter_correlated_tie_breaker_sharpe` | Sharpe identique → moins de jours actifs gagne |
| 7 | `test_filter_correlated_tie_breaker_alpha` | Sharpe et jours actifs identiques → ordre alpha |
| 8 | `test_empty_returns` | returns vide → DataFrame vide, pas d'exception |

---

### 5.5 Modification de [`app/config/instruments.py`](app/config/instruments.py:214)

Ajout dans `AssetConfig` :

```python
@dataclass(frozen=True)
class AssetConfig:
    # Existants
    spread_pips: float
    slippage_pips: float
    commission_pips: float = 0.0
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    min_lot: float = 0.01
    max_lot: float = 10.0
    tp_atr_multiplier: float = 2.0
    sl_atr_multiplier: float = 1.0
    tp_points: float = 200
    sl_points: float = 100
    window_hours: int = 120

    # NOUVEAU — Prompt 09 §5 : slippage stochastique
    spread_bid_ask: float = 0.0   # spread moyen en points (sera divisé par 2: half-spread par côté)
    slippage_min: float = 0.0     # slippage min en points
    slippage_max: float = 0.0     # slippage max en points
    point_value_eur: float = 0.0  # valeur d'un point en EUR
    contract_size: float = 1.0    # taille du contrat
```

Les champs `spread_bid_ask`, `slippage_min`, `slippage_max` sont initialisés à 0.0 pour rétrocompatibilité et seront renseignés dans `ASSET_CONFIGS` progressivement. En attendant, le `slippage_pips` existant reste utilisé.

---

### 5.6 Modification de [`app/backtest/simulator.py`](app/backtest/simulator.py:1)

Ajout de `apply_costs()` :

```python
def apply_costs(
    entry_price: float,
    direction: int,
    asset_cfg: AssetConfig,
    rng: np.random.Generator | None = None,
) -> float:
    """Applique spread bid-ask + slippage stochastique au prix d'entrée.

    Args:
        entry_price: Prix théorique (Close de la barre de signal).
        direction: 1 = LONG, -1 = SHORT.
        asset_cfg: Configuration avec spread_bid_ask, slippage_min/max.
        rng: Générateur aléatoire (seed fixé par set_global_seeds).

    Returns:
        Prix ajusté après coûts.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Half-spread : un seul côté payé (le spread bid-ask complet est payé
    # moitié à l'entrée, moitié à la sortie)
    spread_cost = asset_cfg.spread_bid_ask / 2.0

    # Slippage stochastique uniforme
    slippage = rng.uniform(asset_cfg.slippage_min, asset_cfg.slippage_max)

    # Pour un LONG : on achète au Ask (entry_price + half-spread + slippage)
    # Pour un SHORT : on vend au Bid (entry_price - half-spread - slippage)
    adjusted = entry_price + direction * (spread_cost + slippage)
    return adjusted
```

**Note sur les imports cassés** : [`app/backtest/simulator.py`](app/backtest/simulator.py:18-19) importe `from learning_machine_learning.backtest.filters` et `from learning_machine_learning.core.logging`. Ces imports sont cassés depuis le renommage v2→app (prompt 02). Selon la constitution Règle 4 (pas de scope creep), ce correctif sera documenté dans `JOURNAL.md` sous « Hypothèses à explorer ensuite » — il n'est pas dans le scope strict du prompt 09. Cependant, pour que `apply_costs()` fonctionne sans erreur d'import, les imports cassés doivent être corrigés **a minima** dans ce fichier. Le prompt 09 ne touche pas à `FilterPipeline` ni `get_logger` → on corrige uniquement les imports pour que le module soit importable.

---

## 6. Data flow complet

```
┌──────────────────────────────────────────────────────────────────────┐
│                    scripts/run_h08_portfolio.py                       │
│                                                                       │
│  ┌──────────┐   ┌─────────────┐   ┌────────────────┐                 │
│  │ 7 assets │──▶│ 7 stratégies│──▶│ Backtest       │                 │
│  │ D1 CSV   │   │ (best H06/  │   │ déterministe   │                 │
│  │          │   │  H07 params)│   │ stateful       │                 │
│  └──────────┘   └─────────────┘   └──────┬─────────┘                 │
│                                           │                           │
│                                           ▼                           │
│                              ┌─────────────────────┐                  │
│                              │ returns: dict[str,  │                  │
│                              │   pd.Series]        │                  │
│                              │ 7 sleeves candidates│                  │
│                              └──────┬──────────────┘                  │
│                                     │                                 │
│                                     ▼                                 │
│                      ┌──────────────────────────┐                     │
│                      │ filter_correlated_sleeves│                     │
│                      │   cap = 0.7, window = 60 │                     │
│                      └──────┬───────────────────┘                     │
│                             │                                         │
│                             ▼                                         │
│                      ┌──────────────────────────┐                     │
│                      │ equal_risk_allocation    │                     │
│                      │   target_vol = 10%       │                     │
│                      │   leverage_max = 2.0     │                     │
│                      └──────┬───────────────────┘                     │
│                             │                                         │
│                             ▼                                         │
│               ┌─────────────────────────────┐                         │
│               │ portfolio_equity =          │                         │
│               │   (1 + Σ(w_i × r_i)).cumprod│                         │
│               └─────────────┬───────────────┘                         │
│                             │                                         │
│                             ▼                                         │
│               ┌─────────────────────────────┐                         │
│               │ validate_edge(              │                         │
│               │   equity, all_trades,       │                         │
│               │   n_trials=8)               │                         │
│               └─────────────┬───────────────┘                         │
│                             │                                         │
│                             ▼                                         │
│               ┌─────────────────────────────┐                         │
│               │ read_oos(prompt=09,         │                         │
│               │   hypothesis=H08, ...)      │                         │
│               └─────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7. Go/No-Go attendu

D'après le critère du prompt 09 : **GO si Sharpe portfolio ≥ 0.5**.

Avec 0 sleeve GO en entrée, le Sharpe portfolio sera très probablement négatif. Le scénario le plus probable est un **NO-GO avec Sharpe < 0**, déclenchant un retour au prompt 08. Cependant :

- Le prompt 08 lui-même est NO-GO.
- La roadmap prévoit qu'en cas de NO-GO H08, on continue quand même en H09 (régime) et H10-H12 (méta-labeling) pour tenter d'améliorer les sleeves unitaires.

**Recommandation** : Même si H08 est NO-GO, construire l'infrastructure. Elle sera réutilisée après H10-H12 quand les sleeves seront améliorées.

---

## 8. Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| Toutes les sleeves ont Sharpe négatif → portfolio Sharpe négatif | Très élevée | Sortie NO-GO | Infrastructure prête pour H10-H12 |
| `simulator.py` imports cassés bloquent `apply_costs` | Élevée | Erreur import | Corriger uniquement les 2 lignes d'import (pas le reste du module) |
| Données manquantes pour XAGUSD/USOIL (déjà identifié H06) | Certaine | Sleeves sautées | Try/except par sleeve, continuer avec les sleeves qui chargent |
| Corrélation instable sur petites fenêtres | Moyenne | Filtre trop agressif | `min_periods=window` + fallback à liste complète si < 2 sleeves après filtre |
| `validate_edge` attend `trades['pnl']` mais le backtest utilise `pips_net` | Élevée | KeyError | Renommer `pips_net` → `pnl` avant l'appel |

---

## 9. Séquence d'implémentation (pour le mode Code)

1. Modifier [`app/config/instruments.py`](app/config/instruments.py) — ajouter les champs `AssetConfig`
2. Corriger les imports dans [`app/backtest/simulator.py`](app/backtest/simulator.py) — ajouter `apply_costs()`
3. Créer [`app/portfolio/__init__.py`](app/portfolio/__init__.py)
4. Créer [`app/portfolio/correlation.py`](app/portfolio/correlation.py)
5. Créer [`app/portfolio/constructor.py`](app/portfolio/constructor.py)
6. Créer [`tests/unit/test_portfolio_constructor.py`](tests/unit/test_portfolio_constructor.py) (≥ 5 tests)
7. Créer [`scripts/run_h08_portfolio.py`](scripts/run_h08_portfolio.py)
8. Exécuter `rtk pytest tests/unit/test_portfolio_constructor.py -v`
9. L'utilisateur exécute `python scripts/run_h08_portfolio.py`
10. Rédiger [`docs/v3_hypothesis_08.md`](docs/v3_hypothesis_08.md)
11. Mettre à jour [`JOURNAL.md`](JOURNAL.md)
