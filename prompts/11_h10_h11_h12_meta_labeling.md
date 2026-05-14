# Prompt 11 — H10/H11/H12 : Filtrage par régime + méta-labeling RF mono- et multi-actif

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (verdicts H06–H09)
3. `prompts/10_h09_regime_detector.md`
4. `docs/v3_roadmap.md` sections "H10", "H11", "H12"

## Objectif
Trois hypothèses groupées car interdépendantes :
- **H10** : Ne trader que en régime Trending améliore-t-il le Sharpe ?
- **H11** : Un RF méta-labeling par actif améliore-t-il le Sharpe unitaire ?
- **H12** : Un RF méta-labeling multi-actif (un seul modèle, feature `Asset_ID`) généralise-t-il mieux que H11 ?

## Definition of Done (testable)
- [ ] **H10** : `scripts/run_h10_regime_filter.py` rejoue chaque sleeve GO avec et sans filtre régime. Sortie A/B comparative `predictions/h10_regime_filter.json` + rapport `docs/v3_hypothesis_10.md`.
- [ ] **H11** : `scripts/run_h11_meta_labeling_per_asset.py` :
  - Pour chaque sleeve GO : entraîner un `RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=10)` sur les features (RSI, ADX, Dist_SMA50, Dist_SMA200, ATR_Norm, Log_Return_5d, Signal_Strategy)
  - Cible binaire : 1 si trade gagnant, 0 sinon
  - Calibrer le seuil sur train, **plancher 0.50** (cf. leçon H04)
  - Si seuil optimal élimine > 80 % des trades val → fallback 0.50
  - Comparer Sharpe meta-labeling vs Sharpe baseline déterministe
- [ ] **H12** : `scripts/run_h12_meta_labeling_multi_asset.py` : un seul RF entraîné sur le concat de tous les sleeves, avec feature `Asset_ID` one-hot. Comparer à H11.
- [ ] Trois rapports : `docs/v3_hypothesis_10.md`, `_11.md`, `_12.md`.
- [ ] CPCV obligatoire pour H11 et H12 (cf. `app/analysis/edge_validation.py`). Critère : Sharpe moyen > baseline ET std CPCV < Sharpe moyen (stabilité minimale).
- [ ] `JOURNAL.md` mis à jour avec la config retenue.

## NE PAS FAIRE
- Ne PAS optimiser le seuil sur val (overfit). Optimisation train uniquement.
- Ne PAS choisir un seuil > 0.60 même si train le suggère (leçon H04 : 0.65 → 0 trade val).
- Ne PAS rejouer le test set après lecture (appel `read_oos()` obligatoire pour tracer).
- Ne PAS ajouter de features hors de la liste H04 sans documenter pourquoi.
- Ne PAS oublier `class_weight="balanced"` (classe déséquilibrée : ~30 % gagnants).
- Ne PAS forcer le méta-labeling si aucun seuil ne fonctionne (fallback baseline déterministe).

## Étapes

### H10 — Filtre régime

```python
for sleeve_name, sleeve in sleeves_go.items():
    asset, strat = parse_name(sleeve_name)
    df = load_asset(asset, "D1")
    regime = classify_regime(df)

    sig_no_filter = strat.generate_signals(df)
    sig_filtered = sig_no_filter.where(regime == 1, 0)

    eq_no_filter = simulate(df, sig_no_filter, asset_config[asset])
    eq_filtered = simulate(df, sig_filtered, asset_config[asset])

    results[sleeve_name] = {
        "sharpe_no_filter": sharpe_ratio(eq_no_filter.pct_change().dropna()),
        "sharpe_filtered": sharpe_ratio(eq_filtered.pct_change().dropna()),
        "n_trades_no_filter": ...,
        "n_trades_filtered": ...,
        "improvement": ...
    }
```

Conserver le filtre régime pour les sleeves où :
- Sharpe filtered > Sharpe no_filter
- ET réduction DD

### H11 — Méta-labeling par actif

```python
from sklearn.ensemble import RandomForestClassifier

def train_meta_rf_per_asset(sleeve_name, signals, features, returns):
    trades_mask = signals != 0
    X = features[trades_mask]
    y = (returns[trades_mask] > 0).astype(int)

    X_train = X.loc[:"2022-12-31"]
    y_train = y.loc[:"2022-12-31"]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        class_weight="balanced",   # classe déséquilibrée (~30% gagnants)
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Calibrate threshold on train uniquement
    proba_train = rf.predict_proba(X_train)[:, 1]
    best_t, best_sharpe = None, -np.inf
    candidates = [0.45, 0.50, 0.55, 0.60]
    for t in candidates:
        mask = proba_train > t
        if mask.sum() < len(X_train) * 0.20:  # plancher : garde ≥ 20% des trades
            continue
        sr = sharpe_at_threshold(t, ...)  # backtest sur train
        if sr > best_sharpe:
            best_sharpe, best_t = sr, t

    # Fallback : si aucun seuil ne garde ≥ 20% des trades, désactiver méta-labeling
    if best_t is None:
        logger.warning(f"{sleeve_name} : aucun seuil ne fonctionne, désactivation méta-labeling")
        return None  # signal : utiliser baseline déterministe pour ce sleeve

    threshold = max(best_t, 0.50)  # plancher absolu
    # ... eval val + test ...
```

CPCV via `app/analysis/edge_validation.py:purged_kfold_cv` (k=5, embargo_pct=0.01).
Embargo de 1 % = 1 % de la fenêtre de chaque fold pour éviter les leaks entre folds adjacents.

**Tracer chaque lecture OOS** :
```python
from app.testing.snooping_guard import read_oos
read_oos(prompt="11", hypothesis=f"H11-{asset}", sharpe=sharpe_test, n_trades=n_trades_test)
```

### H12 — Méta-labeling multi-actif

```python
big_X = []
big_y = []
for sleeve_name in sleeves_meta:
    X, y = build_features_target(sleeve_name)
    X["asset_id"] = sleeve_name.split("_")[1]  # asset name
    big_X.append(X)
    big_y.append(y)

big_X = pd.concat(big_X, axis=0)
big_X = pd.get_dummies(big_X, columns=["asset_id"])
big_y = pd.concat(big_y, axis=0)
```

Entraîner un seul RF. Comparer Sharpe moyen multi-actif vs Sharpe moyen H11.

### Verdicts à inscrire dans `JOURNAL.md`
- Config retenue : `<regime_filter: yes/no>` × `<meta_labeling: none/per_asset/multi_asset>`
- Justification chiffrée.

## Critères go/no-go
- **GO prompt 12** si : la meilleure config (entre H10/H11/H12 et baseline) améliore le Sharpe portfolio H08 d'au moins +0.1.
- **NO-GO partiel** : aucune amélioration. Conserver la baseline H08 et continuer.
