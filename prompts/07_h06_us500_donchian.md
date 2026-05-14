# Prompt 07 — H06 : Extension Donchian multi-actif

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/06_validation_framework.md`
4. `docs/v3_roadmap.md` section "H06"

## Objectif
**Question H06** : Le Donchian Breakout (qui fonctionne sur US30 D1) fonctionne-t-il sur d'autres CFD (GER30, US500, XAUUSD, XAGUSD, USOIL, BUND) ?

## Definition of Done (testable)
- [ ] `scripts/run_h06_donchian_multi_asset.py` est créé. Il fait un grid search Donchian sur chaque actif disponible :
  - N ∈ {20, 50, 100}, M ∈ {10, 20, 50} (9 combinaisons)
  - Coûts réalistes par actif (cf. table de `docs/v3_roadmap.md` §3 H06)
  - Split train ≤ 2022 / val = 2023 / test ≥ 2024
- [ ] Pour chaque actif, le script appelle `validate_edge(equity, trades, n_trials=6)`.
- [ ] Sortie : `predictions/h06_donchian_multi_asset.json` avec un block par actif :
  ```json
  {
    "US30": {"best_params": {"N": 20, "M": 20}, "sharpe_test": 3.07, "go": true, ...},
    "GER30": {"best_params": ..., "sharpe_test": ..., "go": false, ...}
  }
  ```
- [ ] Un rapport `docs/v3_hypothesis_06.md` est écrit (modèle : `docs/v2_hypothesis_03.md`) avec :
  - Question, méthode, données utilisées
  - Résultats par actif (Sharpe train / val / test, WR, DD, DSR)
  - Verdict GO / NO-GO global
  - Liste des actifs GO (à utiliser pour H07/H08)
- [ ] `JOURNAL.md` mis à jour avec la liste des actifs GO.

## NE PAS FAIRE
- Ne PAS lancer `python` automatiquement — produire le script et attendre l'instruction utilisateur.
- Ne PAS modifier `app/strategies/donchian.py` (déjà validé v2).
- Ne PAS rejouer le test set après lecture (Règle 9 constitution).
- Ne PAS ajouter d'actif hors de la liste si la donnée n'est pas dans `data/raw/`.

## Étapes

### Étape 1 — Découverte des actifs disponibles
```python
from app.data.registry import discover_assets
assets = discover_assets()
# Filtrer ceux avec D1 disponible
candidates = [a for a, tfs in assets.items() if "D1" in tfs]
```

### Étape 2 — Coûts par actif (depuis `docs/v3_roadmap.md` §3 H06)
Centraliser dans `app/config/instruments.py` une dataclass `AssetConfig` qui contient spread, slippage, TP/SL en points, multiplicateur de devise (pour calculer le PnL en EUR).

### Étape 3 — Boucle de grid search
```python
results = {}
for asset in candidates:
    df = load_asset(asset, "D1")
    df_train = df.loc[:"2022-12-31"]
    df_val = df.loc["2023-01-01":"2023-12-31"]
    df_test = df.loc["2024-01-01":]

    best_sharpe_train = -np.inf
    best_params = None
    for N in [20, 50, 100]:
        for M in [10, 20, 50]:
            equity_train, trades_train = simulate_donchian(df_train, N, M, asset_config[asset])
            sr = sharpe_ratio(equity_train.pct_change().dropna())
            if sr > best_sharpe_train:
                best_sharpe_train = sr
                best_params = {"N": N, "M": M}

    equity_test, trades_test = simulate_donchian(df_test, **best_params, asset_config[asset])
    report = validate_edge(equity_test, trades_test, n_trials=6)
    results[asset] = {
        "best_params": best_params,
        "sharpe_train": best_sharpe_train,
        "sharpe_test": report.metrics["sharpe"],
        "metrics_test": report.metrics,
        "go": report.go,
        "reasons": report.reasons,
    }
```

### Étape 4 — Rapport `docs/v3_hypothesis_06.md`
Section "Verdict" : lister les actifs GO. Si aucun nouvel actif ne passe → fallback documenté (mono-actif US30, cf. critère NO-GO H06).

### Étape 5 — Mettre à jour `JOURNAL.md`
```markdown
## 2026-MM-DD — Prompt 07 / H06 : Donchian multi-actif
- **Statut** : ✅ Terminé
- **Actifs testés** : <liste>
- **Actifs GO** : <liste> (avec Sharpe test)
- **Actifs NO-GO** : <liste> (avec raison)
- **n_trials cumulatif** : 6
- **Fichier rapport** : docs/v3_hypothesis_06.md
- **Décision** : passer à H07 sur <ces actifs>
```

## Critères go/no-go
- **GO prompt 08** si : au moins 1 nouvel actif (en plus de US30) passe `validate_edge` (au moins partiellement — Sharpe > 0 suffit pour passer à H07, le critère final c'est H08/H18).
- **NO-GO partiel** : aucun nouvel actif GO. Continuer quand même mais documenter dans `JOURNAL.md` que l'univers se réduit à US30. Prompt 08 reste pertinent (chercher stratégies alt sur US30 pour décorréler).
