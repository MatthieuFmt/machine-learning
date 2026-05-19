# Audit technique v4 — Findings classés par gravité

**Date** : 2026-05-18
**Auditeur** : analyse automatique du code, pas d'exécution
**Périmètre** : `app/`, `scripts/run_phase_b_*.py`, `scripts/run_validation_finale.py`, `predictions/validation_finale.json`
**Source de vérité** : tous les findings ci-dessous citent **file:line** vérifiable

Le document existant [`docs/audit_final_post_mortem.md`](audit_final_post_mortem.md) couvre les risques *suspectés*. Le présent audit **prouve** lesquels sont des bugs réels dans le code, et en ajoute plusieurs nouveaux.

---

## TL;DR — Verdict synthétique

| Finding | Gravité | Impact sur les conclusions actuelles |
|---|---|---|
| F1. Distribution train/test rompue (Donchian absent du test) | 🔴 BLOQUEUR | Les 4 stratégies Donchian+ML ne sont PAS des stratégies de méta-labeling — c'est du trend-following gated par ML |
| F2. Sharpe gonflé ×3.5 par pct_change sur equity compoundée + sizing fixe | 🔴 BLOQUEUR | Le portfolio Sharpe 4.97 est mécaniquement inflaté. Sharpe "vrai" attendu : ~1.4 |
| F3. TP-prime same-bar (optimiste) au lieu de SL-prime (conservateur) | 🔴 BLOQUEUR | Sur-estime systématiquement le winrate |
| F4. Look-ahead dans Stacking (cv=5 non chronologique) | 🟠 IMPORTANT | EURUSD D1 et USDCHF D1 entraînés avec fuite d'information |
| F5. n_trials=29 utilisé pour le DSR alors que 44 reads OOS sont loggés | 🟠 IMPORTANT | DSR sous-déflaté → fausse significativité |
| F6. Benchmark Monte Carlo mal calibré (US30 seul, signal_freq hardcoded) | 🟠 IMPORTANT | P95=9.96 peu interprétable |
| F7. Cross-asset features par `reindex(method="ffill")` D1→H1 | 🟠 IMPORTANT | Look-ahead possible selon convention timestamp |
| F8. `look_ahead_safe` est un simple marqueur, validation auto skip large | 🟠 IMPORTANT | Faux sentiment de sécurité |
| F9. ETHUSD D1 : 97.1% acc train sur 175 trades → mémorisation flagrante | 🟠 IMPORTANT | Sharpe OOS 2.14 probablement chanceux |
| F10. Coûts PROVISOIRES BTCUSD/ETHUSD/GBPUSD/USDCHF | 🟠 IMPORTANT | Sharpe sensibles aux spreads réels |
| F11. WR=6583.9% dans le rapport markdown (×100 en trop) | 🟢 COSMÉTIQUE | Décrédibilise le rapport humain |
| F12. Max DD strategy-level -1549% — incohérence formule | 🟡 BIZARRE | Numérique impossible mais affiché |
| F13. Walk-forward inappliqué aux 4 stratégies Donchian+ML | 🟡 PROCESS | Robustesse non prouvée |
| F14. Validation 2023 inutilisée (gap dans le split) | 🟡 PROCESS | Information jetée |
| F15. Bootstrap Sharpe iid (non block) | 🟡 STATS | Sous-estime l'incertitude |
| F16. `_compute_sharpe_from_returns` sans annualisation | 🟢 CODE MORT | Bug latent si réactivé |
| F17. Stacking sans tuning (defaults sklearn) | 🟡 STATS | Robustesse non garantie |
| F18. Window_hours converti via durée moyenne biaisée par weekends | 🟡 STATS | Fenêtre plus courte que prévu |
| F19. tests/unit/ contient 15 fichiers `.bak` | 🟢 DETTE | Coverage incomplète |

Verdict combiné : **les 6 stratégies "GO" ne tiennent plus une fois F1+F2+F3 corrigés**. Les chiffres affichés (Sharpe 4.97, DSR 19.47) sont des artefacts de la chaîne de calcul, pas un edge réel.

---

## 🔴 BLOQUEURS (méthodologie cassée)

### F1 — Distribution train/test rompue : le Donchian disparaît au test

**Fichiers** :
- [scripts/run_validation_finale.py:189-224](../scripts/run_validation_finale.py#L189-L224)
- [scripts/run_phase_b_c5_extra_gbpusd_d1.py:100-139](../scripts/run_phase_b_c5_extra_gbpusd_d1.py#L100-L139)
- Idem pour les 5 autres scripts `phase_b_c5_*` qui clonent ce pattern

**Bug** :
```python
# TRAIN — modèle entraîné sur features aux entrées Donchian (target = winner)
donchian_signals_train = _generate_donchian_signals(df_train)
trades_donchian_train = run_deterministic_backtest(df_train, donchian_signals_train, ...)
X_train = features.loc[entry_times_train]    # features uniquement aux breakouts Donchian
y_train = (pips_net > 0).astype(int)
model.fit(X_train, y_train)

# TEST — Donchian COMPLÈTEMENT IGNORÉ
def _generate_model_signals(df, model):
    features = build_features(df)               # features sur TOUTES les barres
    proba = model.predict_proba(features)[:, 1]
    trend_sign = features[["slope_sma_20", "slope_sma_50", "dist_sma_200"]].mean(axis=1).apply(np.sign)
    long_mask  = (proba > threshold) & (trend_sign > 0)
    short_mask = (proba > threshold) & (trend_sign < 0)
```

**Preuve chiffrée** : GBPUSD D1, 12 ans de train → 597 trades Donchian (~50/an). 2 ans de test → 483 trades. **Un D1 Donchian ne génère pas 240 signaux/an**. C'est ~quotidien → la phase test est devenue du trend-following pur avec un gate de probabilité.

**Conséquence** :
- Le modèle prédit `P(winner | features à un breakout Donchian)`.
- Il est appliqué à `P(winner | features à n'importe quelle barre)` — distribution conditionnelle complètement différente.
- Le `trend_sign` (mean des slopes SMA) **décide seul de la direction**. Le ML ne fait que filtrer.
- Pendant un marché trendant (USD fort en 2024-2026), le trend-following naïf gagne quoi qu'il arrive.

**Le nom "Donchian + méta-labeling" est trompeur**. Réalité : "trend-following SMA gated par RF entraîné sur autre chose".

### F2 — Sharpe gonflé ×3.5 par incohérence sizing / equity

**Fichiers** :
- [app/backtest/metrics.py:128-136](../app/backtest/metrics.py#L128-L136) (`sharpe_annualized` mode daily)
- [scripts/run_validation_finale.py:107-146](../scripts/run_validation_finale.py#L107-L146) (`_trades_to_equity`)
- [predictions/phase_b_c5_extra_gbpusd_d1.json:47-49](../predictions/phase_b_c5_extra_gbpusd_d1.json#L47-L49)

**Preuve** : Même set de trades, deux Sharpe :
```json
"sharpe_backtest":         1.4916,   // calculé sur equity en pips, daily
"sharpe_compute_metrics":  5.1686,   // calculé sur equity en € compoundée, daily pct_change
```

**Cause** :
- `compute_position_size` utilise `capital_eur=10000` (constant) → lots ~constants.
- Le sizing en lots ne croît PAS avec l'equity.
- Mais `compute_metrics` construit `equity = capital_eur + pnl_eur.cumsum()` puis fait `equity.pct_change()`.
- Le `pct_change` divise chaque variation par l'equity **précédente** — implicitement, c'est un calcul de Sharpe en composé.
- Or, en réalité, les lots sont fixés sur 10k €, pas sur 90k €. Les retours réels sont **linéaires**, pas composés.

**Correction**: utiliser `daily_pnl_eur.cumsum() / capital_eur` puis prendre le `diff()` → returns **linéaires**. Le Sharpe vrai chute à ~1.4.

### F3 — TP-prime same-bar (optimiste), incohérent avec la spec

**Fichier** : [app/backtest/deterministic.py:132-156](../app/backtest/deterministic.py#L132-L156)

**Bug** :
```python
if tp_hit and sl_hit:
    # Même barre : TP prime (conservateur, spec H03 §5.1)
    pips_net = tp_pips - cost_total        # ← prend le TP
    result_type = "win"
```

Le commentaire dit "conservateur" mais le code est **OPTIMISTE**. SL-prime = conservateur. Le simulateur stateful [`app/backtest/simulator.py:118-128`](../app/backtest/simulator.py#L118-L128) fait correctement SL-prime.

**Conséquence** : sur les bougies à grand range où TP+SL sont tous deux dans la fourchette, le `run_deterministic_backtest` compte un win au lieu d'une perte. Inflate WR et Sharpe.

Surtout impactant pour BTCUSD/ETHUSD/XAUUSD/US30 D1 (grands ranges quotidiens) avec TP=20, SL=10 (ratio 2:1 → SL plus souvent atteignable).

---

## 🟠 IMPORTANT (méthodologie fragile)

### F4 — Stacking : look-ahead par cv=5 KFold non-chronologique

**Fichier** : [app/models/candidates.py:42-59](../app/models/candidates.py#L42-L59)

```python
stacking = StackingClassifier(
    estimators=[("rf", rf), ("hgbm", hgbm)],
    final_estimator=meta,
    cv=5,                          # ← KFold default, non-shuffle mais contigus
    stack_method="predict_proba",
)
return CalibratedClassifierCV(stacking, method="isotonic", cv=3)
```

Sklearn `StackingClassifier(cv=5)` utilise `KFold(5)` par défaut. Pour générer les meta-features du fold 0, il entraîne les base estimators sur les folds 1-4 → **utilise des données du futur (folds 2-4) pour prédire le fold 0**.

C'est du look-ahead sur séries temporelles. Devrait utiliser `TimeSeriesSplit`.

**Touchés** : EURUSD D1 (Sharpe affiché 4.01), USDCHF D1 (Sharpe 3.29). Les Sharpe sont probablement gonflés en train, mais cela n'affecte pas le test set lui-même — c'est le **modèle final** qui est entraîné sur des prédictions auto-générées avec look-ahead. Le test reste OOS mais le modèle est plus optimiste que la réalité, surtout sur petits datasets.

### F5 — DSR utilise n_trials=29 alors que 44 reads OOS sont loggés

**Fichiers** :
- [scripts/run_validation_finale.py:61](../scripts/run_validation_finale.py#L61) — `N_TRIALS_CUMUL = 29`
- [TEST_SET_LOCK.json:3](../TEST_SET_LOCK.json#L3) — `"n_reads": 44`

Chaque entrée dans `read_history` est une lecture du test set (≥ 2024). Bailey & López de Prado définissent N comme le nombre de **configurations testées** (n_trials), pas le nombre de runs. Si chaque "read" correspond à une config différente, N=44 est la bonne valeur.

**Effet** : DSR diminue avec N. Avec N=29 → SR₀ ≈ 1.94. Avec N=44 → SR₀ ≈ 2.04. Pour un Sharpe observé de 4.97 et n_obs ≈ 500 jours, l'effet est modeste (~5%), mais le principe pose problème : on a sous-déflaté.

**Plus grave** : la définition même de n_trials est ambiguë. Si on inclut **toutes les configs grid-search en Phase A/C**, on dépasse facilement 1000 trials. Le DSR avec N=1000 demande SR₀ ≈ 2.8.

### F6 — Monte Carlo benchmark : US30 seul, signal_freq hardcoded

**Fichier** : [scripts/run_validation_finale.py:625-678](../scripts/run_validation_finale.py#L625-L678)

```python
def monte_carlo_random_benchmark(n_iter=1000, signal_freq=0.05, start="2024-01-01"):
    asset = "US30"             # ← un seul actif
    ...
    entry_mask = rng.random(n_bars) < signal_freq    # ← 5 % de signaux/bar
    direction = rng.choice([1, -1], size=n_bars)     # ← bernoulli 50/50
```

Problèmes :
1. Le portfolio combine 6 stratégies sur 5 actifs. Le benchmark est sur 1 actif seulement.
2. `signal_freq=0.05` ne reflète pas la fréquence réelle des stratégies (qui varient de ~daily à 1100 trades).
3. La direction random gagne le TP/SL pile-ou-face — mais le payoff TP=200/SL=100 (US30) donne un **edge structurel positif** au random (E[pips] = 0.5×200 - 0.5×100 = +50). Avec spread+slippage, l'edge diminue mais reste positif si le marché tendance assez.
4. Le bug F3 (TP-prime) **inflate le Sharpe random** : sur les bougies large-range, le random gagne le TP en priorité.

P95=9.96 est cohérent avec ce setup biaisé. **Le benchmark est plus exigeant que le portfolio simplement parce qu'il bénéficie davantage du même bug**.

### F7 — Cross-asset features : reindex(method="ffill") D1→H1 sans contrôle de convention

**Fichier** : [app/features/superset.py:340-348](../app/features/superset.py#L340-L348)

```python
df_macro = load_asset(sym, "D1")           # toujours D1
ret = np.log(df_macro["Close"] / df_macro["Close"].shift(5))
out[name] = ret.reindex(price_index, method="ffill")
```

Si l'index D1 de BTCUSD a un timestamp à `2024-03-01 00:00 UTC` représentant la **clôture** de la journée 2024-03-01 (convention end-of-day), alors :
- L'H1 du 2024-03-01 10:00 récupère via ffill la valeur à la dernière D1 timestamp ≤ 10:00 → potentiellement 2024-03-01 00:00 si MT4/MT5 met le close en début de journée suivante.
- Si la convention est start-of-day (timestamp = ouverture), pas de fuite.

**À vérifier** : la convention exacte des CSV. Sans assertion, c'est un risque latent.

### F8 — `look_ahead_safe` est un faux marqueur

**Fichier** : [app/testing/look_ahead_validator.py:46-53](../app/testing/look_ahead_validator.py#L46-L53)

```python
def look_ahead_safe(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._look_ahead_safe = True   # ← juste un drapeau, zéro validation
    return wrapper
```

Le **décorateur n'exécute aucune vérification**. Il pose un attribut. La validation arrive (peut-être) via le test générique [tests/unit/test_indicators_look_ahead.py:144-180](../tests/unit/test_indicators_look_ahead.py#L144-L180), mais ce test **skip silencieusement** sur :
- `TypeError`, `ValueError` (toute signature inhabituelle)
- `Exception` (toute erreur runtime)
- Fonctions prenant `price_index: DatetimeIndex` (economic, sessions, cross_asset → exclues)

→ Les fonctions les plus risquées (cross-asset, économique) **ne sont jamais testées**.

### F9 — ETHUSD D1 : sur-apprentissage flagrant (97.1% accuracy sur 175 trades)

Voir [`audit_final_post_mortem.md` §3.3](audit_final_post_mortem.md). Confirmé par [`predictions/phase_b_c5_b3_ethusd_d1.json`].

**HGBM** avec `max_depth=3` peut atteindre 97% sur 175 échantillons sans difficulté. Le test set OOS (328 trades) montre WR=43.9% — c'est pile au-dessus du seuil aléatoire pour TP/SL 2:1 (33.3%). Le "Sharpe +2.14" est compatible avec un modèle aléatoire qui bénéficie du bug TP-prime sur ETHUSD.

### F10 — Coûts PROVISOIRES : BTCUSD slippage = 30 USD non vérifié

Voir [`audit_final_post_mortem.md` §3.4](audit_final_post_mortem.md). Confirmé en lisant [`app/config/instruments.py:419-447`](../app/config/instruments.py#L419-L447).

Pour BTCUSD : `spread_pips=30, slippage_pips=30 → total=60 USD round-trip`. TP=2000 USD → coût = 3% du TP. Si le vrai coût XTB est 100 USD round-trip, c'est 5% — non négligeable sur 328 trades.

---

## 🟡 PROCESS / STATS (à corriger pour rigueur)

### F11 — WR=6583.9% dans v3_final_report.md

**Fichier** : [scripts/run_validation_finale.py:934](../scripts/run_validation_finale.py#L934)

```python
f"{s.get('wr', 0):.1%}"
```

`wr` est déjà en % (65.83), `{:.1%}` multiplie encore par 100 → "6583.0%". Cosmétique mais décrédibilise. Idem pour les max_dd_pct affichés à 4 chiffres.

### F12 — max_dd_pct strategy-level à -1549% (formule cassée)

**Cas observé** : EURUSD D1, `max_dd_pct = -1549.40` dans [`validation_finale.json:77`](../predictions/validation_finale.json#L77).

Le mode A1 de `compute_metrics` borne théoriquement à [-100, 0] (via `equity.clip(lower=0.01)`). La valeur -1549 suggère que :
- soit le code d'A1 n'est pas pris (les trades n'ont pas `position_size_lots`),
- soit le mode legacy est utilisé et `_pips_to_return(dd, pip_value_eur=1.0, initial=10000)` est calculé avec un pip_value erroné.

À tracer. Plusieurs chemins coexistent dans `compute_metrics`, ils ne produisent pas les mêmes échelles.

### F13 — Walk-forward inappliqué aux 4 stratégies Donchian+ML

GBPUSD D1, EURUSD D1, USDCHF D1, ETHUSD D1 utilisent un **train/test simple unique** :
- Train ≤ 2022-12-31 (12 ans)
- Test ≥ 2024-01-01 (2 ans)

Pas de re-entraînement glissant. Sur 2 ans de test, le modèle vieillit. Les régimes 2025 vs 2024 peuvent diverger, et le modèle ne s'adapte pas.

### F14 — Validation 2023 inutilisée

Constitution §3 : "train ≤ 2022, **val = 2023**, test ≥ 2024".
Réalité : `df.loc[:TRAIN_CUTOFF]` puis `df.loc[TEST_START:]` → 2023 **disparaît**.

Information perdue. 2023 aurait pu :
- Servir de validation pour le tuning (n_trials non comptés).
- Détecter le shift train→test.
- Calibrer le seuil de probabilité hors-test.

### F15 — Bootstrap Sharpe iid (non block)

**Fichier** : [app/analysis/edge_validation.py:123-160](../app/analysis/edge_validation.py#L123-L160)

```python
idx = rng.integers(0, n, size=n)            # ← échantillonnage iid avec remise
sample = clean[idx]
```

Les retours financiers sont **autocorrélés** (volatility clustering, momentum). Le bootstrap iid casse cette structure, sous-estime la variance, donne des intervalles de confiance trop étroits.

Bonne pratique : **stationary bootstrap** (Politis-Romano) ou **moving block bootstrap** avec block_size = 5–20.

### F16 — `_compute_sharpe_from_returns` sans annualisation

**Fichier** : [app/analysis/edge_validation.py:527-536](../app/analysis/edge_validation.py#L527-L536)

```python
def _compute_sharpe_from_returns(returns: np.ndarray) -> float:
    std = np.std(returns, ddof=1)
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std)    # ← pas de √252
```

Cette fonction est appelée par `validate_edge_distribution()` (ligne 661) sur `Pips_Nets` (PnL/trade). Résultat : Sharpe non annualisé **et** sur PnL/trade — double violation de la Règle 10 de la Constitution.

Visiblement code legacy, mais à supprimer pour éviter futures confusions.

### F17 — Stacking sans tuning (defaults sklearn)

Voir [`audit_final_post_mortem.md` §3.9](audit_final_post_mortem.md). Confirmé : `HYPERPARAMS_TUNED[(EURUSD, D1)]` et `[(USDCHF, D1)]` sont vides ou défaut.

### F18 — Window_hours converti via durée moyenne biaisée

**Fichier** : [app/backtest/deterministic.py:76-81](../app/backtest/deterministic.py#L76-L81)

```python
typical_td = (times[-1] - times[0]) / n
typical_hours = typical_td.total_seconds() / 3600.0
window_bars = max(1, int(window_hours / typical_hours))
```

Sur 12 ans H1 (~70k bars) : (105 120 h) / 70 000 ≈ 1.5 h/bar moyenne, à cause des weekends (gap 48h). Pour `window_hours=120`, on obtient `window_bars = 80` au lieu de 120.

Le timeout des trades est donc **30% plus court que spécifié**. Aiguille tous les Sharpe vers une fin plus brutale.

### F19 — 15 tests `.bak` dans tests/unit/

```
test_calendar_features.py.bak
test_cost_aware_labeling.py.bak
test_cpcv.py.bak
test_data_validation.py.bak
test_diagnostics.py.bak
test_evaluation.py.bak
test_macro.py.bak
test_merger.py.bak
test_prediction.py.bak
test_regressor_training.py.bak
test_sizing.py.bak
...
```

→ Coverage incomplète. Plusieurs tests d'intégrité (CPCV, cost-aware, sizing) désactivés sans justification.

---

## Conclusion

Les findings F1, F2, F3 invalident **mécaniquement** les conclusions actuelles. Avant de chercher de nouvelles stratégies, il faut :
1. **Corriger** les bugs critiques (voir [plan_v5_correction_critique.md](plan_v5_correction_critique.md)).
2. **Rejouer** la validation sur les mêmes 6 couples avec le pipeline corrigé.
3. **Décider** ensuite si l'edge est réel.

→ Le verdict NO-GO actuel est **probablement faux pour la mauvaise raison** : le portfolio ne bat pas le P95 random parce que le random est aussi biaisé par F3. Une fois le pipeline corrigé, le P95 chute, mais le Sharpe portfolio aussi.

Suite : voir [`audit_v4_strategies_viability.md`](audit_v4_strategies_viability.md), [`plan_v5_correction_critique.md`](plan_v5_correction_critique.md), [`plan_v5_amelioration_strategies.md`](plan_v5_amelioration_strategies.md).
