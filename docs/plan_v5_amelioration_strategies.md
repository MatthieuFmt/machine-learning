# Plan v5 — Amélioration et extension des stratégies

**Pré-requis** : avoir terminé les actions de [`plan_v5_correction_critique.md`](plan_v5_correction_critique.md). Tant que les bugs F1, F2, F3 ne sont pas corrigés, ajouter des stratégies = empiler des artefacts sur des artefacts.

**Cadrage utilisateur** : pas de production, juste **améliorer** les stratégies existantes et **en ajouter** d'autres. Approche exploratoire, pas de pression de timing.

---

## Trois axes complémentaires

```
Axe A — Améliorer ce qui existe        Axe B — Ajouter de nouvelles familles    Axe C — Robustifier l'évaluation
  ├─ A1. Méta-labeling fidèle           ├─ B1. Stratégies non testées            ├─ C1. Walk-forward partout
  ├─ A2. Calibration seuils             ├─ B2. Régimes adaptatifs                ├─ C2. Bootstrap block
  ├─ A3. Sizing volatility-targeting    ├─ B3. Multi-TF stacking                 ├─ C3. CPCV étendu
  ├─ A4. Filtres régime/séance          ├─ B4. Mean-reversion bands              └─ C4. Robustesse régime
  └─ A5. Ensembles entre actifs         └─ B5. Range vs trend dispatch
```

---

## Axe A — Améliorer les 6 stratégies existantes (après correction)

### A1. Méta-labeling fidèle (vraie Option A de Phase 1)

Une fois F1 corrigé, le méta-labeling doit **filtrer les signaux Donchian** sans en générer de nouveaux. Travail à faire **par couple** :

- [ ] Refaire le tuning du seuil de probabilité sur **2023** (val set), pas sur train.
- [ ] Comparer 3 variantes :
  - Donchian seul (baseline, sans ML).
  - Donchian + RF méta-labeling.
  - Donchian + HGBM méta-labeling.
- [ ] Si Donchian seul ≥ Donchian+ML → le ML n'apporte rien sur ce couple. Documenter et passer.

**Indicateur de succès** : alpha vs Donchian seul ≥ 0.5 en Sharpe, p<0.05 par bootstrap.

### A2. Calibration des seuils sur validation 2023

Les seuils actuels (`HYPERPARAMS_TUNED[key]["threshold"]`) ont été calibrés sur le train via CPCV, ce qui consume du n_trials. Calibration alternative :
1. Train ≤ 2022 : entraînement modèle.
2. Val = 2023 : balayage de threshold ∈ [0.45, 0.70] par pas de 0.01, retenir le seuil qui maximise Sharpe sur 2023.
3. Test ≥ 2024 : un seul run, ne pas modifier.

Bonus : appliquer une **calibration Platt** ou **isotonique** sur 2023 pour rendre les probabilités vraiment probabilistes.

### A3. Sizing volatility-targeting

Le sizing actuel = risque fixe 2 % sur SL → équivaut à une **équipondération en pips**, pas en risque réel.

Volatility-targeting : taille de position inversement proportionnelle à l'ATR récent. Mathématiquement :
```
risk_eur = capital × risk_pct
target_vol_daily = capital × 0.005     # 0.5 %/jour
lots = target_vol_daily / (atr_pips × pip_value_eur)
```

Effets attendus :
- Réduit le sizing en régime volatil (drawdowns plus petits).
- Stabilise le Sharpe.
- Casse la sensibilité à F2 (puisque sizing varie avec marché → Sharpe pct_change devient cohérent).

À tester sur les 6 couples. Hypothèse à logger.

### A4. Filtres régime et séance

Les `regime_features` existent ([`app/features/superset.py:222-248`](../app/features/superset.py#L222-L248)) mais ne sont pas utilisées comme **filtre actif**. Idée :
- N'autoriser les signaux Donchian que si `trend_strength = 1` (ADX > 25) → réduit les faux breakouts en range.
- N'autoriser les signaux H4/H1 que durant **London + NY overlap** (13:00-17:00 UTC) → meilleur liquidité.

Approche : implémenter un `FilterPipeline` configurable et tester deux ou trois combinaisons par couple. Compter chaque variante comme un n_trial.

### A5. Ensembles entre actifs

Les 4 couples D1 (GBPUSD, EURUSD, USDCHF, ETHUSD) sont **fortement corrélés** sur la dimension USD. Tester :
- **Risk parity** (allocation inverse vol) plutôt qu'equal-weight.
- **Max-correlation cap** : exclure une stratégie si sa corrélation avec une autre dépasse 0.7.
- **Hedging USD** : si on a `long GBPUSD + long EURUSD + short USDCHF`, on a 3× exposition USD short. Vérifier que l'agrégat n'est pas un pari USD déguisé.

---

## Axe B — Nouvelles familles de stratégies (après A1-A4)

### B1. Stratégies non testées

Le dossier [`app/strategies/`](../app/strategies/) contient déjà des **squelettes** non backtestés :
- `bollinger.py`
- `chandelier.py`
- `dual_ma.py`
- `keltner.py`
- `parabolic.py`
- `rsi_contrarian.py`
- `sma_crossover.py`
- `ts_momentum.py`

Les seules testées en profondeur : `donchian.py`, `mean_reversion.py`.

**Plan** : lancer le **même protocole** (train ≤ 2022, val 2023, test ≥ 2024) sur les 8 stratégies non-testées × 6 actifs × 3 TF = 144 combinaisons. **Grid search déterministe**, pas de ML.

Garde-fou : déclarer en avance les **3 stratégies les plus prometteuses théoriquement** (ex : Keltner, ts_momentum, RSI contrarian) et ne lire que celles-là en OOS — sinon n_trials explose.

### B2. Régimes adaptatifs (régime detector)

Idée : entraîner un classifieur "régime" (trend vs range vs vol high) sur des features non-financières (ATR%, ADX, return autocorrelation), puis :
- Régime "trend" → activer Donchian/dual_ma.
- Régime "range" → activer Bollinger/RSI contrarian.
- Régime "vol high" → no trade.

Référence dans le repo : `prompts/10_h09_regime_detector.md` (jamais implémenté).

### B3. Multi-TF stacking

Le superset cross-asset est D1 only ([`app/features/superset.py:330`](../app/features/superset.py#L330)). Pour une stratégie H1 :
- Empiler les features H1, H4, D1 du même actif.
- Le merge_asof doit être **STRICTEMENT** anti-leak (shift de 1 bar TF supérieur).
- Tester si la convergence multi-TF améliore le edge.

Référence : `prompts/16_h15_tf_decision.md`, `prompts/17_h16_timeframe_stacking.md`.

### B4. Mean-reversion bands (extension EURUSD H4)

EURUSD H4 mean-reversion a 54 trades OOS — trop peu. Variantes à tester :
- Bandes plus larges (BB 2.5 au lieu de 2.0) → moins de signaux mais plus de conviction.
- RSI 30/70 → 20/80 (extrêmes plus rares).
- Filtrer par régime "range" (B2).
- Étendre à USDCHF H4, AUDUSD H4, NZDUSD H4 (autres mean-reverting).

### B5. Dispatch range vs trend par actif

Hypothèse : chaque actif a une **personnalité** :
- US30, BTCUSD, ETHUSD : trend (momentum).
- EURUSD H4, USDCHF H4, USDJPY H4 : range (mean-reversion).
- GBPUSD : mixte selon la session.

Au lieu d'appliquer Donchian à tous, **assigner la bonne famille à chaque couple** via un test exploratoire sur train (≤ 2022).

---

## Axe C — Robustifier l'évaluation

### C1. Walk-forward systématique (F13 corrigée)

Pour **tous** les couples retenus, remplacer le train/test simple par :
- Train initial : 2010-2020 (10 ans).
- Re-entraînement annuel.
- Test = chaque année 2021, 2022, 2023, 2024, 2025.

→ 5 Sharpe par couple, écart-type → mesure de stabilité. Si Sharpe varie de 4.0 à -1.0 selon l'année, c'est un signal fort de fragilité.

### C2. Bootstrap block (F15 corrigée)

Sur les retours quotidiens du portfolio, appliquer un **stationary bootstrap** (block size moyen = 10 jours) pour calculer :
- IC95% sur Sharpe.
- p-value de `Sharpe > 0`.
- p-value de `Sharpe > 1.0`.

Si l'IC95% inclut 0, l'edge n'est pas significatif statistiquement.

### C3. CPCV étendu sur plusieurs scénarios

Ajouter des **scénarios stress** au CPCV :
- Coûts +20 % (vs valeurs provisoires).
- Slippage +50 %.
- Régime "bear" (extraire les sous-périodes 2020-Q1, 2022-Q3 → re-tester).
- Régime "bull" (2021, 2024).

Si Sharpe < 0 dans tout scénario stress → fragile.

### C4. Robustesse aux régimes

Découper le test 2024-2026 en sous-périodes (par trimestre) et calculer :
- Sharpe par trimestre.
- Si > 50 % des trimestres ont Sharpe < 0 → modèle pas robuste, même si Sharpe global > 0.

---

## Roadmap proposée (séquentielle, exploratoire)

| Étape | Contenu | Pré-requis | Effort |
|---|---|---|---|
| 0 | Corrections critiques (Phase 1-3 de [plan_v5_correction_critique](plan_v5_correction_critique.md)) | — | 5-7 j |
| 1 | A1 méta-labeling fidèle, recalibrer sur 2023 (A2) | 0 | 2-3 j |
| 2 | Rejeu validation finale corrigée → verdict réaliste | 1 | 1 j |
| 3 | A3 vol-targeting + A4 filtres régime | 2 | 2-3 j |
| 4 | A5 portfolio risk-parity + diagnostics corrélations | 3 | 1 j |
| 5 | B1 nouvelles familles (3 stratégies max) | 4 | 3-5 j |
| 6 | B2 régime detector + B5 dispatch | 5 | 4-7 j |
| 7 | C1 walk-forward + C2 bootstrap block + C4 régimes | 6 | 2-3 j |
| 8 | (Optionnel) B3 multi-TF stacking | 7 | 5-7 j |

**Pas de contrainte de timing**. Exploratoire.

---

## Ce qu'on n'ajoutera PAS (volontairement)

1. **LightGBM / XGBoost** : la roadmap original Step 03 proposait LightGBM+Optuna. Tant que HGBM (équivalent sklearn) suffit, pas besoin d'ajouter un dependency lourd. À reconsidérer si A3-A5 ne suffisent pas.
2. **Deep learning / RNN** : aucun intérêt sur le volume de données disponible (~few thousand trades). Sur-paramétré.
3. **Live trading / API broker** : hors scope confirmé par utilisateur.
4. **Telegram alerts / scheduler / VPS** (prompts 21, 22, 23) : production = exclu.

---

## Garde-fous méthodologiques pour la suite

À respecter pour chaque nouvelle hypothèse :

1. **Une lecture OOS par hypothèse**. Compter dans le snooping_guard.
2. **Documenter dans JOURNAL.md** avant d'exécuter (nom, hypothèse, résultat attendu).
3. **Calibrer sur 2023**, pas sur 2024+.
4. **Bootstrap block** sur tous les Sharpe affichés.
5. **Comparer au benchmark naïf** : Donchian seul, B&H, random Monte Carlo (post-F3 fix).
6. **Ne jamais ajuster** un seuil/feature en réaction à un résultat OOS — c'est un nouveau n_trial.

---

## Indicateurs de succès à long terme

Le projet sera "validé" (au sens scientifique, pas production) si **un seul** des éléments suivants est vrai après les corrections :

- 🎯 Une stratégie unique avec Sharpe ≥ 1.0, DSR > 0 (p<0.05 avec N=50+), max DD < 15 %, ≥ 30 trades/an, robuste sur 4 sous-périodes test.
- 🎯 Un portfolio de 3+ stratégies décorrélées (corr < 0.5) avec Sharpe portefeuille ≥ 1.5.
- 🎯 Une famille de stratégies qui montre une consistance (Sharpe > 0 sur tous les actifs d'une classe).

Si **aucun** des trois après tout le plan v5 → conclusion honnête : pas d'edge détectable avec les méthodes courantes ; soit explorer des approches très différentes (HFT, sentiment, options), soit acter la fin du projet.

C'est un résultat valide. Ne pas confondre "j'ai trouvé un Sharpe 5" et "j'ai un edge".
