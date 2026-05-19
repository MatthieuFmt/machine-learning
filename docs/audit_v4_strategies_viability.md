# Audit v4 — Viabilité réelle des 6 stratégies "GO"

**Question centrale** : les 6 stratégies validées individuellement sont-elles des **vrais edges** ou des **artefacts de pipeline** ?

Référence : [`audit_v4_findings.md`](audit_v4_findings.md) pour les bugs sous-jacents.

---

## Synthèse rapide

| Couple | Sharpe affiché | Sharpe vraisemblable | Verdict | Cause |
|---|---|---|---|---|
| GBPUSD D1 (rf) | 5.17 | ~1.5 | 🟡 Trend-following gated, viabilité non prouvée | F1+F2 |
| EURUSD D1 (stacking) | 4.01 | ~1.0–1.5 | 🔴 Look-ahead Stacking + F1+F2 | F1+F2+F4 |
| USDCHF D1 (stacking) | 3.29 | ~0.5–1.0 | 🔴 Idem EURUSD | F1+F2+F4 |
| ETHUSD D1 (hgbm) | 2.14 | ~0 ou négatif | 🔴 97% acc train sur 175 trades → mémorisation | F1+F2+F9 |
| GBPUSD H4 (rf, WF) | 3.19 | ~1.0–1.5 | 🟡 Le seul vrai walk-forward, mais 1103 trades → coûts dominants si spread réel > provisoire | F2+F10 |
| EURUSD H4 (rf, mean-rev WF) | 1.73 | ~0.5–1.0 | 🟡 Seulement 54 trades OOS — variance énorme | F2 + petit échantillon |

**Aucune des 6 ne tient comme "edge" prouvé statistiquement** une fois les bugs F1, F2, F3 corrigés.

---

## Analyse détaillée par couple

### GBPUSD D1 — Sharpe 5.17 (rf)

**Train** : 597 trades Donchian (2010-2022, ~12 ans).
**Test** : 483 trades (2024-2026, ~2 ans) → **~240 trades/an**, totalement incompatible avec Donchian D1 (qui produit ~50/an).

**Ce qui se passe vraiment** :
- Le RF entraîne `P(win) = f(features at Donchian breakout)`.
- En test, on l'évalue sur **toutes** les barres, et la direction vient de `sign(mean(slope_sma_20, slope_sma_50, dist_sma_200))`.
- Sur 2024-2026, GBP a une tendance USD très claire → trend-following ramasse de l'alpha mécaniquement.
- Le RF avec `class_weight=balanced` produit des proba ~0.5±0.1 → seuil 0.5 filtre peu.
- Résultat : presque tous les jours, on prend la direction du trend.

**Le "ML" n'ajoute presque rien**. La stratégie est : "long quand SMAs montent, short quand elles baissent, sur D1".

**Test naturel à faire** : refaire le backtest **sans le RF** (juste trend_sign), comparer le Sharpe. Si identique, le ML n'apporte rien.

### EURUSD D1 — Sharpe 4.01 (stacking)

Mêmes problèmes que GBPUSD D1 (F1, F2) + **F4 (look-ahead Stacking)**.

- 610 train trades, 630 test trades = ~315/an. Pas du Donchian D1.
- Stacking avec `cv=5` KFold → meta-features générés avec fuite d'information du futur.
- Accuracy train 68% — pas suspecte en soi, mais le test repose sur le pipeline cassé.

`max_dd_pct = -1549%` affiché → numériquement impossible avec mode A1, suggère un bug additionnel (F12).

### USDCHF D1 — Sharpe 3.29 (stacking)

- 585 train trades, 423 test trades = ~210/an.
- Stacking idem EURUSD (F4).
- Accuracy train **86.9%** sur 585 trades → suspect, suggère mémorisation modérée.
- USDCHF a moins de tendance pure que GBPUSD/EURUSD en 2024-2026 (range-bound) → 423 trades sur D1 est étrange.

### ETHUSD D1 — Sharpe 2.14 (hgbm)

Voir F9. Le plus exposé au pur overfitting :
- 175 trades train, 97.1% accuracy → 5 features de profondeur 3 suffisent à mémoriser.
- 328 test trades, WR=43.9% (à comparer au 33.3% aléatoire pour ratio TP/SL 2:1).
- Sharpe 2.14 cohérent avec un edge marginal + F3 (TP-prime).
- Coûts provisoires (F10) très sensibles : si vrai spread = 5 USD au lieu de 3 USD, le PnL change sensiblement.

### GBPUSD H4 — Sharpe 3.19 (rf, walk-forward méta-labeling)

**Seul cas avec vrai walk-forward** (re-train tous les 6 mois). Plus crédible méthodologiquement, mais :
- 19 282 échantillons "n_train_total" → train cumulé sur tous les segments WF.
- 1 103 trades OOS sur 2 ans = ~550/an = ~quotidien sur H4 (≈ 4 barres/jour).
- Le pipeline méta-labeling est appliqué sur des **signaux primaires bootstrap** (quintile top/bottom de `trend_score`), pas Donchian → encore une fois, pas du Donchian.
- Le méta-labeling RF filtre, mais le générateur primaire est un trend-follower déguisé.
- F2 (Sharpe inflation) s'applique : 1103 trades → tpy >> 100 → mode daily compoundé → Sharpe gonflé.

### EURUSD H4 — Sharpe 1.73 (rf, mean-rev WF)

- Seulement **54 trades OOS** sur 2 ans → variance ÉNORME.
- Stratégie mean-reversion RSI+BB avec méta-labeling RF.
- Le Sharpe 1.73 sur 54 trades a un IC95% probablement [0, 3] — incertitude massive.
- Vraisemblablement le moins corrompu par F1 (mean-reversion est une vraie stratégie discriminante, pas un alias de trend-following). Mais le walk-forward méta-labeling souffre de F2.

---

## Test de stress mental : que reste-t-il si on enlève F1+F2+F3 ?

**Hypothèse correctionnelle** :
1. F1 : remplacer `_generate_model_signals` par "appliquer le RF UNIQUEMENT sur les barres Donchian primaires" → le test n'a plus que ~10× moins de trades.
2. F2 : Sharpe sur returns linéaires (`diff(equity)/initial_capital`) → diviser par ~3.5.
3. F3 : SL-prime sur same-bar conflict → WR baisse de quelques points.

Estimation grossière :
- GBPUSD D1 : 50 trades OOS, Sharpe ~1.0, p>0.05 → NON significatif.
- EURUSD D1 : 80 trades OOS, Sharpe ~0.8, p>0.05 → NON significatif.
- USDCHF D1 : 60 trades OOS, Sharpe ~0.5 → NON significatif.
- ETHUSD D1 : 50 trades OOS, Sharpe ~0 → NON significatif.
- GBPUSD H4 : 200 trades OOS, Sharpe ~0.9 (selon vrais coûts) → marginal.
- EURUSD H4 : 30 trades OOS, Sharpe ~0.8 → IC trop large.

→ Le portfolio combiné aurait probablement Sharpe ~1.0–1.5, **DSR pas significatif après correction n_trials=44+**.

**Verdict** : il est plausible qu'**aucun edge réel n'existe** dans le pipeline actuel. Mais aussi possible que **un ou deux edges modestes** émergent après correction. Impossible à trancher sans rejouer.

---

## Ce que les bugs cachaient

Ironie : le pipeline a **bénéficié** d'erreurs qui se sont compensées :
- F1 (test signals != train signals) augmente les signaux → augmente Sharpe en bull market.
- F2 (Sharpe compound) gonfle mécaniquement le Sharpe.
- F3 (TP-prime) gonfle le WR.
- **Mais** F3 gonfle aussi le Monte Carlo P95 random → benchmark exigeant.

→ Le verdict NO-GO actuel est **un faux négatif** : le portfolio est gonflé, mais le benchmark l'est encore plus. Une fois corrigé, **le portfolio ET le benchmark baissent ensemble** ; on ne peut pas prédire le rapport.

---

## Recommandations de prudence

1. **Ne pas conclure** "ça marchait avant qu'on découvre les bugs". Le Sharpe 4.97 n'est pas un edge ; c'est un mirage numérique.
2. **Ne pas réinvestir** dans la même structure (Donchian+méta-labeling sur les mêmes 5 actifs) sans correction préalable.
3. **Considérer que la v2** (US30 D1 Donchian + méta-labeling, Sharpe 8.84 H05) est **probablement aussi** affectée par F2 et F3. Le résultat fondateur du projet est à re-vérifier.
4. **Ne pas faire confiance aux chiffres dans v3_final_report.md** (WR 6583%, max_dd -1549%). Le rapport est cosmétiquement cassé.

Suite : [`plan_v5_correction_critique.md`](plan_v5_correction_critique.md) — les actions concrètes pour fiabiliser.
