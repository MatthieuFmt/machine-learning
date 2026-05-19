# Diagnostic final — Donchian D1 est mort. Pourquoi, et quoi faire ensuite.

**Date** : 2026-05-19
**Statut projet** : pivot stratégique nécessaire — Donchian D1 disqualifié sur tous actifs

---

## Résumé en une page

Le projet a été bâti pendant des mois sur la prémisse que **Donchian Breakout D1 sur US30 a un edge** (résultat H03 v2 : Sharpe walk-forward +3.07 à +8.84). Les Phases A, C, B du pivot v4 ont étendu cette base à 11 couples, retenu 6 stratégies "GO", et abouti à un portfolio affiché Sharpe +4.97.

L'audit v5 a découvert :

1. **3 bugs bloqueurs** dans le pipeline (F1 distribution train/test rompue, F2 Sharpe gonflé ×3.5, F3 simulateur optimiste TP-prime) → tous corrigés Phase 1.
2. Après correction, **portfolio Sharpe = -5.42** (vs +4.97 affiché). Verdict honnête : pas d'edge.
3. **Le diagnostic post-fix** révèle que ce n'est pas le ML qui inverse l'edge (le ML est neutre voire utile sur ETHUSD) — c'est **Donchian D1 lui-même qui n'a pas d'edge**.
4. **Le grid TP/SL ATR-based** sur 9 couples × 4 ratios = 36 backtests sur train 2010-2022 montre : **aucun couple, aucun ratio ne produit Sharpe > 0.5 sur train**.

Le résultat originel (H03 v2, US30 D1 Sharpe +3.07) était presque certainement un artefact du bug F3 TP-prime. Avec le simulateur corrigé, US30 D1 train donne **Sharpe -0.66 baseline, 0.00 au meilleur ratio**.

→ **Conclusion** : Donchian D1 est disqualifié. Le projet doit pivoter vers d'autres familles de stratégies.

---

## Chronologie de la découverte (chemin d'investigation)

```
État affiché avant audit
  Portfolio Sharpe : +4.97, DSR 19.5, WR 54%, NO-GO (P95 random 9.96)
              ↓ Audit code (find F1+F2+F3)
Phase 1 correction (bloqueurs)
              ↓ Re-run validation_finale
État réel post-fix
  Portfolio Sharpe : -5.42, DSR -18.3, WR 22.5%, NO-GO
              ↓ Diagnostic 1 (scripts/diagnose_ml_inverse_edge.py)
Le ML n'est PAS le coupable
  - Acceptés WR ≈ Rejetés WR sur 3 couples (pas d'inversion)
  - ETHUSD : le ML AMÉLIORE (acceptés WR 37.5% vs rejetés 25.4%)
              ↓ Diagnostic 2 (scripts/diagnose_donchian_tp_sl.py)
La config TP/SL est inadaptée
  - SL=10 pips = 10% de l'ATR D1 GBPUSD (102 pips)
  - 87% loss_sl sur train → mécaniquement perdant
  - Le SL était trop serré sur 4/9 couples
              ↓ Diagnostic 3 (scripts/diagnose_donchian_atr_grid.py)
Aucun ratio TP/SL ne sauve la stratégie
  - 9 couples × 4 ratios = 36 backtests train
  - Meilleur Sharpe atteint = +0.42 (ETHUSD SL=1.5×ATR)
  - Aucun couple ne passe le critère 0.5 sur train
```

---

## Ce qui restera utile (les acquis)

Même si Donchian est disqualifié, l'audit v5 laisse un **pipeline fiable** :

| Composant | Statut |
|---|---|
| Simulateur déterministe (TP/SL/timeout, SL-prime) | ✅ Correct (fix F3) |
| Sharpe annualisé linéaire (capital fixe) | ✅ Correct (fix F2) |
| Méta-labeling pipeline (Option A López de Prado) | ✅ Correct (fix F1) |
| Stacking sans look-ahead (TimeSeriesHoldoutStacking) | ✅ Correct (fix F4) |
| Bootstrap stationnaire (block, Politis-Romano) | ✅ Correct (fix F15) |
| Cross-asset features shift(1) | ✅ Correct (fix F7) |
| n_trials aligné avec read_history | ✅ Correct (fix F5) |
| Monte Carlo multi-asset représentatif | ✅ Correct (fix F6) |
| Validation 2023 disponible (opt-in) | ✅ Disponible (fix F14) |
| 49 tests neufs anti-régression | ✅ En place |

→ Le pipeline peut maintenant tester n'importe quelle nouvelle stratégie **sans biais**.

---

## Pourquoi Donchian D1 ne marche pas

Trois raisons cumulées :

1. **Le breakout du high N-bars n'a pas d'edge directionnel sur D1**. Les marchés ne sont pas systématiquement trending ou mean-reverting à cette échelle — ils alternent.

2. **Le ratio TP/SL=2:1 est seulement viable si WR > 33%**. Sur Donchian D1, le WR plafonne à 47% à SL=1.5×ATR (le random pile-ou-face). Mais à 47% WR, on est à peine au-dessus du breakeven 33% × 1.5 (TP) - 67% × 1 (SL) - costs ≈ 0.05 - costs. Les frictions absorbent l'edge mathématique.

3. **Les frictions sur D1 sont proportionnellement plus impactantes** que sur H1/H4 quand on regarde le ratio cost/TP. Mais elles sont quand même non-négligeables.

---

## Roadmap proposée — Que tester ensuite ?

Plan : explorer les 8 stratégies déjà présentes dans `app/strategies/` mais **jamais évaluées** sur le pipeline corrigé. Toutes ont des squelettes ; aucune n'a été testée.

```
app/strategies/
├── donchian.py       ❌ DISQUALIFIÉ
├── mean_reversion.py ⚠️  Testé partiellement (EURUSD H4)
├── bollinger.py      ⬜ À tester
├── chandelier.py     ⬜ À tester (test_chandelier en échec → bug à fixer d'abord)
├── dual_ma.py        ⬜ À tester
├── keltner.py        ⬜ À tester
├── parabolic.py      ⬜ À tester
├── rsi_contrarian.py ⬜ À tester
├── sma_crossover.py  ⬜ À tester
└── ts_momentum.py    ⬜ À tester
```

### Protocole de screening proposé

Pour chaque stratégie × chaque actif × chaque ratio TP/SL ATR-based :
1. Calcul sur **train ≤ 2022 uniquement** (0 n_trial consommé).
2. Critères de pré-sélection (sur train) :
   - Sharpe ≥ 0.8 (marge vs le critère final 1.0)
   - WR ≥ 35%
   - n_trades ≥ 60 (pour pouvoir mesurer)
   - max_dd ≤ 25%
3. Si une stratégie × couple × ratio passe → candidate pour test OOS unique.

Voir script [scripts/screen_strategies_train.py](../scripts/screen_strategies_train.py).

### Critères d'arrêt du projet

Si, après le screening complet :
- **Aucune** stratégie × couple ne montre Sharpe ≥ 0.8 sur train → admettre que les approches techniques simples ne marchent pas sur D1 (pour ces 12 ans / 9 actifs). Pivot vers : autres timeframes (H1/H4), autres types de stratégies (calendar, sentiment, options), ou fin honnête du projet.
- **Une ou plusieurs** stratégies passent → tester en OOS unique, comme une nouvelle hypothèse H_v5_01.

---

## Recommandation immédiate

1. Lancer [scripts/screen_strategies_train.py](../scripts/screen_strategies_train.py) (à créer dans cette session).
2. Analyser la sortie : **est-ce qu'une stratégie alternative montre un signal sur train, ou est-ce que tout est plat ?**
3. Décider :
   - **Si signal** : implémenter le pipeline méta-labeling complet sur la stratégie gagnante, refaire validation finale en OOS unique.
   - **Si rien** : passer à des approches qualitativement différentes (régime detector, multi-TF stacking, ou pivot fondamental).

Le pipeline est prêt. La question est : **existe-t-il un edge technique simple détectable sur ce dataset ?** Le screening répondra.
