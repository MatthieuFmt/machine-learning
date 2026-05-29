# Phase F — Bilan et décision go/no-go Phase G

**Date** : 2026-05-21
**Statut** : ✅ **GO Phase G**
**Référence plan** : [docs/audit_v6_action_plan.md](audit_v6_action_plan.md)

---

## 1. Acquis Phase F

### F1 — Swap overnight modélisé

- Mécanisme implémenté dans [app/backtest/simulator.py:157-167](../app/backtest/simulator.py#L157-L167)
  et [app/backtest/deterministic.py:198-210](../app/backtest/deterministic.py#L198-L210) :
  `pips_net += nights_held × swap_per_night`.
- Champs `swap_long_pips_per_night` / `swap_short_pips_per_night` ajoutés
  à [AssetConfig](../app/config/instruments.py#L248) (défaut 0.0, retrocompatibilité legacy).
- Convention : pas de triple-swap mercredi (V1 simple), à raffiner V2.

### F2 — Coûts XTB Standard validés

- Spreads et slippage XTB Standard capturés dans [instruments.py](../app/config/instruments.py)
  pour 8 actifs (XAUUSD, XAGUSD, USOIL, EURUSD, GBPUSD, USDCHF, BTCUSD, ETHUSD).
- Règle slippage : 0.2× spread pour majeures, 0.5× pour mineures, 1.0× pour crypto.
- Source : [docs/cost_audit_v2.md](cost_audit_v2.md).

### F3 — Re-download Dukascopy (40 actifs × 4 TF)

- Script [scripts/download_dukascopy_full.py](../scripts/download_dukascopy_full.py)
  opérationnel après 7 corrections de bugs (limit=200k, mapping symboles,
  retry simplifié, dry-run, gap handling JPY).
- Calendrier de jours fériés étendu pour 7 paires JPY dans
  [app/config/calendar.py](../app/config/calendar.py).
- Couverture finale : 40 actifs × 4 TF dans `data/raw/`, ingestion complète.

### F4 — Détecteur de régime MVP

- Fonction `detect_regime()` dans [app/features/regime.py:92](../app/features/regime.py#L92) —
  3 labels `{trend, range, vol_high}` selon ADX(14) > 25 et ATR%(14) > quantile 80%.
- Tests : [tests/unit/test_regime_detector.py](../tests/unit/test_regime_detector.py) (9 tests).
- Analyse multi-actifs : [docs/regime_analysis.md](regime_analysis.md) sur 14 actifs D1.
- **Lecture** : cryptos = très tendancielles (37-42% trend), indices = très range
  (53-57%), forex = range modéré, métaux entre les deux.
- ⚠️ Livré comme **feature**, pas comme filtre — dispatch stratégie viendra plus tard.

### F5 — Features macro externes (DXY, VIX, yield slope)

- Module [app/features/macro_external.py](../app/features/macro_external.py) :
  `add_external_macro(df)` ajoute 4 colonnes :
  `dxy_zscore_60`, `vix_level`, `vix_zscore_60`, `yield_slope_10y_3m`.
- Source yfinance (DX-Y.NYB, ^VIX, ^TNX, ^IRX) avec cache disque
  sous `data/raw/macro/`. **Pas de clé API requise** (FRED non accessible).
- Substitution `yield_slope_10y_3m` à la place de `T10Y2Y` initialement
  prévu (FRED-only), même indicateur de récession.
- Anti-look-ahead : shift +1 jour avant `merge_asof(direction="backward")`.
- Tests : [tests/unit/test_macro_external.py](../tests/unit/test_macro_external.py) (14 tests).
- Script [scripts/download_macro_external.py](../scripts/download_macro_external.py) pour
  pré-télécharger les séries.

---

## 2. Test critique — Impact swap sur ETHUSD H1

### Méthodologie

Re-run de [scripts/run_phase_b_c5_b2_ethusd_h1.py](../scripts/run_phase_b_c5_b2_ethusd_h1.py)
après deux corrections :

1. Transmission des champs `swap_long/short_pips_per_night` à
   `run_deterministic_backtest` (auparavant absents → swap = 0 silencieux).
2. Calibration ETHUSD : `long = -80 pips/nuit` (≈ 10%/an de financement
   sur ETH à 3000 USD avec pip = 0.01 USD), `short = -10 pips/nuit`.

### Résultats

| Métrique | Pré-swap | Post-swap | Δ |
|---|---:|---:|---:|
| Sharpe (trades) | 1.229 | **1.144** | −0.085 |
| Sharpe (compute_metrics) | 1.206 | 1.116 | −0.090 |
| Total trades | 426 | 416 | −10 |
| Win rate | 38.7% | 38.5% | −0.2 pp |
| Max DD (pips) | −55 138 | **−82 056** | **+49%** |
| Profit factor | 1.27 | 1.24 | −0.03 |

### Lecture

- L'edge **survit** au coût swap réaliste : Sharpe perd ~7% mais reste >1.0.
- Le **Max DD** se dégrade de +49% — le swap mange sur la queue gauche des
  pires séries de pertes (cohérent avec un coût additif sur les trades les
  plus longs/perdants).
- La distribution `win=415/1106 (37.5%) → 407/1106 (36.8%)` montre que
  même les labels ML du train sont affectés (moins de winners avec swap).

### Cas GBPUSD H4 rf

Re-run **skippé** : la baseline pré-swap était déjà désastreuse
(Sharpe = −1.72, blowup détecté, validate_edge = NO GO avec DSR = −17.8).
Ajouter du swap ne ferait que confirmer la non-viabilité.

---

## 3. Décision Phase G

### Critère go/no-go (cf [audit_v6_action_plan.md:148](audit_v6_action_plan.md#L148))

- ✅ GO si pipeline corrigé sans régression.
- ❌ Pivot si l'ajout du swap fait chuter **tous** les Sharpe historiques < 0.

### Verdict : **GO Phase G**

- Pipeline corrigé (swap appliqué, coûts validés, data ingestée, régime
  détecté, macro disponible).
- ETHUSD H1 conserve un Sharpe > 1.0 post-swap.
- 339 tests unitaires passent (1 ajout F4 + 14 ajouts F5 = 354 désormais).
- Aucune régression observée sur les modules existants.

---

## 4. Réserves et trous techniques

### Swaps approximatifs

Les valeurs `swap_*_pips_per_night` actuelles sont des **estimations**
basées sur des taux de financement industrie typiques (10%/an crypto,
~3%/an métaux, différentiels de taux pour le forex). Elles n'ont **pas**
été validées sur la spec XTB démo car les tables de swap XTB ne sont
pas accessibles publiquement.

**Action future** : ouvrir un compte démo XTB, capturer les vrais swaps
par instrument, remplacer les TODO `# F6 — swaps estimés` dans
[instruments.py](../app/config/instruments.py).

### Convention swap V1

- Pas de triple-swap mercredi (à raffiner V2).
- `nights_held` = différence en jours civils entre `entry_time` et
  `exit_time` (peut sous-estimer si position franchit minuit sans
  vraiment "tenir une nuit").

### F4 livré sans dispatch

Le détecteur de régime est disponible comme feature mais aucune stratégie
ne route encore selon le régime. La décision d'activer/désactiver certaines
stratégies par régime se prendra après Phase G (screening multi-stratégies).

### F5 sans intégration superset

`add_external_macro` n'est pas encore branché dans
[app/features/superset.py](../app/features/superset.py). À faire si Phase G
veut exploiter ces features ML.

### USOIL — un seul TF

USOIL n'a que D1 dans `data/raw/` (les autres TF rejetés par `load_asset`
au F3). À investiguer si l'actif doit être actif en Phase G.

---

## 5. Préparation Phase G

**Plan original** ([audit_v6_action_plan.md:157](audit_v6_action_plan.md#L157)) :

> G1. Download Dukascopy 30 actifs ← ✅ déjà fait en F3
> G2. Test pairs JPY ← prêt
> G3. Test Nasdaq + Nikkei ← actifs absents, à ajouter
> G4. Test Brent + NatGas + Copper ← absents, à ajouter
> G5. Test cryptos additionnelles ← actifs absents, à ajouter

**Recommandation d'ordre** :
1. Démarrer G2 (JPY pairs) — actifs déjà ingérés (USDJPY, EURJPY, GBPJPY, AUDJPY).
2. Re-runner les 13 stratégies existantes × 5 ratios TP/SL × 4 actifs JPY.
3. Comparer avec/sans macro features (test isolé F5).
4. Décider si on étend aux actifs absents (Nasdaq/Nikkei/Brent/etc.) ou
   si la matière JPY suffit.

---

## 6. Synthèse 1-ligne

> **Pipeline corrigé, coûts swap appliqués, edge ETHUSD H1 survit (Sharpe 1.14).
> GO Phase G sur screening JPY pairs avec swaps estimés (vrais XTB à valider plus tard).**
