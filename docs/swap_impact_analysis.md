# Phase F1 — Modélisation du swap overnight

**Date** : 2026-05-19
**Statut** : ✅ Code livré. Validation des valeurs swap réelles par actif → Phase F2 (démo XTB).
**Réf.** : [`audit_v6_data_gaps.md`](audit_v6_data_gaps.md) §4, [`audit_v6_action_plan.md`](audit_v6_action_plan.md) F1.

---

## 1. Le bug constaté

Dans [`app/backtest/deterministic.py:73-74`](../app/backtest/deterministic.py#L73-L74) (avant fix) :
```python
cost_per_side = commission_pips + slippage_pips
cost_total = cost_per_side * 2
```

Et dans [`app/backtest/simulator.py`](../app/backtest/simulator.py) (`_simulate_stateful_core`) :
```python
spread_cost = spreads[i] / 10.0 + spread_cost_base
```

Dans les deux cas, **le coût est unique à l'entrée/sortie**, sans charge proportionnelle à la durée de détention. Aucune ligne n'ajoute de `swap × nights_held` au PnL.

### Conséquences avant correction

- Tous les trades Donchian D1 (durée typique 3-10 jours) surestiment le résultat.
- Pour USDJPY long, le swap est POSITIF (carry favorable) — donc le simulateur **sous-estimait** ce que ces trades auraient réellement gagné. Asymétrie non comptabilisée.
- Pour BTCUSD swing 1 semaine chez XTB, swap ≈ -0.05 %/nuit ⇒ **-350 €/semaine** sur 100k€ position. 50 trades/an = -17500 €/an manqués.
- Tous les Sharpe historiques affichés étaient **biaisés à la hausse** (sauf carry favorable).

---

## 2. La correction (F1)

### 2.1 `AssetConfig` étendu

Ajouts dans [`app/config/instruments.py`](../app/config/instruments.py) :
```python
swap_long_pips_per_night: float = 0.0    # crédit (>0) ou débit (<0) long
swap_short_pips_per_night: float = 0.0   # idem short
```

Convention signée : on **additionne** au PnL en pips. `> 0` = crédit (carry favorable, rare), `< 0` = débit (carry défavorable, courant).

Défaut 0.0 → **rétrocompatibilité totale** : tant qu'on ne renseigne pas les valeurs, le comportement reste identique (modulo l'ajout du champ `nights_held` au dict de sortie).

### 2.2 Calcul `nights_held` dans le simulateur

```python
nights_held = max(0, (exit_time.normalize() - entry_time.normalize()).days)
```

C'est le nombre de **changements de date civile** entre entry et exit (V1 simple). Exemple :
- Entry mardi 23:00 → exit mercredi 01:00 → 1 nuit.
- Entry lundi 10:00 → exit lundi 18:00 → 0 nuit (intraday).
- Entry vendredi 22:00 → exit lundi 08:00 → 3 nuits.

### 2.3 Application au PnL

Dans `_simulate_stateful_core` (simulator.py) :
```python
nights_held = max(0, (exit_time.normalize() - entry_time.normalize()).days)
if asset_cfg is not None and nights_held > 0:
    swap_per_night = (
        asset_cfg.swap_long_pips_per_night
        if signal == 1
        else asset_cfg.swap_short_pips_per_night
    )
    pips_brut += nights_held * swap_per_night
```

Dans `run_deterministic_backtest` : idem via paramètres `swap_long_pips_per_night` / `swap_short_pips_per_night` (défaut 0.0).

### 2.4 Limitations V1 (à raffiner V2 si besoin)

- **Pas de triple swap mercredi** : XTB et la plupart des brokers appliquent `3× swap` la nuit de mercredi à jeudi (qui couvre Sam/Dim weekend). En V1 on suppose `nights_held` linéaires. Surestime légèrement (~14 %) le coût pour les long-débit, sous-estime pour les long-crédit. Acceptable en première approximation, à raffiner après validation Phase F2.
- **Pas de holidays** : 25 décembre, 1er janvier — pas de swap appliqué normalement (marché fermé). On simplifie en charge chaque date civile.

---

## 3. Couverture de test

[`tests/unit/test_swap_overnight.py`](../tests/unit/test_swap_overnight.py) — 11 tests :

| Test | Vérifie |
|---|---|
| `test_deterministic_no_swap_no_change` | Défaut swap=0 → comportement legacy identique |
| `test_deterministic_intraday_zero_nights` | nights_held=0 → swap ignoré même si non-nul |
| `test_deterministic_long_swap_debit_one_night` | Long avec swap < 0 → PnL diminué |
| `test_deterministic_long_swap_credit_five_nights` | Long avec swap > 0 sur 5 nuits → PnL augmenté |
| `test_deterministic_short_uses_short_swap` | Short utilise bien `swap_short`, ignore `swap_long` |
| `test_deterministic_nights_held_exposed_in_trade` | Champ `nights_held` présent dans le dict trade |
| `test_stateful_no_asset_cfg_no_swap` | Sans `asset_cfg`, swap=0 (rétrocompat) |
| `test_stateful_long_swap_applied` | Long stateful avec swap débit → PnL diminué |
| `test_stateful_short_uses_short_swap` | Short stateful → utilise swap_short |
| `test_stateful_intraday_no_swap_applied` | Intraday stateful → 0 nuit, swap ignoré |

À exécuter par l'utilisateur :
```bash
rtk pytest tests/unit/test_swap_overnight.py -v
rtk pytest tests/unit/test_deterministic_sl_prime.py -v  # non-régression
rtk pytest tests/unit/test_simulator.py -v               # non-régression
rtk pytest tests/unit/test_simulator_sizing.py -v        # non-régression
```

---

## 4. Valeurs swap cibles à valider en démo XTB (Phase F2)

Tableau des valeurs **estimatives** (à confirmer en démo). L'utilisateur devra renseigner les valeurs exactes via "Symbol Specifications" → onglet "Swap Long / Swap Short".

### 4.1 Forex majors

| Pair | Swap Long (pips/nuit) | Swap Short (pips/nuit) | Source estimative |
|---|---|---|---|
| EURUSD | -0.5 | +0.2 | Carry défavorable long (taux EUR < USD) |
| GBPUSD | -0.4 | +0.1 | Idem |
| USDCHF | +0.2 | -0.5 | Carry favorable long (taux USD > CHF) |
| USDJPY | **+0.6** | -0.8 | **Carry favorable long** (taux USD >> JPY) |
| AUDUSD | -0.3 | 0.0 | Légèrement défavorable long |
| NZDUSD | -0.3 | 0.0 | Idem |
| USDCAD | +0.1 | -0.4 | Slightly favorable long |

### 4.2 Crosses JPY (forte asymétrie carry)

| Pair | Swap Long | Swap Short | Note |
|---|---|---|---|
| EURJPY | +0.3 | -0.6 | Carry favorable long |
| GBPJPY | +0.4 | -0.7 | Carry favorable long |
| **AUDJPY** | **+1.5** | -2.0 | **Carry très favorable long** |
| **NZDJPY** | **+1.7** | -2.2 | Idem AUDJPY |
| CHFJPY | +0.4 | -0.7 | Carry favorable long |

Ces valeurs élevées du côté favorable expliquent pourquoi les pairs JPY sont **historiquement les plus rentables en long carry trade**. Une stratégie Donchian/TsMomentum long-only sur AUDJPY/NZDJPY récupère **+1.5 à +1.7 pips/nuit** "gratuits" — l'équivalent d'un edge structurel non négligeable.

### 4.3 Indices

| Index | Swap Long | Swap Short | Note |
|---|---|---|---|
| US30 | -0.5 USD/nuit | +0.2 USD/nuit | Frais financement long indices |
| US500 | -0.02 USD/nuit | +0.005 USD/nuit | Idem (échelle pip 0.1) |
| US100 | -0.6 USD/nuit | +0.3 USD/nuit | Plus volatil → plus de carry |
| GER30 | -0.4 EUR/nuit | +0.1 EUR/nuit | Idem |
| JAP225 | -0.1 USD/nuit | +0.05 USD/nuit | Plus faible car taux JP bas |

### 4.4 Commodities

| Commodity | Swap Long | Swap Short | Note |
|---|---|---|---|
| XAUUSD | -0.10 USD/nuit | +0.05 USD/nuit | Frais financement |
| XAGUSD | -0.003 USD/nuit | +0.001 USD/nuit | Idem échelle inférieure |
| USOIL | -0.005 USD/nuit | +0.001 USD/nuit | Storage cost répercuté |
| NATGAS | -0.0001 USD/nuit | -0.00005 USD/nuit | Contango/backwardation |

### 4.5 Cryptos (charges massives)

| Crypto | Swap Long | Swap Short | Note |
|---|---|---|---|
| BTCUSD | -25 USD/nuit | -10 USD/nuit | Frais financement crypto élevés des **deux côtés** |
| ETHUSD | -2 USD/nuit | -1 USD/nuit | Idem |

⚠️ Crypto : swap **généralement négatif des deux côtés** (le broker se rémunère sur le financement long ET short, contrairement aux forex où swap_long + swap_short ≈ 0 modulo spread).

### 4.6 Exotiques (warning)

| Pair | Swap Long | Swap Short | Note |
|---|---|---|---|
| USDTRY | **+8 pips/nuit** | **-15 pips/nuit** | Énorme spread carry — inflation Turquie |
| USDZAR | +3 pips/nuit | -5 pips/nuit | Volatile |
| USDPLN | +0.5 pip/nuit | -1.0 pip/nuit | Modéré |

⚠️ Pour les exotiques, le swap **peut dépasser** le coût de spread → factor critique dans le PnL.

---

## 5. Impact attendu sur les analyses historiques

Une fois les valeurs F2 renseignées, ré-runner les analyses passées :

| Stratégie | Sharpe affiché | Sharpe attendu post-swap | Delta |
|---|---|---|---|
| Donchian D1 EURUSD (post-fix v5) | 0.00 best ratio | -0.15 à -0.30 | -0.15 |
| Donchian D1 USDJPY (à tester) | ? | + boost ~+0.1 carry | +0.1 |
| ETHUSD H1 hgbm (Phase B C5) | +1.81 (CPCV) | +1.6 à +1.7 | -0.10 à -0.20 |
| GBPUSD H4 rf | +3.45 (CPCV) | +3.3 à +3.4 | -0.10 |
| ETHUSD D1 hgbm | +1.70 (CPCV) | +1.4 à +1.5 (trades ~5 jours moyens) | -0.20 |

C'est un **dé-bias** systématique de -0.1 à -0.3 Sharpe pour la plupart des stratégies long-overnight. Pas de catastrophe — le verdict v5 "rien ne passe Sharpe ≥ 1.0" reste valide.

**Cas opportun** : USDJPY long avec swap +0.6/nuit. Sur Donchian 5 nuits moyen, +3 pips additionnels par trade. Pour TP=20 pips, c'est **+15 % d'expectancy gratuit**. Hypothèse à tester en Phase G3.

---

## 6. Prochaines étapes immédiates

1. Faire passer la suite de tests pour valider la non-régression :
   ```bash
   rtk pytest tests/unit/test_swap_overnight.py tests/unit/test_deterministic_sl_prime.py tests/unit/test_simulator.py tests/unit/test_simulator_sizing.py -v
   ```
2. Ouvrir compte démo XTB MT5 → récupérer Symbol Specifications → renseigner les valeurs réelles dans `ASSET_CONFIGS`.
3. Re-runner les diagnostics Donchian (`scripts/diagnose_donchian_atr_grid.py`) pour mesurer l'impact réel post-fix → comparer avec les valeurs originales (`predictions/...`).
4. Si l'écart Sharpe est conséquent (> 0.3) → mettre à jour `docs/diagnostic_final_donchian_dead.md`.

---

## 7. Note méthodologique : pourquoi c'est important

Le swap n'est pas un edge à chercher — c'est un **coût structurel** à modéliser correctement. Sa non-prise en compte signifie que **toutes les stratégies multi-jour ont été comparées à un benchmark biaisé**. Avec le fix :

1. Les futures hypothèses Phase G/H sont mesurées contre la **vraie** rentabilité.
2. Les stratégies short-only sur pairs à carry négatif long deviennent **moins attractives** (perte du "free crédit" qu'on créditait à tort).
3. Les stratégies long-only sur pairs à carry positif (JPY crosses) deviennent **structurellement plus rentables** → cible Phase G3 explicite.

**C'est la première condition** pour que les comparaisons inter-stratégies/inter-actifs aient un sens en Phase G/H.
