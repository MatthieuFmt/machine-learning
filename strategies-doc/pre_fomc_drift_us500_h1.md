# Pre-FOMC Drift US500 H1 — Sleeve #1 (V1 déterministe)

> 🚨 **CADUC (audit 2026-06-09)** : les p-values/DSR de cette fiche ont été
> calculés avec le bug « DSR ×√252 » (Sharpe annualisé passé au DSR avec
> n_obs = nb de trades → z gonflé). Le signal reste le candidat le plus
> crédible du projet (effet documenté, Lucca-Moench 2015), mais le statut GO
> est **suspendu** jusqu'à re-mesure : `python scripts/screen_pre_fomc.py`
> (stats corrigées : DSR canonique + t-test par trade + bootstrap).

> **Statut** : ⏸️ SUSPENDU (ex-GO net) — re-mesure requise, voir bandeau ci-dessus
> **Version** : V1 baseline déterministe sans ML
> **Date de gel** : 2026-05-22
> **n_trial consommé** : 1 (Étape 1) + 1 (Étape 2 ML rejetée) sur 56 cumulés
> **Hypothèse théorique** : Lucca & Moench (2015), *"The Pre-FOMC Announcement Drift"*, Journal of Finance

---

## 1. Résumé

Stratégie calendar event-driven, **long-only** sur l'indice S&P 500 (US500 CFD)
dans les 24 heures précédant chaque annonce de décision de taux du FOMC.
Edge théorique documenté (Lucca-Moench 2015 sur 1994-2011 : Sharpe ~1.5)
attribué à une prime de risque pré-annonce.

**Fréquence** : 8 FOMC scheduled/an → 8 trades/an. Bas volume mais edge stable
attendu si l'effet persiste.

---

## 2. Stratégie primaire — [`app/strategies/pre_fomc_drift.py`](../app/strategies/pre_fomc_drift.py)

| Paramètre | Valeur |
|-----------|--------|
| **Timeframe** | H1 (1 heure) |
| **Actif** | US500 (S&P 500 CFD) |
| **Direction** | Long-only (pas de short) |
| **Source événements** | `data/raw/economic_calendar/<YEAR>.csv` — filtre `event == "FOMC Statement"` |
| **Heures FOMC** | ET → UTC via `pytz` (gère DST automatiquement) |
| **Entrée** | Première barre H1 ≥ `FOMC_time − 24h` (au prix de Close) |
| **Sortie** | Dernière barre H1 ≤ `FOMC_time − 1h` (au prix de Close) |
| **TP/SL** | Aucun — hold à durée fixe (~23h) |
| **Position sizing** | 1 unité par trade (à raffiner en déploiement, cf §8) |
| **Anti-look-ahead** | Calendrier scrapé Forex Factory ex-ante, conversion ET→UTC déterministe |

**Note** : la fenêtre traverse 1 nuit UTC dans la quasi-totalité des cas
(FOMC à 17:00-19:00 UTC, entrée veille même heure), d'où un swap long
appliqué (cf §4).

---

## 3. Tentative ML méta-labeling rejetée (Étape 2 cascade)

Un méta-modèle `HistGradientBoostingClassifier` a été testé pour filtrer
les trades baseline (10 features : VIX level/z-score, DXY z-score,
yield slope 10y-3m, US500 returns 5d/20d, ATR%, dist SMA200, RSI,
days_since_last_fomc).

**Verdict : NO-GO ferme** — overfit catastrophique :

| Métrique | Train baseline | Train ML | OOS baseline | OOS ML |
|---|---:|---:|---:|---:|
| n_trades | 88 | 34 (filtre P≥0.45) | 16 | 2 |
| Sharpe | +0.48 | +0.97 | +0.94 | +1.05 |
| WR | 42.0% | **100%** | 68.8% | 50.0% |
| Mean PnL | +46 pips | +222 pips | +123 pips | **−231 pips** |

Le 100% WR train et seulement 2 trades retenus OOS confirment un overfit
typique sur petit échantillon. **Cause structurelle** : 81 samples train
utilisables (warmup macro shifte le sample) pour 10 features = ratio 8:1,
en-dessous du minimum statistique pour ML calibré sur séries temporelles.

**Pas de quick fix** sur cette stratégie : la fréquence (8 FOMC/an) limite
intrinsèquement le sample. Le LLM (Étape 3) souffrirait du même problème.

**Décision** : on garde la baseline déterministe comme sleeve final.
L'architecture cascade ML/LLM sera réutilisée sur les stratégies haute
fréquence (H2 Asian Range, ~100-300 trades/an).

Source détaillée : [`predictions/h1_pre_fomc_drift_ml_us500.json`](../predictions/h1_pre_fomc_drift_ml_us500.json).

---

## 4. Calibration coûts — [`app/config/instruments.py`](../app/config/instruments.py)

`ASSET_CONFIGS["US500"]` (XTB Standard, capture 2026-05-15) :

| Composant | Valeur (pips US500, pip = $0.1) |
|---|---:|
| Spread | 0.5 |
| Slippage | 0.1 (majeure, 0.2× spread) |
| Commission | 0.0 |
| **Coût round-trip** | **0.7 pips** |
| Swap long /nuit | −16.0 (financement SOFR + ~5%) |
| Swap short /nuit | +2.0 |
| Pip size | 0.1 (cotation au dixième de point) |
| Pip value EUR | 0.092 |

**Coût total par trade** : 0.7 (round-trip) + 16.0 (1 nuit swap long) = **16.7 pips**.

Vérification : sur le 1er trade train (2012-01-25), `pips_brut = −22.5`,
`pips_net = −39.2`, soit −16.7 pips de coûts → cohérent.

---

## 5. Performances Train (2012-01 → 2022-12)

| Métrique | Valeur |
|---|---:|
| FOMC events Train | 105 (incl. ~17 unscheduled/skippés sample data) |
| Trades effectués | **88** |
| Sharpe per-trade | **+0.48** |
| Win rate | 42.0% |
| Mean PnL | +46.1 pips |
| Median PnL | −11.9 pips |
| Total PnL | +4 057 pips |
| Max DD | −967 pips |
| p-value bootstrap (mean > 0) | **0.033** |

**Lecture** : Sharpe modeste mais p-value en-dessous de 5% — l'edge est
détectable statistiquement sur l'historique. La median négative pour une
mean positive révèle une **distribution à queue droite** (quelques gros
gagnants compensent une majorité de petites pertes, cohérent avec
"prime de risque pré-annonce" + frais fixes par trade).

Source : [`predictions/h1_pre_fomc_drift_us500.json`](../predictions/h1_pre_fomc_drift_us500.json).

---

## 6. Performances OOS (2024-01 → 2026-05)

| Métrique | Valeur | Seuil GO |
|---|---:|---:|
| FOMC events Test | 16 (scheduled) | — |
| Trades effectués | **16** | — |
| **Sharpe per-trade** | **+0.94** | ≥ 0.7 ✅ |
| Win rate | **68.75%** | > 50% (info) ✅ |
| Mean PnL | **+122.9 pips** | > 0 ✅ |
| Median PnL | +82.2 pips | > 0 ✅ |
| Total PnL | +1 966 pips | — |
| Max DD | −583 pips | < 1500 ✅ |
| **p-value bootstrap (mean > 0)** | **0.094** | < 0.10 ✅ (marginal) |

### ⚠️ Marginalité de la p-value

Le critère GO `p < 0.10` est respecté mais **de justesse** (0.094 vs 0.10).
Avec **n=16 seulement**, la puissance statistique est faible :
- Élargir l'IC du Sharpe : ±0.5 environ → vrai Sharpe pourrait être entre +0.4 et +1.4.
- La validation est **probatoire**, pas définitive. Il faut **8-16 nouveaux FOMC
  (1-2 ans)** pour confirmer la persistance.

Le sleeve est activé mais doit être monitoré strictement (cf §8).

---

## 7. Critères GO/NO-GO appliqués

| Critère Étape 1 cascade | Seuil | Observé | Verdict |
|---|---:|---:|---|
| Sharpe OOS | ≥ 0.7 | +0.94 | ✅ |
| Mean PnL OOS | > 0 | +122.9 | ✅ |
| p-value bootstrap | < 0.10 | 0.094 | ✅ (marginal) |

**Verdict global** : GO Étape 1 → sleeve activé. Étape 2 (ML) testée et rejetée
(§3). Étape 3 (LLM) **non envisagée** sur cette stratégie (même limite de sample).

---

## 8. Déploiement — Checklist

### Prérequis techniques
- [ ] Python 3.12 avec dépendances [`requirements.txt`](../requirements.txt)
- [ ] Accès données US500 H1 (Dukascopy ou broker équivalent)
- [ ] Compte XTB Standard (spread 0.5 pt + slippage 0.1 sur US500)
- [ ] Calendrier économique mis à jour mensuellement (Forex Factory scrape ou API)
- [ ] Cron job pour rafraîchir `data/raw/economic_calendar/<YEAR>.csv`

### Fichiers à déployer
```
app/
├── config/
│   └── instruments.py            ← ASSET_CONFIGS["US500"] (coûts F6)
├── strategies/
│   └── pre_fomc_drift.py         ← load_fomc_announcement_times + simulate
└── core/
    └── logging.py
data/raw/economic_calendar/
└── <year>.csv                    ← scrape Forex Factory à jour
```

### Pipeline opérationnel
1. **J-2 à 09:00 UTC** : vérifier que le prochain FOMC scheduled est dans `<year>.csv`.
2. **FOMC_time − 24h ± 30min** : ouvrir long US500 au prix Close H1 le plus proche.
3. **FOMC_time − 1h** : fermer la position au prix Close H1 le plus proche.
4. Logger entrée/sortie + slippage réalisé pour suivi (§8).

### Position sizing (à raffiner)
- V1 : 1 unité fixe par trade (backtest).
- Production : risk per trade = 2% du capital, mais SL "artificiel" requis
  pour calculer la position. Suggestion : SL = −300 pips US500 (= ~3% capital
  à 2% risk) sans qu'il soit déclenché en simulation (pas de SL dans la stratégie).
- **À calibrer en démo avant déploiement réel.**

### Variables d'environnement
```bash
PYTHONIOENCODING=utf-8
TZ=UTC
```

### Vérifications avant mise en prod
- [ ] `rtk -- pytest tests/unit/ -v` (tous tests passent)
- [ ] Lecture manuelle des 16 trades OOS dans le JSON pour visualiser le profil
- [ ] Simulation paper sur les 2-3 prochains FOMC avant capital réel

---

## 9. Monitoring — Métriques à suivre en production

| Métrique | Seuil alerte | Action si dépassé |
|---|---|---|
| Sharpe glissant 8 derniers FOMC | < 0.3 | Review manuelle, possible pause |
| Win rate glissant 8 derniers FOMC | < 30% | Vérifier régime macro (decay post-discovery ?) |
| Drawdown cumulé sleeve | > −800 pips (~10% capital) | Pause + audit |
| Délai signal → exécution | > 5 min | Problème data/cron |
| Slippage réalisé vs modélisé (0.1 pip) | > 3× modélisé | Recalibrer ou changer broker |
| Nouvelle FOMC manquante du calendar | toute | Bug ingestion calendar |

**Re-évaluation périodique** : après chaque cycle de 8 FOMC (≈ 1 an),
re-runner `scripts/run_h1_pre_fomc_drift.py` et comparer Sharpe glissant.

---

## 10. Limites connues et risques

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **Sample size faible** (n=16 OOS) | Élevée (structurelle) | Élevé | Monitoring strict §9, re-évaluation tous les 8 FOMC. Sleeve probatoire. |
| **Post-discovery decay** (papier publié 2015, effet potentiellement arbitré) | Moyenne | Très élevé | OOS 2024-26 montre que l'effet persiste pour l'instant. À surveiller |
| **Régime macro spécifique** (hausse Fed 2022-2023 = environnement bull pré-FOMC) | Moyenne | Élevé | Si Fed coupe les taux → effet pourrait inverser. Inclure VIX/DXY en filtre futur (Étape 2 cascade impossible ici faute de sample) |
| **Calendar incorrect/decalé** (heure FOMC mal scrapée, événement supprimé) | Faible | Critique | Validation manuelle de chaque event J-7 avant trade |
| **Swap variable XTB** (taux Fed mouvant) | Moyenne | Faible | Re-calibrer `swap_long_pips_per_night` trimestriellement |
| **Slippage news intra-fenêtre** (statement décalé d'1h, FOMC anticipé) | Faible | Moyen | Time-stop à FOMC − 1h limite l'exposition au statement lui-même |
| **Drawdown intra-trade non bridé** (pas de SL) | Moyenne | Moyen | Max DD historique −583 pips OOS, acceptable si sizing à 1-2% capital |

### Quand arrêter le sleeve
- 3 FOMC consécutifs avec PnL < −100 pips chacun.
- Sharpe glissant sur 8 FOMC < 0.0.
- DSR < 0 après ré-évaluation annuelle (en incluant les nouveaux OOS dans le bootstrap).

---

## 11. Références croisées

| Document | Lien |
|----------|------|
| Stratégie code | [`app/strategies/pre_fomc_drift.py`](../app/strategies/pre_fomc_drift.py) |
| Tentative ML rejetée | [`app/strategies/pre_fomc_meta.py`](../app/strategies/pre_fomc_meta.py) |
| Script lanceur Étape 1 | [`scripts/run_h1_pre_fomc_drift.py`](../scripts/run_h1_pre_fomc_drift.py) |
| Script lanceur Étape 2 (ML) | [`scripts/run_h1_pre_fomc_drift_ml.py`](../scripts/run_h1_pre_fomc_drift_ml.py) |
| Résultats Étape 1 (GO) | [`predictions/h1_pre_fomc_drift_us500.json`](../predictions/h1_pre_fomc_drift_us500.json) |
| Résultats Étape 2 (NO-GO ML) | [`predictions/h1_pre_fomc_drift_ml_us500.json`](../predictions/h1_pre_fomc_drift_ml_us500.json) |
| Bilan Phase G (pivot cascade) | [`docs/phase_g_summary.md`](../docs/phase_g_summary.md) |
| Bilan Phase F (swap + coûts) | [`docs/phase_f_summary.md`](../docs/phase_f_summary.md) |
| Plan stratégique global | [`docs/audit_v6_action_plan.md`](../docs/audit_v6_action_plan.md) |
| Publication originale | Lucca, D. O., & Moench, E. (2015). *The Pre-FOMC Announcement Drift*. Journal of Finance, 70(1), 329–371 |

---

> **Dernière mise à jour** : 2026-05-22 — Phase H1 terminée, sleeve activé probatoire
> **Prochaine revue** : après 8 FOMC supplémentaires (≈ 2027-05) — ré-évaluation Sharpe glissant + DSR cumul
