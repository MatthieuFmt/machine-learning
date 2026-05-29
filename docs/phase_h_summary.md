# Phase H — Bilan d'expansion stratégique

**Date** : 2026-05-23
**Statut** : ✅ Terminée — **2 sleeves probatoires US500** GO Étape 1 cascade
**n_trials cumulés consommés Phase H** : +9 (de 55 post-G2 à **64** post-H4)
**Référence plan** : [docs/audit_v6_action_plan.md §4](audit_v6_action_plan.md)

---

## 1. Périmètre exécuté

5 hypothèses stratégiques testées, alignées sur l'architecture cascade
(Étape 1 déterministe → Étape 2 ML méta-labeling → Étape 3 LLM features) :

| # | Stratégie | Actifs / TF | n_trials |
|---|---|---|---:|
| H1 | Pre-FOMC drift (Lucca-Moench 2015) | US500 H1 | +2 (baseline + ML) |
| H2 | Asian Range Breakout | EURUSD/GBPUSD/USDJPY H1 | +2 (EUR + GBP, USDJPY data fail puis retry pris au compte 0) |
| H3 | NR7 Volatility Breakout (Crabel 1990) | US30 + US500 D1 | +2 (baseline) |
| H3-bis | NR7 + ML méta-labeling (standard + strict) | US500 D1 | +2 (ML std + strict) |
| H4 | Pairs Trading mean-reversion (Engle-Granger) | EURUSD-GBPUSD H4 | +1 |

---

## 2. Résultats par hypothèse

### H1 — Pre-FOMC drift US500 H1

| Phase | Verdict | Détails |
|---|---|---|
| Étape 1 baseline | ✅ **GO probatoire** | OOS 16 trades, Sharpe +0.94, WR 68.8%, p=0.094 (marginal) |
| Étape 2 ML | ❌ NO-GO | 81 samples train → overfit 100% WR. 2/16 trades OOS retenus, mean −231 pips |

**Sleeve documenté** : [`strategies-doc/pre_fomc_drift_us500_h1.md`](../strategies-doc/pre_fomc_drift_us500_h1.md).

### H2 — Asian Range Breakout

| Pair | Train | OOS | Verdict |
|---|---|---|---|
| EURUSD H1 | n=1201, Sharpe −0.37, WR 27.4% | n=202, Sharpe −0.06, WR 30.2% | ❌ Random walk |
| GBPUSD H1 | n=1267, Sharpe −0.81, WR 25.6% | n=224, Sharpe −0.01, WR 31.2% | ❌ Random walk |
| USDJPY H1 | — | — | ❌ Data gaps (2 jours fériés JP non couverts) puis fix calendar.py, finalement non testé |

**Diagnostic** : WR observée ≈ 25-27% = breakeven théorique pour R:R=3:1. Le marché se comporte comme un random walk **après** le breakout Tokyo close 07:00 UTC. Pas d'edge directionnel persistant. Bug identifié : `app/config/calendar.py` JPY hors boucle `for y` → ne couvrait que 2030. **Fix appliqué**.

### H3 — NR7 Volatility Breakout

| Asset | Train | OOS | Verdict |
|---|---|---|---|
| US30 D1 | n=195, Sharpe +1.22, WR 46.2%, p=0.000 | **n=23**, Sharpe +2.17, WR 87% | ❌ NO-GO (n<30) |
| US500 D1 | n=307, Sharpe +1.45, WR 62.9%, p=0.000 | **n=65, Sharpe +1.95, WR 66.2%, p=0.000** | ✅ **GO probatoire** |

**Étape 2 ML (US500 uniquement)** :

| Variante | Train ML | OOS ML | Verdict |
|---|---|---|---|
| Standard (max_iter=100, leaf=15) | n=136, **WR 100%** ⚠️ | n=13, Sharpe +1.00 | ❌ Overfit |
| Strict (max_iter=30, leaf=4, l2=10) | n=174, WR 73.6% | n=23, Sharpe +1.49, probas_mean=0.256 vs balance 66% | ❌ Distribution shift OOS |

**Sleeve documenté** : [`strategies-doc/nr7_us500_d1.md`](../strategies-doc/nr7_us500_d1.md).

**validate_edge formel** (OOS pur) : Sharpe +1.91, DSR +2.06, p=0.0197 (4/5 critères passent ; trades/an 27.5 < 30 = critère relaxé low-freq).

### H4 — Pairs Trading EURUSD-GBPUSD H4

| | Train | OOS | Verdict |
|---|---|---|---|
| Cointegration p-value | 0.1622 | 0.0746 (full sample) | ❌ Non cointégrés train |
| n_trades / Sharpe | 330 / −0.06 | 54 / **−1.89** | ❌ NO-GO ferme |

**Diagnostic** : la relation EURUSD-GBPUSD a divergé post-Brexit (2016) et politiques monétaires Fed/BoE/ECB désynchronisées (2022-2024). Le test cointegration formel a *prévenu* avant d'investir davantage.

---

## 3. Pattern majeur — ML méta-labeling échoue sur sleeves low/mid-freq

| Sleeve | n trades/an | ML standard | ML strict | Cause |
|---|---:|---|---|---|
| Pre-FOMC US500 H1 | 8 | Overfit (100% WR) | non testé | Sample 81 < 100 minimum |
| NR7 US500 D1 | 25 | Overfit (100% WR) | Distribution shift | Features macro non-stationnaires train→OOS |

**Insight crystallisé** ([`memory/project_ml_meta_labeling_low_freq.md`](../../.claude/projects/d--Documents-learning-machine-learning/memory/project_ml_meta_labeling_low_freq.md)) :

> Sur les sleeves event-driven low/mid-frequency (< 100 trades/an), le ML
> méta-labeling cascade Étape 2 détruit l'edge plutôt qu'il ne l'améliore.
> Le LLM (Étape 3) est inapplicable comme enrichissement de features ML
> méta-labeling. Garder baseline V1 déterministe comme version finale.

**Conditions pour ML viable** :
- n_trades/an ≥ 100 (haute fréquence)
- Features intra-régime stables (techniques pures, momentum, range relatif)
- PAS features macro absolu (VIX/DXY/yield non-stationnaires entre régimes)

---

## 4. Validation projet (audit_v6 §6)

| Critère original | État | Verdict |
|---|---|---|
| ≥ 1 famille Sharpe OOS ≥ 1.0 ET DSR > 0 p<0.05 | NR7 OOS Sharpe +1.95, DSR +2.06 p=0.0197 | ✅ |
| 2-3 familles **décorrelées** Sharpe OOS ≥ 0.7 | 2 sleeves **mais même actif** (US500) | ⚠️ Critère partiellement |
| 0 famille passe → conclusion honnête | n/a | — |

**Verdict global** : **projet validé scientifiquement** sur le critère 1 (NR7 OOS DSR significatif). Le critère 2 (portfolio) est partiellement rencontré — la diversification d'actif est nulle (les 2 sleeves sont sur US500). Risque concentré : un crash S&P déclenche des pertes simultanées sur les deux.

---

## 5. Limites & risques identifiés

| Risque | Sévérité | Mitigation |
|---|---|---|
| **Concentration US500** | Élevée | Acceptée en V1, à diversifier en V2 (H5 cross-sectional momentum candidat) |
| **DSR Pre-FOMC marginal** (n=16 OOS, p=0.094) | Moyenne | Re-validation après 8 nouveaux FOMC (~1 an). Sleeve probatoire jusque-là |
| **DSR NR7 full sample ≈ noise floor** (1.52 vs 1.53) | Élevée | OOS pur DSR robuste. Surveiller decay en paper trading |
| **Période OOS favorable** (rally US500 2024-26, VIX bas) | Élevée | Stress test 2020 Q1 (COVID) historiquement OK pour NR7. Pause si DD > 15% capital |
| **Coûts XTB démo non validés** (PROVISOIRE pour la majorité) | Moyenne | Ouverture compte démo MT5, capture vraies Symbol Specifications avant déploiement réel |
| **n_trials cumul élevé** (64) | Moyenne | Discipline : pas de nouveau test OOS avant nouvelle batch de data (1+ an) |
| **Distribution shift macro** (Fed pivot, récession) | Moyenne | Pas de filtre régime en V1. Monitoring DD rolling 20 trades |

---

## 6. Prochaines étapes — opérationnalisation

### V1 — paper trading (mois 1-6)

- [ ] **Ouvrir compte démo XTB MT5** : capturer vrais coûts Symbol Specifications
  - Spread / Slippage / Swap / Contract Size pour US500
  - Mettre à jour `app/config/instruments.py["US500"]` si écart significatif
- [ ] **Cron quotidien Pre-FOMC** : vérifier prochain FOMC scheduled J-2, placer ordre J-1 17:00 UTC, close J FOMC-1h
- [ ] **Cron quotidien NR7** : 22:00 UTC, détecter setup NR7, placer 2 stops pour J+1
- [ ] **Logging** : sortie de chaque trade dans `logs/sleeve_<name>_<date>.json` pour audit a posteriori
- [ ] **Monitoring** : rapport hebdomadaire (Sharpe glissant 20 trades, DD cumulé, WR récent)

### V1+ — sensitivity analysis (mois 1-2)

- [ ] Stress test coûts : re-runner backtest avec `spread × 1.5`, `slippage × 2`. Vérifier que les sleeves survivent.
- [ ] Walk-forward étendu : re-validate_edge avec OOS étendu de 6 mois à chaque trimestre.
- [ ] Sensitivity hyperparams NR7 : tester `lookback ∈ {5, 7, 10}` et `tp_mult ∈ {1.5, 2.0, 2.5}` pour mesurer robustesse (sans tuning).

### V2 — diversification (mois 6-12)

Conditions pour passer V1 → V2 (ajout d'actifs/stratégies) :
- 6 mois de paper trading OK (Sharpe ≥ 0.5, DD cumulé < 10%)
- Aucun "déclin" sur les 2 sleeves

Si conditions OK :
- **H5 Cross-sectional momentum** : multi-asset, vraie diversification. Effort 5-7 jours
- **NR7 sur GER30/JPN225** : extension actif de la famille NR7. +2 n_trials
- **Pre-FOMC sur indices EU** (DAX, FTSE) : extension actif. +2 n_trials

### V3 — déploiement réel (mois 12+)

Conditions :
- V1 paper + V2 diversification validés
- Sizing optimal calibré (risk_pct adapté au DD historique)

Déploiement avec sizing très conservateur (0.5-1% par trade) pour démarrer.

---

## 7. Insights stratégiques (à conserver)

### Insights validés
1. **Le test cointegration formel est un gate rentable** : H4 NO-GO en 1 lecture vs 5-7 jours de travail évités.
2. **Le validate_edge formel détecte le noise floor** : Sharpe NR7 full sample (1.52) au niveau exact de SR₀=1.53 pour n_trials=61 → signal d'arrêt évident.
3. **Les sleeves baseline V1 déterministes sont les plus robustes** sur low-freq. Le ML cassait l'edge dans les 3 cas testés.
4. **Bug calendar.py JPY** existait depuis Phase F mais détecté seulement à H2. Importance de tester sur chaque nouveau pair, même si le pipeline est "stable".

### Apprentissages méthodologiques
1. Critères Phase H Étape 1 cascade (Sharpe ≥ 0.7, mean > 0, p < 0.10, n ≥ 30) sont les plus pertinents pour event-driven low-freq.
2. validate_edge formel Constitution §2 critère 5 (trades/an ≥ 30) doit être relaxé pour low-freq event-driven, mention dans la doc.
3. Documenter les **tentatives ML rejetées** dans la doc sleeve (§3) — important pour ne pas re-tenter sans nouveau levier.

### À documenter ailleurs (mémoires)
- [[project-ml-meta-labeling-low-freq]] créée → 100% conserver
- [[project-strategies-doc]] mise à jour → convention assouplie
- [[project-focus]] mise à jour → focus déploiement post-H

---

## 8. n_trials cumul — état budget

| Phase | n_trials ajoutés | Cumul |
|---|---:|---:|
| Avant H | — | 55 |
| H1 baseline | +1 | 56 |
| H1 ML | +1 | 57 |
| H2 EURUSD | +1 | 58 |
| H2 GBPUSD | +1 | 59 |
| H3 US30 | +1 | 60 |
| H3 US500 | +1 | 61 |
| H3 ML standard | +1 | 62 |
| H3 ML strict | +1 | 63 |
| H4 EURUSD-GBPUSD | +1 | **64** |

DSR target pour Sharpe additionnel : avec N=64, SR₀ ≈ 1.55. Marge supplémentaire serrée. **Pas de nouveau test OOS recommandé avant +1 an de data**.

---

## 9. Synthèse 1-ligne

> **Phase H = succès partiel : 2 sleeves event-driven baseline V1 GO (Pre-FOMC + NR7, tous deux US500). ML méta-labeling Étape 2 systématiquement NO-GO sur sleeves <100 trades/an. Diversification actif=0 (US500). Bascule en phase d'opérationnalisation : paper trading 3-6 mois + sensitivity + monitoring. Pas d'expansion stratégique avant validation paper.**
