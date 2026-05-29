# Audit final post-mortem — Pipeline ML Trading v4.1.0-extended

**Date** : 2026-05-18
**Verdict validation finale** : ❌ NO-GO (Portfolio Sharpe 4.97 ≤ P95 Monte Carlo 9.96)
**n_trials cumulé** : 29
**Document à destination d'un auditeur externe (humain ou IA)**

---

## 1. Résumé du projet

- **Projet** : Machine learning appliqué au trading de CFD (Forex + Crypto + Indices)
- **Pipeline** : v4.1.0-extended (Phase A originale A5→A9 + Phase C extension C1→C5)
- **Broker cible** : XTB
- **Capital de référence** : 10 000 €, risque 2 % par trade
- **Split temporel** : train ≤ 2022, val = 2023 (inutilisée), test ≥ 2024
- **Stratégie de base** : Donchian Breakout (N, M) + méta-labeling ML (RF / HGBM / Stacking)
- **Méthodologie** : Pipeline gelé sur train uniquement (A5-A9), puis test OOS unique (B1-B4, Phase C5 extra, validation finale)
- **12 couples testés en OOS** → **6 stratégies validées individuellement** (Sharpe OOS ≥ 0.5)
- **Portefeuille combiné equal-risk** : Sharpe +4.97, DSR 19.47 (p=0.000), Max DD 2.7 %, WR 54.2 %, 1310 trades/an
- **Verdict final** : ❌ NO-GO — le benchmark Monte Carlo (P95 random Sharpe = 9.96) n'est pas battu
- **Critères constitution passés** : 5/5 (Sharpe ≥ 1.0, DSR > 0, DD < 15 %, WR > 30 %, trades/an ≥ 30)
- **Benchmarks** : Beat B&H+0.3 ✅ (4.97 > 0.82), Beat P95 random ❌ (4.97 < 9.96)

---

## 2. Fichiers à lire pour comprendre le contexte (dans l'ordre)

```
1. prompts/00_constitution.md — règles du projet, critères de succès, split temporel, garde-fous
2. CLAUDE.md — architecture du repo, conventions, design patterns
3. JOURNAL.md — historique complet des étapes, compteur n_trials, sessions
4. prompts/pivot_v4/00_README.md — vue d'ensemble pivot v4, dépendances, ordre canonique
5. prompts/pivot_v4/C0_README_phase_c.md — Phase C extension, règles strictes, coûts provisoires
6. docs/phase_a_extended_summary.md — bilan Phase A+C, 11 couples figés, shortlist C5
7. docs/v3_final_report.md — rapport validation finale, verdict, benchmarks
8. predictions/validation_finale.json — résultats bruts (métriques, strategy_details, benchmarks)
9. docs/feature_ranking_v4_extended.md — ranking features Phase C2, stabilité, patterns par classe d'actif
10. docs/model_selection_v4_extended.md — sélection modèles Phase C3, CPCV, shortlist C4
11. docs/hyperparam_tuning_v4_extended.md — tuning hyperparams Phase C4, nested CPCV, gaps
```

---

## 3. Erreurs potentielles à rechercher

### 3.1 Look-ahead bias

**Question** : Le cutoff 2022-12-31 est-il strictement respecté dans tous les scripts ?

- Vérifier que toutes les features à l'instant `t` n'utilisent que l'information ≤ `t`. La constitution (Règle 7) l'exige, et un test [`test_indicators_look_ahead.py`](tests/unit/test_indicators_look_ahead.py) existe.
- Vérifier que les features cross-asset (`btcusd_return_5`, `usdchf_return_5`, `xauusd_return_5`) utilisent bien le retour à `t` et non `t+1`.
- Vérifier que les features de session (`session_tokyo`, etc.) n'utilisent pas l'information de la bougie en cours avant sa clôture.
- Vérifier que le `merge_asof` multi-TF (H4/D1 → H1) ne forward-fill pas accidentellement des données futures.

**Fichiers à auditer** :
- [`app/features/indicators.py`](app/features/indicators.py) — toutes les fonctions de feature
- [`app/features/superset.py`](app/features/superset.py) — assemblage du superset
- [`app/features/merger.py`](app/features/merger.py) — merge multi-TF (s'il existe encore)
- [`tests/unit/test_indicators_look_ahead.py`](tests/unit/test_indicators_look_ahead.py) — couverture suffisante ?

### 3.2 Data snooping

**Question** : Le test set ≥ 2024 a-t-il été consulté avant la validation finale ?

- Vérifier que [`verify_no_snooping.py`](scripts/verify_no_snooping.py) n'a jamais détecté d'accès suspects.
- Vérifier l'historique des `read_oos()` dans [`app/testing/snooping_guard.py`](app/testing/snooping_guard.py) — chaque lecture OOS doit être documentée avec prompt, hypothèse, Sharpe, n_trades.
- Vérifier que les 29 n_trials sont tous documentés dans [`JOURNAL.md`](JOURNAL.md) (tableau n_trials).
- Vérifier que les hypothèses H06, H07, H1, H5 (brûlées en OOS, cf. pivot v4) n'ont pas été re-testées.
- **Point critique** : la Phase C (C1→C5) s'est déroulée entièrement sur train ≤ 2022 (0 n_trial). Mais les scripts `run_c5_pipeline_lock_extended.py` et suivants ont-ils accidentellement lu ≥ 2024 ?

### 3.3 Sur-apprentissage (overfitting)

**Cas suspects** :

| Couple | Accuracy train | Accuracy test | Gap | Analyse |
|---|---|---|---|---|
| GBPUSD D1 | 75.5 % | 65.8 % (WR) | ~10 pts | Gap acceptable. Mais Sharpe train=8.62 → test=5.17, gap Sharpe=3.45. Overfitting modéré ? |
| ETHUSD D1 | **97.1 %** | 43.9 % (WR) | **53 pts** | **Sur-apprentissage FLAGRANT**. 175 trades train, accuracy 97.1 % = le modèle a mémorisé le train. WR test 43.9 % reste positive mais l'accuracy classification est probablement proche de l'aléatoire. |
| USDCHF D1 | 86.9 % | 54.1 % (WR) | ~33 pts | Écart significatif. Stacking avec defaults → overfitting probable. |
| EURUSD D1 | 68.1 % | 58.1 % (WR) | ~10 pts | Écart modéré, stacking defaults. Acceptable. |

**Questions** :
- ETHUSD D1 : le modèle HGBM avec 97.1 % d'accuracy sur 175 échantillons est-il statistiquement valide ? La taille d'échantillon (n=175) est très faible. L'accuracy stratosphérique suggère un pur overfitting, même si le Sharpe OOS reste positif (+2.14).
- GBPUSD D1 : exclu de C5 pour gap=1.92 (inner-outer Sharpe), mais réintégré dans la validation finale avec Sharpe=5.17. Le gap C4 était-il un faux positif ? Ou le Sharpe test est-il gonflé par la chance ?
- Les modèles Stacking (EURUSD D1, USDCHF D1) utilisent des hyperparams par défaut — aucun tuning nested CPCV. Sont-ils réellement robustes ?

### 3.4 Coûts de trading PROVISOIRES

**Contexte** : BTCUSD, ETHUSD, GBPUSD, USDCHF ont des coûts marqués PROVISOIRES dans C1 (cf. [`C0_README_phase_c.md`](prompts/pivot_v4/C0_README_phase_c.md:44)).

**Risques** :
- Si les vrais spreads XTB sont plus élevés que les valeurs provisoires, les Sharpe OOS sont sur-estimés.
- Impact majeur sur les stratégies à haute fréquence (H1, H4) où les coûts pèsent proportionnellement plus.
- ETHUSD H1 (Sharpe +1.81 outer) serait le plus impacté par une correction des coûts.
- GBPUSD H4 (Sharpe +3.45 outer) — 1103 trades OOS, chaque trade subit le spread.

**Fichier à auditer** :
- [`app/config/instruments.py`](app/config/instruments.py) — comparer `friction_pips` pour BTCUSD, ETHUSD, GBPUSD, USDCHF avec les vrais spreads XTB (compte démo).
- [`docs/cost_audit_v2.md`](docs/cost_audit_v2.md) — justification des coûts.

### 3.5 Hypothèse Donchian — biais potentiel

**Question** : La target est basée sur Donchian (N, M). Les features pourraient-elles « apprendre » le Donchian ?

- La target est un breakout Donchian : `1` si le prix casse le high des N dernières barres, `-1` si casse le low, `0` sinon.
- Les features incluent des distances aux MAs, ATR, volatilité, etc. Une feature qui contient l'information du breakout (ex : `close > high_N`) serait redondante avec la target et créerait un biais.
- Mais le méta-labeling filtre les signaux Donchian : le modèle n'apprend PAS à prédire le breakout, il apprend à filtrer les faux breakouts. Le risque est donc limité.
- **Vérifier** : les features de type `dist_donchian_high` ou équivalentes ne doivent pas exister dans le superset.

### 3.6 Méta-labeling inversé

**Constat** : B1 a montré que le méta-labeling dégrade le Sharpe sur US30 D1 (NO-GO, Sharpe=0.82, 12 trades OOS). Sans méta-labeling, Donchian brut avait Sharpe +8.84 en H05 (v2).

**Hypothèses** :
1. **Erreur d'implémentation** : le méta-labeling filtre-t-il correctement ? Vérifier [`app/backtest/meta_labeling.py`](app/backtest/meta_labeling.py).
2. **Problème fondamental** : sur US30 D1, Donchian est déjà un excellent filtre. Ajouter un méta-labeling ML ajoute du bruit sans valeur ajoutée.
3. **Seuil mal calibré** : le seuil 0.55 (A8) est-il trop conservateur ? Trop de trades rejetés → trop peu de trades OOS (12).
4. **Taille d'échantillon train insuffisante** : 232 trades train pour Donchian US30 D1 → le méta-labeling n'a pas assez d'exemples pour apprendre.

### 3.7 Walk-forward vs train/test simple

**Incohérence potentielle** :
- Phase A (A7, A8) utilise CPCV (Combinatorial Purged Cross-Validation).
- Phase C (C3, C4) utilise également CPCV.
- H_new2 (walk-forward rolling) a été testé et abandonné (NO-GO).
- La validation finale utilise un split train/test simple (train ≤ 2022, test ≥ 2024).

**Question** : Le CPCV est utilisé pour la sélection et le tuning, mais le backtest final est un split simple. Est-ce cohérent ? Un walk-forward glissant sur plusieurs fenêtres serait plus robuste.

### 3.8 Taille d'échantillon critique

| Couple | Trades train | Trades test | Modèle | Accuracy train | Risque |
|---|---|---|---|---|---|
| ETHUSD D1 | 175 | 328 | HGBM | 97.1 % | 🔴 Taille train très faible, overfitting massif |
| XAUUSD D1 | 85 | — | Stacking | — | 🔴 Exclu pour insuffisance trades |
| EURUSD H4 | 506 | 54 | RF | — | 🟠 Seulement 54 trades OOS |

**Question** : ETHUSD D1 avec 175 trades train et accuracy 97.1 % — le modèle HGBM est-il fiable ? À 175 échantillons, un HGBM avec `max_depth=3, max_leaf_nodes=15` peut facilement mémoriser.

### 3.9 Stacking sans tuning

**Cas** : EURUSD D1 et USDCHF D1 utilisent Stacking (RF + HGBM + LogReg meta) avec hyperparams par défaut.

**Problème** : Le nested CPCV était trop lent pour le stacking → defaults conservés. Les performances OOS (Sharpe 4.01 et 3.29) sont bonnes, mais :
- Le stacking est-il réellement meilleur que RF seul ou HGBM seul ?
- Les defaults scikit-learn sont-ils optimaux pour ces données ?
- Le LogReg meta-classifier est-il bien calibré ?

### 3.10 Monte Carlo benchmark — P95 irréaliste

**Constat** : Le P95 Monte Carlo = 9.96 sur le portefeuille. Cette valeur semble extrêmement élevée.

**Questions** :
- La méthodologie Monte Carlo est-elle correcte ? Génère-t-on des signaux aléatoires avec la même fréquence de trading que la stratégie réelle ?
- Le P95 est-il calculé in-sample (train ≤ 2022) ? Si oui, il reflète le « hasard chanceux » sur une période potentiellement plus facile.
- Les signaux aléatoires respectent-ils la même distribution de trades par mois/année ?
- 9.96 est 2× le Sharpe observé du portefeuille (4.97). Un P95 aussi élevé suggère soit une méthodologie incorrecte, soit une période train extrêmement volatile favorable au hasard.

**Fichier à auditer** :
- [`app/analysis/edge_validation.py`](app/analysis/edge_validation.py) — fonction de calcul du Deflated Sharpe Ratio et Monte Carlo.

### 3.11 Fuite de données entre timeframes

**Question** : Les features D1 utilisent-elles des données H4/H1 qui « voient » le futur ?

- Une bougie D1 se forme à partir des bougies H4/H1 de la même journée. Si on utilise la valeur H4 de 16:00 pour prédire le breakout D1 à 00:00 le lendemain, c'est légitime.
- Mais si on utilise la valeur H4 de 20:00 (dernière barre H4 du jour D) pour une décision D1 à l'ouverture de D, il y a look-ahead car la bougie D1 n'est pas encore clôturée.
- **Vérifier** : dans le merge multi-TF, le décalage temporel est-il correctement géré ?

### 3.12 Gap 2023

**Constat** : L'année 2023 n'est utilisée ni en train (≤ 2022) ni en test (≥ 2024).

**Problème** :
- 2023 était destinée à la validation. Mais dans les faits, elle n'a pas été utilisée systématiquement.
- Perte d'information : 2023 contient des données qui auraient pu servir à calibrer les seuils ou détecter l'overfitting avant le test final.
- La constitution dit `val = 2023` mais dans les implémentations, le cutoff est `train ≤ 2022, test ≥ 2024`. Où est passée 2023 ?
- **Conséquence** : si 2023 est un régime très différent de 2022, le gap train→test est artificiellement grand.

### 3.13 Régime de marché

**Question** : 2024-2026 est très différent de 2010-2022. Le modèle est-il robuste à un changement de régime ?

- Période train : 2010-2022 (bull market prolongé post-2008, COVID crash 2020, recovery 2021, bear 2022).
- Période test : 2024-2026 (bull AI, taux élevés, volatilité macro).
- Les modèles entraînés sur 12 ans de données « anciennes » sont-ils adaptés au régime actuel ?
- GBPUSD D1 : Sharpe test 5.17 — suggère que le modèle gère le changement. Mais est-ce robuste ou chanceux ?
- **Recommandation** : tester la robustesse sur des sous-périodes contrastées (2020 crash, 2021 recovery, 2022 bear).

### 3.14 Multiplicité des tests (correction non appliquée)

**Constat** : 12 couples testés → 6 GO (50 % de succès). Aucune correction pour tests multiples.

- Avec correction de Bonferroni : seuil α = 0.05 / 12 = 0.00417. La p-value du DSR portefeuille = 0.000 — cela passe même avec Bonferroni.
- Mais chaque couple individuellement : le DSR est-il significatif après correction ?
- Le problème n'est pas le portefeuille (DSR fort), mais le risque de « cherry-picking » : on ne garde que les 6 GO sur 12 et on jette les 6 NO-GO. Sans correction, le biais de sélection est réel.

### 3.15 pip_value_eur provisoire

**Constat** : Les nouveaux actifs (BTCUSD, ETHUSD, GBPUSD, USDCHF) ont des `pip_value_eur` définis en C1, potentiellement approximatifs.

**Impact** :
- Les métriques monétaires (PnL en €, DD en €) dépendent de `pip_value_eur`.
- Si la valeur est incorrecte, le Sharpe et le DD en € sont biaisés.
- Les calculs de taille de position (2 % risk) seraient également affectés en production.

---

## 4. Choses qu'on aurait dû faire et qu'on n'a pas faites

1. **Vérifier les vrais spreads XTB** (Option B du post-Phase C) — jamais exécutée. BTCUSD, ETHUSD, GBPUSD, USDCHF ont des coûts PROVISOIRES. Tout résultat OOS est conditionnel à ces coûts.

2. **Tester ETHUSD H4 avec des hyperparams tunés** — exclu de C5 pour Sharpe outer 0.39 < 0.5, mais avec des paramètres différents (learning_rate plus bas, max_depth différent), le Sharpe pourrait passer au-dessus de 0.5. Jamais exploré.

3. **Tester US30 D1 en OOS ≥ 2024 avec le nouveau pipeline** — H05 (v2) avait Sharpe walk-forward +8.84 sur US30 D1 sans méta-labeling. B1 (v4) a testé avec méta-labeling → NO-GO (Sharpe=0.82). Mais US30 D1 sans méta-labeling n'a jamais été retesté avec le simulateur corrigé et les vrais coûts XTB.

4. **Appliquer une correction pour tests multiples** — 6 GO sur 12 couples testés = 50 %. Avec correction Bonferroni, combien survivent ? Le DSR portefeuille passe (p=0.000), mais la confiance statistique par couple est inconnue.

5. **Faire un walk-forward complet sur tous les couples** — remplacer le split train/test simple par un walk-forward glissant (rolling window) avec re-entraînement périodique. H_new2 a testé cette approche sur US30+XAUUSD (NO-GO), mais n'a pas été étendu aux 6 couples GO.

6. **Tester la robustesse sur différents régimes** — sous-périodes : crise COVID 2020, reprise 2021, bear market 2022, bull AI 2024-2025. Les modèles performent-ils uniformément ou seulement dans un régime ?

7. **Calculer le drawdown en % du capital** — le Max DD rapporté (2.7 % portefeuille) est calculé sur l'equity curve. Mais un DD en € avec sizing 2 % serait plus informatif qu'un DD en pourcentage de l'equity.

8. **Backtest avec sizing adaptatif** — vol targeting (ajuster la taille selon la volatilité récente), risk parity (allocation égale au risque par sleeve). Le sizing actuel est un equal-risk naïf (même risque par trade, pas par actif).

9. **Comparer avec un benchmark passif plus pertinent** — le benchmark actuel est US30 B&H. Un portefeuille 60/40 (actions/obligations) ou un ETF monde serait plus représentatif d'une alternative passive pour un capital de 10 000 €.

10. **Documenter les corrélations entre stratégies** — le portefeuille equal-weight combine 6 stratégies, mais si elles sont corrélées (ex : GBPUSD D1 + GBPUSD H4 + EURUSD D1 partagent le même driver USD), la diversification réelle est inférieure à la diversification apparente.

11. **Faire un test de robustesse avec des coûts majorés** — +20 % spread/slippage pour simuler des conditions de marché dégradées (news, faible liquidité). Les Sharpe survivent-ils ?

12. **Vérifier que les features ne contiennent pas d'information du futur** — look-ahead validation systématique pour chaque feature du superset, pas seulement un test unitaire générique.

13. **Nettoyer les scripts morts et les fichiers .bak** — le repo contient des fichiers `.bak` et des scripts qui ne sont plus utilisés. Dette technique.

14. **Ajouter des tests de non-régression pour toutes les étapes C1-C5** — les tests unitaires actuels couvrent les résultats (features sélectionnées, modèle retenu, hyperparams tunés) mais pas le processus complet.

---

## 5. Questions ouvertes pour le futur

1. **Le pipeline peut-il être exécuté en temps réel ?**
   - Les données actuelles sont des CSV historiques. Un flux live (XTB API, Dukascopy, OANDA) est nécessaire.
   - Fréquence de mise à jour : D1 = 1 fois/jour, H4 = 6 fois/jour, H1 = 24 fois/jour.
   - Latence acceptable pour D1/H4, critique pour H1.

2. **Quel serait le coût d'infrastructure ?**
   - VPS : 5-20 €/mois (Hetzner, OVH).
   - Données live : 0-50 €/mois selon la source (XTB gratuit si compte, Dukascopy payant).
   - Connexion broker : 0 € (XTB API gratuite via compte réel ≥ 1000 €).
   - Total estimé : 10-70 €/mois.

3. **Comment gérer les corrélations entre stratégies en portefeuille ?**
   - Le portefeuille equal-weight ignore les corrélations.
   - Risk parity ou mean-variance optimization (Markowitz) nécessitent une matrice de covariance stable.
   - H13 (corrélation weighting) était prévu dans la roadmap v3 mais jamais implémenté.

4. **Faut-il ajouter un filtre de volatilité ou de régime ?**
   - Un filtre de régime (H09 jamais fait) pourrait éviter de trader en périodes adverses.
   - Un filtre de volatilité pourrait réduire le risque de queues épaisses.

5. **Les modèles doivent-ils être ré-entraînés périodiquement ?**
   - Fréquence proposée : tous les 3-6 mois avec les nouvelles données.
   - Problème : chaque ré-entraînement = nouveau n_trial si on regarde le test set pour valider.
   - Solution : walk-forward avec fenêtre glissante, le ré-entraînement fait partie du processus.

6. **Comment monitorer la performance en production ?**
   - Dashboard temps réel (Equity curve, Sharpe glissant, DD courant, WR glissant).
   - Alertes si DD > 10 % ou Sharpe glissant < 0 sur 30 jours.
   - Détection de dérive du modèle (drift detection).

7. **Quel niveau de capital minimum pour que les coûts de trading ne dévorent pas les gains ?**
   - Avec 2 % de risque par trade, un trade perdant coûte 200 € (sur 10 000 €).
   - Le spread est fixe en pips, donc son impact relatif diminue avec le capital.
   - Estimation : à 1000 € de capital, les coûts représentent 10× plus en proportion qu'à 10 000 €. Capital minimum recommandé : 5 000-10 000 €.

---

## 6. Recommandations pour la suite

### Recommandation 1 — Option B d'abord : vérifier les spreads XTB réels

**Priorité** : 🔴 CRITIQUE

Avant toute mise en production, capturer les spreads réels sur compte démo XTB pour BTCUSD, ETHUSD, GBPUSD, USDCHF. Mettre à jour [`app/config/instruments.py`](app/config/instruments.py). Si l'écart avec les valeurs provisoires est significatif (> 20 %), refaire tourner la validation finale avec les vrais coûts.

### Recommandation 2 — Corriger la méthodologie du benchmark Monte Carlo

**Priorité** : 🔴 CRITIQUE

Le P95 = 9.96 est probablement irréaliste. Investiguer :
- Le calcul est-il fait in-sample (train ≤ 2022) ou OOS (test ≥ 2024) ?
- Les signaux aléatoires respectent-ils la même distribution de fréquence ?
- Le nombre de simulations Monte Carlo est-il suffisant (≥ 1000) ?
- Comparer avec des benchmarks académiques : le P95 typique d'un portefeuille multi-actifs est autour de Sharpe 2-3, pas 10.

### Recommandation 3 — Refaire la validation avec des coûts majorés de 20 %

**Priorité** : 🟠 IMPORTANT

Simuler +20 % de spread/slippage pour tester la robustesse. Si le Sharpe portefeuille reste ≥ 1.0 et DSR > 0, le portefeuille est robuste à la dégradation des conditions de marché.

### Recommandation 4 — Combiner en portefeuille réaliste

**Priorité** : 🟠 IMPORTANT

- Remplacer l'equal-weight par du risk parity (allocation inversement proportionnelle à la volatilité de chaque sleeve).
- Ajouter une contrainte de corrélation maximum entre sleeves (max 0.7).
- Tester avec sizing au risque (vol targeting) plutôt que risque fixe par trade.
- Le portefeuille actuel combine GBPUSD D1 + GBPUSD H4 (même actif, timeframes différents) — la corrélation est probablement > 0.8.

---

**Fin du document d'audit.**
