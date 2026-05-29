# Projet : Bot de Trading ML — CFD XTB → Alerte Telegram

**Dernière mise à jour** : 2026-05-29 (refonte « bases saines » — audit complet v6+)
**Mainteneur** : matthieu.fremont12@gmail.com

> ⚠️ **CE FICHIER REMPLACE l'ancien `CLAUDE.md` (daté 2026-05-12, « EURUSD H1 / roadmap 7 steps »), qui était OBSOLÈTE.**
> L'ancienne architecture « v1 » (RF supervisé sur EURUSD H1, modes `triple_barrier`/`forward_return`/…) **n'existe plus** dans le code. Le repo réel est un pipeline méta-labeling multi-actifs (voir §3).

---

## 1. Mission

Construire un bot qui :
1. **Analyse des CFD disponibles chez XTB** (indices, métaux, énergies, forex, crypto).
2. **Détecte un edge statistiquement valide** (DSR > 0, p < 0.05 sur un OOS jamais consulté).
3. **Envoie une alerte Telegram** (entrée / SL / TP / taille) quand un bon trade potentiel apparaît.
4. Tourne en continu sur une pipeline (VPS / Docker / GitHub Actions).

**Ordre de priorité décidé par le mainteneur (2026-05-29) : trouver un VRAI edge AVANT tout déploiement.** Pas de bot d'alerte tant qu'une stratégie n'a pas survécu à une validation propre.

---

## 2. ⚠️ STATUT RÉEL — À LIRE AVANT TOUT

**Aucun edge statistiquement valide n'a jamais été trouvé à ce jour.** C'est le point de départ honnête, établi par les propres post-mortems du projet (`docs/audit_final_post_mortem.md`, `docs/diagnostic_final_donchian_dead.md`, `docs/archive_v1/step_08_*`).

Les deux seuls résultats « positifs » jamais affichés étaient des **artefacts de bugs** :

| « Résultat » affiché | Réalité après correction des bugs |
|---|---|
| Donchian US30 D1 — **Sharpe walk-forward +8.84** (v2) | Artefact du simulateur « TP-prime » optimiste (bug F3) |
| Portfolio v4 — **Sharpe +4.97 / DSR 19.5 (p=0.000)** | Après fix F1/F2/F3 → **Sharpe −5.42 / DSR −18.3 (p=1.000)** |

> 🚨 **`prompts/00_constitution.md` §1 affirme que la base « a trouvé un edge réel (Donchian +8.84) ». C'EST FAUX.** Voir le bloc de correction en tête de ce fichier. Toute décision bâtie sur cette prémisse est à reconsidérer.

**Ce qui reste vrai :** l'infrastructure (features, métriques, DSR, coûts XTB) est de bonne facture *en isolation*. Ce qui est cassé, c'est la **chaîne de validation** (fuites + data-snooping) et **l'absence totale de l'étage temps réel**. Voir le backlog §6.

**Pistes les moins désespérées** (jamais validées proprement) : actifs tendanciels **crypto (ETH/BTC) et or** en **D1/H4** (pas H1), avec **filtrage de régime** ; familles non explorées : carry JPY, ORB/Asian range, pairs trading/cointégration, pre-FOMC drift.

---

## 3. Architecture RÉELLE

```
app/
├── config/         # backtest, calendar, instruments (ASSET_CONFIGS = coûts XTB),
│                   # features_selected, hyperparams_tuned, model_selected, models, ml_pipeline_v4
├── core/           # exceptions, logging, retry, seeds, types
├── data/           # ⛔ MANQUANT — jamais commité. loader/registry attendus par
│                   #    app/features/research.py, superset.py + ~45 scripts (imports CASSÉS)
├── features/       # indicators, regime, macro_external, calendar, economic,
│                   # ranking, research, superset
├── targets/        # labels.py (triple_barrier, meta-labels)
├── models/         # meta_rf, meta_labeling(_pipeline), build, candidates,
│                   # cpcv_evaluation, nested_tuning
├── backtest/       # simulator, deterministic, meta_labeling, cpcv, walk_forward,
│                   # grid_search, metrics, filters, sizing
├── strategies/     # 17 stratégies : donchian, bollinger, keltner, chandelier, dual_ma,
│                   # sma_crossover, ts_momentum, mean_reversion, rsi_contrarian,
│                   # volatility_breakout, parabolic, asian_range, nr7_meta,
│                   # pairs_trading, pre_fomc_drift, pre_fomc_meta
├── analysis/       # edge_validation.py (DSR, PSR, bootstrap, walk-forward, purged k-fold)
├── portfolio/      # constructor.py
├── pipelines/      # base, us30, xauusd, walk_forward, walk_forward_rolling
└── testing/        # look_ahead_validator, snooping_guard  (⚠️ actuellement no-op, voir §6)

scripts/            # ~60 scripts run_*/diagnose_*/screen_* (exploration historique)
prompts/            # 00_constitution → 24 (specs ; 20-24 = étage live XTB/Telegram, NON codé)
docs/               # historique v1→v6, post-mortems, audits  (beaucoup de docs contradictoires)
tests/              # unit/ integration/  (pytest installé à la demande)
```

---

## 4. Contraintes d'environnement (session cloud)

- **Aucune donnée dans le repo** (`data/` vide). Les backtests **ne peuvent pas tourner** sans re-télécharger.
- **`app/data/` jamais commité** → imports cassés. À restaurer (loader + registry + downloader Dukascopy).
- **PyPI accessible** (pip install OK) ; **Dukascopy/Yahoo en 403** sur appel direct (à valider via lib).
- Format de données attendu (constitution) : `data/raw/<ASSET>/<TF>.csv`, index UTC tz-aware.
- ➜ **Le travail de fondation (code/docs/tests) se fait ici ; la recherche empirique d'edge nécessite que le mainteneur fournisse/télécharge les données.**

---

## 5. Protocole anti-fuite & anti-snooping (NON NÉGOCIABLE)

Les règles ci-dessous existaient déjà (constitution) mais ont été **violées en pratique**. Désormais elles doivent être **mécaniquement appliquées**, pas seulement écrites.

1. **OOS vierge.** L'ancien test ≥2024 est **BRÛLÉ** (48+ scripts l'ont consulté). Un edge ne compte que sur une période **jamais touchée** (ex. ≥2026), verrouillée dans `TEST_SET_LOCK.json`. Le garde `snooping_guard` doit **réellement tourner** (aujourd'hui no-op).
2. **`n_trials` du DSR = compteur automatique cumulé**, jamais une constante en dur. Chaque config/seuil/grid/actif testé = +1 essai.
3. **Labels calculés PAR FOLD**, jamais sur le dataset entier (sinon fuite train→test).
4. **Embargo + purge ≥ horizon de la cible, des DEUX côtés** du test.
5. **Une seule définition de Sharpe** : sur retours quotidiens (`equity.pct_change()`), annualisé `√252`. Jamais `Sharpe_per_trade × √n`.
6. **Anti-look-ahead testé** : pour chaque feature, `feature(df[:n])[-1] == feature(df)[n-1]`. Le décorateur `look_ahead_safe` doit *vérifier*, pas juste marquer.
7. **Résolution intrabar conservatrice** : si TP et SL touchés dans la même barre → **SL gagne** (pas TP).
8. **Un seul regard par hypothèse OOS.** Réagir au résultat = data-snooping = nouvelle hypothèse.

Critères GO (constitution §2, maintenus) : Sharpe WF ≥ 1.0 · DSR > 0 (p<0.05) · MaxDD < 15 % · WR > 30 % · ≥ 30 trades/an. **Tous, sur OOS vierge.**

---

## 6. Backlog de bugs CRITIQUES (issu de l'audit 2026-05-29)

Priorité décroissante. Chacun fausse les résultats ou casse le pipeline.

**Validation / fuites (invalident tout edge passé)**
- `run_meta_labeling_cpcv.py:253` — CPCV final sur train+val+test **fusionnés** (C1).
- `meta_labeling.py:147` — meta-labels calculés sur dataset entier avant split (C2).
- `app/models/cpcv_evaluation.py:38-61` — k-fold **non causal** (train inclut barres post-test) + Sharpe `×√n_trades` (C3, M2).
- `walk_forward.py:174` — embargo 2j < horizon 120h (H2).
- `meta_labeling.py:116` / `targets/labels.py:222` — intrabar « TP gagne » optimiste (H1).

**Anti-snooping (crédibilité statistique)**
- `scripts/verify_no_snooping.py:22` — **no-op** (pas de `TEST_SET_LOCK.json`) (C5).
- `app/testing/look_ahead_validator.py:50` — décorateur **cosmétique**, ne vérifie rien (H3).
- `n_trials` codé en dur et incohérent (4/5/23/29/62…) alors que ~48 scripts touchent l'OOS (C4).

**Backtest (edge gonflé)**
- `app/pipelines/base.py:156` — `asset_cfg` **jamais passé** → swap overnight + sizing au risque **jamais exécutés** (C1-bt).
- `app/backtest/simulator.py:79` — entrée au **close de la barre de signal** = look-ahead d'exécution (corriger → `open[i+1]`) (C2-bt).
- `simulator.py` — SL/TP rempli au prix exact, **aucun stop-slippage/gap** ; friction 1.5 pip optimiste vs ~2-3 pips round-trip XTB réel.
- `metrics.py:139` — resample daily + `ffill` écrase la variance → **Sharpe gonflé** (E4).

**Infra**
- `app/data/` manquant → restaurer (loader/registry/downloader).
- Docs contradictoires (v1→v6) → consolider en 1 source de vérité ; `prompts/00` §1 à corriger.

---

## 7. Conventions

- **Python 3.12** — `from __future__ import annotations` partout ; mypy `--strict`, pas de `Any` hors `Protocol`.
- **Vectorisation pandas** — zéro boucle sur les rows ; `.shift()`/`.rolling()`/`.where()`.
- **Logging** — `get_logger(__name__)` par module.
- **Tests** — < 100 ms/test unitaire, fixtures synthétiques, pas d'I/O.
- **RTK** — préfixer `rtk ` toute commande à sortie longue (>20 lignes) : `rtk pytest …`, `rtk python run_*.py`.
- **Langue** — code en anglais, docs en français.
- **Pas de commit/push ni d'exécution `python run_*`/`pytest` sans accord explicite** (constitution Règles 2 & 3).
- **Dépendances** : numpy, scipy, pandas>=3, pandas-ta, scikit-learn>=1.8, statsmodels, numba, matplotlib, tqdm, colorama, dukascopy-python, yfinance. Pas de LightGBM/XGBoost/PyTorch.

---

## 8. Où regarder pour comprendre l'histoire

- `docs/audit_final_post_mortem.md` — post-mortem v4 (14 manques).
- `docs/diagnostic_final_donchian_dead.md` — pourquoi Donchian est mort (bugs F1/F2/F3).
- `docs/archive_v1/step_08_postmortem_and_v2_roadmap.md` — fin de la lignée EURUSD H1.
- `docs/audit_v6_what_we_missed.md` — angles morts (swap absent, espace de recherche <15 %).
- `docs/regime_analysis.md` — quels actifs sont tendanciels (crypto/or) vs range (forex/indices).
