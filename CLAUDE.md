# Projet : Bot de Trading ML — CFD XTB → Alerte Telegram

**Dernière mise à jour** : 2026-06-09 (audit indépendant : bug « DSR ×√252 » découvert et CORRIGÉ ; les 3 « signaux réels » de `docs/signaux_reels_phase1.md` sont SUSPENDUS en attente de re-mesure ; stratégie manuelle TradingView portée en backtestable)
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

**Ordre de priorité décidé par le mainteneur (2026-05-29) : trouver un VRAI edge AVANT tout déploiement.** Pas de bot d'alerte tant qu'une stratégie n'a pas survécu à une validation propre. Le mainteneur est ouvert à des stratégies **non-ML** (le ML n'a jamais aidé ici) : priorité aux stratégies simples, robustes, peu paramétrées.

➡️ **Pour lancer la recherche d'edge : voir `docs/HOWTO_recherche_edge.md`** (guide débutant) + `scripts/screen_edge.py` (CLI) qui s'appuie sur `app/research/edge_harness.py`.

---

## 2. ⚠️ STATUT RÉEL — À LIRE AVANT TOUT

**Aucun edge statistiquement valide n'a jamais été trouvé à ce jour.** C'est le point de départ honnête, établi par les propres post-mortems du projet (`docs/audit_final_post_mortem.md`, `docs/diagnostic_final_donchian_dead.md`, `docs/archive_v1/step_08_*`).

Les TROIS seuls résultats « positifs » jamais affichés étaient des **artefacts de bugs** :

| « Résultat » affiché | Réalité après correction des bugs |
|---|---|
| Donchian US30 D1 — **Sharpe walk-forward +8.84** (v2) | Artefact du simulateur « TP-prime » optimiste (bug F3) |
| Portfolio v4 — **Sharpe +4.97 / DSR 19.5 (p=0.000)** | Après fix F1/F2/F3 → **Sharpe −5.42 / DSR −18.3 (p=1.000)** |
| ORB US500 M5 — **DSR +11.29 (p=0.000)** avec Sharpe 0.17 (2026-05-30) | Artefact du bug « DSR ×√252 » (corrigé 2026-06-09) : Sharpe/trade ≈ 0.011 → **z ≈ 0.6, p ≈ 0.27 = bruit** |

> 🚨 **`prompts/00_constitution.md` §1 affirme que la base « a trouvé un edge réel (Donchian +8.84) ». C'EST FAUX.** Voir le bloc de correction en tête de ce fichier. Toute décision bâtie sur cette prémisse est à reconsidérer.

**Ce qui reste vrai :** l'infrastructure (features, métriques, DSR, coûts XTB) est de bonne facture *en isolation*. Ce qui est cassé, c'est la **chaîne de validation** (fuites + data-snooping) et **l'absence totale de l'étage temps réel**. Voir le backlog §6.

**État de la recherche (2026-06-09)** : 10 familles testées honnêtement en local (2026-05-29→06-01) → mortes, sauf 3 signaux retenus dans `docs/signaux_reels_phase1.md` (pre-FOMC US500, carry JPY, ORB M5). **Leurs DSR sont CADUCS** (bug « DSR ×√252 ») → statuts **SUSPENDUS** ; re-mesure en local avec la pile corrigée = prochaine étape obligatoire. Candidat le plus crédible : **pre-FOMC** (effet documenté). Carry = dépend des swaps réels XTB (provisoires → `docs/checklist_couts_xtb.md`). La **stratégie manuelle** TradingView (`strategie-forex/`) est portée en module backtestable (`app/strategies/trend_pullback.py` + `scripts/screen_trend_pullback.py`) — NO-GO attendu (famille morte 10×), l'objectif est de donner un CHIFFRE au mainteneur.

---

## 3. Architecture RÉELLE

```
app/
├── config/         # backtest, calendar, instruments (ASSET_CONFIGS = coûts XTB,
│                   # 15 actifs dont JPY ; swaps JPY/crypto PROVISOIRES),
│                   # features_selected, hyperparams_tuned, model_selected, models, ml_pipeline_v4
├── core/           # exceptions, logging, retry, seeds, types
├── data/           # ✅ RESTAURÉ (2026-05-29) : loader.py (load_asset), registry.py
│                   #    (discover_assets). Les CSV restent hors repo (machine du mainteneur).
├── features/       # indicators, regime, macro_external, calendar, economic,
│                   # ranking, research, superset
├── targets/        # labels.py (triple_barrier, meta-labels)
├── models/         # meta_rf, meta_labeling(_pipeline), build, candidates,
│                   # cpcv_evaluation, nested_tuning
├── backtest/       # simulator, deterministic, meta_labeling, cpcv, walk_forward,
│                   # grid_search, metrics, filters, sizing
├── strategies/     # 23 stratégies : donchian, bollinger, keltner, chandelier, dual_ma,
│                   # sma_crossover, ts_momentum, mean_reversion, rsi_contrarian,
│                   # volatility_breakout, parabolic, asian_range, nr7_meta,
│                   # pairs_trading, pre_fomc_drift, pre_fomc_meta, opening_range,
│                   # gap_fade, crypto_trend, turn_of_month, trend_pullback (= stratégie manuelle)
├── analysis/       # edge_validation.py (DSR canonique, PSR, bootstrap, t-test, purged k-fold)
├── research/       # ⭐ edge_harness.py — POINT D'ENTRÉE Phase 1 (backtest honnête +
│                   #    split IS/OOS gelé + DSR n_trials auto + record_and_resolve_n_trials
│                   #    pour les screens). CLI: scripts/screen_edge.py
├── portfolio/      # constructor.py
├── pipelines/      # base, us30, xauusd, walk_forward, walk_forward_rolling
└── testing/        # look_ahead_validator (cosmétique), snooping_guard (✅ opérant,
                    #   branché dans edge_harness + tous les screens Phase 1)

scripts/            # ~70 scripts ; les SCREENS Phase 1 actifs : screen_pre_fomc,
                    #   screen_orb(_fine), screen_carry(_voltarget), screen_crypto_trend,
                    #   screen_turn_of_month, screen_event_drift, screen_gap_fade,
                    #   screen_asian_range, screen_pairs, screen_trend_pullback, screen_edge
strategie-forex/    # stratégie MANUELLE TradingView (2 indicateurs Pine v1 + v2 améliorés
                    #   + strategie_backtest.pine + guide HTML) — cf. README.md du dossier
prompts/            # 00_constitution → 24 (specs ; 20-24 = étage live XTB/Telegram, NON codé)
docs/               # historique v1→v6, post-mortems, audits  (beaucoup de docs contradictoires)
tests/              # unit/ integration/  (pytest installé à la demande)
```

---

## 4. Contraintes d'environnement (session cloud)

- **Aucune donnée dans le repo** (`data/` absent du repo). Les données complètes
  (Dukascopy 2010→2026-05, dont US500 M5 ~704k bougies) sont **sur la machine
  locale du mainteneur** → les screens empiriques tournent CHEZ LUI, pas ici.
- **`app/data/` est COMMITÉ et fonctionnel** depuis 2026-05-29 (loader + registry).
- **PyPI accessible** (pip install OK) ; **Dukascopy/Yahoo en 403** sur appel direct depuis le cloud.
- Format de données : `data/raw/<ASSET>/<*>_<TF>.csv` (ex. `EURUSD_H4.csv`), index UTC tz-aware.
- `TEST_SET_LOCK.json` (registre anti-snooping) est **local et gitignoré** : le
  compteur n_trials de référence vit sur la machine du mainteneur.
- ➜ **Le travail de fondation (code/docs/tests) se fait ici ; la recherche empirique d'edge tourne en local.**

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

## 6. Backlog de bugs CRITIQUES (audits 2026-05-29 et 2026-06-09)

Priorité décroissante. Chacun fausse les résultats ou casse le pipeline.

**Statistique (verdicts faussés)**
- ✅ **« DSR ×√252 » (2026-06-09)** — `validate_edge` passait le Sharpe ANNUALISÉ
  au DSR avec `n_obs` = nb de trades → z gonflé jusqu'à ×√252. Fabriquait l'« ORB
  DSR +11.29 ». CORRIGÉ : DSR canonique par-période (`deflated_sharpe`), SR₀ à
  l'échelle σ_SR, + t-test par trade et bootstrap stationnaire dans le rapport.
  Même bug corrigé dans `screen_carry._metrics` (réutilisé par carry_voltarget
  et crypto_trend). **Toute valeur de DSR antérieure au 2026-06-09 est caduque.**
- ✅ `n_trials` des screens autonomes — était `len(assets)` local ; désormais
  cumul du registre via `record_and_resolve_n_trials` (edge_harness), branché
  dans les 11 screens Phase 1 + screen_trend_pullback (C4 étendu).

**Validation / fuites (invalident tout edge ML passé)**
- `run_meta_labeling_cpcv.py:253` — CPCV final sur train+val+test **fusionnés** (C1).
- `meta_labeling.py:147` — meta-labels calculés sur dataset entier avant split (C2).
- `app/models/cpcv_evaluation.py:38-61` — k-fold **non causal** (train inclut barres post-test) + Sharpe `×√n_trades` (C3, M2).
- `walk_forward.py:174` — embargo 2j < horizon 120h (H2).
- `meta_labeling.py:116` / `targets/labels.py:222` — intrabar « TP gagne » optimiste (H1).

**Anti-snooping (crédibilité statistique)**
- ✅ `snooping_guard` — registre opérant (read_oos/n_unique_hypotheses), branché
  dans edge_harness ET les screens. `TEST_SET_LOCK.json` local/gitignoré.
- `app/testing/look_ahead_validator.py:50` — décorateur **cosmétique**, ne vérifie rien (H3).

**Backtest (edge gonflé)**
- `app/pipelines/base.py:156` — `asset_cfg` **jamais passé** → swap overnight + sizing au risque **jamais exécutés** dans le chemin ML (C1-bt). *(NB : le chemin déterministe `deterministic.py` applique déjà le swap.)*
- ✅ `deterministic.py` — fill honnête `entry_on_next_open` ajouté (entrée `open[i+1]`, scan depuis la barre d'entrée). **Défaut False (legacy)** ; la recherche d'edge Phase 1 DOIT passer True. *(`simulator.py:79` ML path reste à corriger.)*
- `simulator.py` / `deterministic.py` — SL/TP rempli au prix exact, **aucun stop-slippage/gap** ; friction optimiste vs réel (à traiter). *(NB : `trend_pullback.py` gère déjà le gap au-delà du SL → fill à l'open.)*
- ✅ `metrics.py` `sharpe_daily_from_trades` — annualisation **routée par fréquence** (daily/weekly/per-trade) → tue l'inflation basse-fréquence (E4).
- Swaps JPY/crypto **PROVISOIRES** dans `ASSET_CONFIGS` → relevés démo requis
  (`docs/checklist_couts_xtb.md`) ; d'ici là `--cost-margin 1.5` dans les screens.

**Infra**
- ✅ `app/data/` restauré (loader/registry). Downloader : `scripts/download_orb_data.py` (M5) + scripts Dukascopy historiques.
- Docs contradictoires (v1→v6) → consolider ; `prompts/00` §1 corrigé par bandeau ; fiches `strategies-doc/` marquées SUSPENDU/INVALIDÉ (2026-06-09).

---

## 7. Conventions

- **Python 3.12** — `from __future__ import annotations` partout ; mypy `--strict`, pas de `Any` hors `Protocol`.
- **Vectorisation pandas** — zéro boucle sur les rows ; `.shift()`/`.rolling()`/`.where()`.
- **Logging** — `get_logger(__name__)` par module.
- **Tests** — < 100 ms/test unitaire, fixtures synthétiques, pas d'I/O.
- **RTK** — préfixer `rtk ` toute commande à sortie longue (>20 lignes) : `rtk pytest …`, `rtk python run_*.py`.
- **Langue** — code en anglais, docs en français.
- **Style de réponse au mainteneur** — réponses COURTES et compréhensibles pour un débutant complet en trading (éviter le jargon non expliqué).
- **Pas de commit/push ni d'exécution `python run_*`/`pytest` sans accord explicite** (constitution Règles 2 & 3).
- **Dépendances** : numpy, scipy, pandas>=3, pandas-ta, scikit-learn>=1.8, statsmodels, numba, matplotlib, tqdm, colorama, dukascopy-python, yfinance. Pas de LightGBM/XGBoost/PyTorch.

---

## 8. Où regarder pour comprendre l'histoire

- `docs/audit_final_post_mortem.md` — post-mortem v4 (14 manques).
- `docs/diagnostic_final_donchian_dead.md` — pourquoi Donchian est mort (bugs F1/F2/F3).
- `docs/archive_v1/step_08_postmortem_and_v2_roadmap.md` — fin de la lignée EURUSD H1.
- `docs/audit_v6_what_we_missed.md` — angles morts (swap absent, espace de recherche <15 %).
- `docs/regime_analysis.md` — quels actifs sont tendanciels (crypto/or) vs range (forex/indices).
