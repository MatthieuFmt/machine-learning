# Projet : Bot de Trading ML — CFD XTB → Alerte Telegram

**Dernière mise à jour** : 2026-08-22 (audit externe. **Verdict : (B) NOTHING TESTABLE** — les critères d'acceptation et la quantité de données OOS sont MATHÉMATIQUEMENT incompatibles, voir §2bis. Fondations réparées : données restaurées, coûts mesurés ENFIN écrits dans le code, garde anti-coût-estimé, sizing au risque, diagnostic de puissance OOS.)
**Mise à jour précédente** : 2026-08-06 (re-mesure finale sur bougies des 2 derniers candidats : **pre-FOMC et pre-ECB = NO-GO**. Le projet n'a plus AUCUN signal survivant. Données H1 + calendrier économique désormais VERSIONNÉS → les screens tournent en session.)
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

**Les 4 candidats sont désormais tous tombés (2026-08-06)** — verdict complet dans `JOURNAL.md` §2026-08-06 et bandeau de `docs/signaux_reels_phase1.md` :

| Candidat | Verdict | Cause de mort |
|---|---|---|
| ORB US500 M5 | ☠️ MORT | t = 0.56 · p = 0.287 → bruit (artefact « DSR ×√252 ») |
| Carry JPY | ☠️ MORT | swap réel **+0.16 %/an** au lieu de +0.7 à +3 % supposés |
| Tendance crypto | ☠️ MORTE | financement XTB **mesuré à 35.4 %/an** (seuil de viabilité : 10 %) |
| **Pre-FOMC US500** | ❌ **NO-GO** | t = 2.11 (p=0.018) MAIS **médiane/trade négative (−16.2 pips)**, 76 % du gain sur 5 trades, kurtosis 4.46 ; le meilleur trade est le **2020-03-03 = réunion d'URGENCE COVID non programmée** (20 % du gain) → hors elle, t = 1.84 |
| **Pre-FOMC US30** | ⏸️ **NON CONCLUABLE** | repose sur `spread_pips=1.5` **jamais relevé** ; une erreur ×9.1 annule le résultat, et l'erreur mesurée sur GER30 était ×9.2 → **capture app XTB requise, ne PAS ré-estimer** |
| **Pre-ECB GER30** | ❌ **NO-GO** | t = 0.53 · p = 0.298, coûts DE40 mesurés → rien |

⚠️ **Le test de décroissance pré/post-2015 du pre-FOMC est STRUCTURELLEMENT impossible sur ces données** : les prix commencent en 2012-01, l'étude d'origine (Lucca & Moench) portait sur 1994-2011 → aucune période pré-publication dans l'échantillon. Le résultat affiché (−26.6 → +69.4 pips) n'est donc PAS une validation : année par année, **2012-2019 est plat à négatif** et tout le gain vient de 2020/2022/2024 (années très volatiles).

✅ **Contrôle ajouté** (absent des screens) : test **placebo** contre 20 000 fenêtres de 23 h tirées au hasard — la stratégie étant *toujours longue*, il fallait vérifier qu'elle ne capture pas que du beta. Elle bat le hasard (p = 0.041 US500 / 0.036 US30), mais trop faiblement pour engager du capital. **À reproduire pour toute future stratégie directionnelle.**

✅ **Calendrier FOMC vérifié** (`scripts/verify_fomc_calendar.py`) : 0 doublon, 0 manquant sur 2010-2018. Les écarts s'expliquent — la réunion du 2020-03-18 a été **annulée** (COVID), donc **le calendrier local a raison et c'est la liste de référence du script qui est fausse**. L'angle mort redouté n'existe pas.

> 🚨 **`prompts/00_constitution.md` §1 affirme que la base « a trouvé un edge réel (Donchian +8.84) ». C'EST FAUX.** Voir le bloc de correction en tête de ce fichier. Toute décision bâtie sur cette prémisse est à reconsidérer.

**Ce qui reste vrai :** l'infrastructure (features, métriques, DSR, coûts XTB) est de bonne facture *en isolation*. Ce qui est cassé, c'est la **chaîne de validation** (fuites + data-snooping) et **l'absence totale de l'étage temps réel**. Voir le backlog §6.

**État de la recherche (2026-08-06)** : 10 familles testées honnêtement (2026-05-29→06-01) → mortes. Les 3 « signaux réels » qui avaient survécu ont été re-mesurés avec la pile corrigée → **tous tombés** (tableau ci-dessus). Le pre-ECB, dernier test de généralisation, est également NO-GO. ➡️ **Le projet est à ZÉRO stratégie validée, et cette fois les mesures sont propres** (coûts XTB relevés à l'écran, DSR canonique, t-test primaire, registre n_trials, calendrier vérifié).

**Seul point encore ouvert** : le verdict US30 attend une **capture de l'app XTB** (spread, valeur du pip, valeur du contrat + taille de lot, swap Achat/Vente, commission). Même s'il survivait : Sharpe 0.73 < 1.0 et **~8 trades/an** → la contrainte de déploiement du 2026-08-01 (« 8 trades/an ne peut pas être un moteur ») s'applique telle quelle. Ce serait au mieux un complément, jamais un socle.

**Leçon transverse la plus coûteuse du projet** : sur 5 estimations de coût confrontées à un relevé réel, **5 étaient fausses, toujours dans le sens qui arrangeait l'hypothèse testée** (spreads US500 ×15, GER30 ×9.2, BTCUSD ×6.3 ; swap crypto ×3.8 ; swap carry JPY jusqu'au **signe inverse**). ➡️ **Tout `spread_pips`/`swap_*` non relevé à l'écran doit être traité comme FAUX jusqu'à mesure.** Ne jamais estimer un coût : demander la capture.

La **stratégie manuelle** TradingView (`strategie-forex/`) est portée en module backtestable (`app/strategies/trend_pullback.py` + `scripts/screen_trend_pullback.py`) — NO-GO attendu (famille morte 10×), l'objectif est de donner un CHIFFRE au mainteneur.

---

## 2bis. ⛔ LE VERROU MATHÉMATIQUE (audit 2026-08-22) — À LIRE AVANT DE PROPOSER UNE HYPOTHÈSE

**Le goulot d'étranglement n'est PAS le manque d'hypothèses. C'est le manque de
temps hors-échantillon.**

En résolvant `SR_période × √(n_obs−1) − z_mix ≥ 1.645` pour un Sharpe annualisé
de exactement 1.0 (le critère GO), le nombre d'observations OOS nécessaires pour
franchir `DSR > 0 (p<0.05)` est :

| n_trials | obs. requises | années de données OOS |
|---|---|---|
| 1 — comptabilité la plus généreuse possible | 683 | **2.7** |
| 88 — registre mécanique actuel | 4 300 | **17.1** |
| ~1500 — compte honnête (la seule Phase G = 1 404 backtests) | 6 335 | **25.1** |

**La fenêtre encore vierge (2026-01-01 → 2026-05-19) fait 0.38 an.** Même à
n_trials = 1 elle est **7× trop courte** — le constat ne dépend donc d'aucune
convention de comptage.

Deux corollaires :
1. **Le gate DSR est infranchissable par construction.** Il n'a jamais
   discriminé quoi que ce soit. (La plupart des 45 morts échouaient AUSSI sur
   Sharpe < 1.0 : le cimetière reste largement réel.)
2. `deflated_sharpe` renvoie NaN si `n_obs < 30`, et `n_obs = n_trades − 1`.
   Obtenir un DSR non-NaN sur la fenêtre vierge exige **> 82 trades/an** — or à
   ce rythme la friction dépasse tout edge brut plausible. **Puissance et coût
   tirent en sens opposés, et il n'existe aucun N où les deux tiennent.**

➡️ **Avant de consommer une fenêtre OOS, appeler
`app.analysis.edge_validation.oos_power_report(n_obs, n_trials)`.** Une fenêtre
sous-puissante ne peut que RÉFUTER, jamais CONFIRMER — et la lire coûte quand
même un essai, ce qui durcit définitivement le seuil suivant.

**Second verrou, structurel** : swap long US500 mesuré = −0.021 %/nuit × 365 =
**−7.7 %/an**, contre ~8 %/an de rendement prix du S&P. **Le CFD confisque la
prime de risque actions.** « Multi-day mort », « crypto trend mort » et « carry
mort » ne sont pas trois résultats mais trois instances de ce seul fait. Toute
stratégie viable doit donc soit tenir très peu de temps, soit être short, soit
porter sur un instrument à carry neutre — le forex est le seul coin du panier
où le financement ne taxe pas la position.

---

## 2ter. Les deux derniers certificats de décès (2026-08-22)

- **`trend_pullback` (stratégie MANUELLE du mainteneur)** — lancée pour la
  PREMIÈRE fois sur données réelles. ❌ **NO-GO sur les 4 actifs**, espérance
  NÉGATIVE par trade partout (−4.4 à −17.7 pips), 11–15 trades/an contre un
  plancher de 30, stable avant/après 2020. Ce n'est pas une dégradation de
  régime : la règle n'a jamais eu d'edge. Dossier clos.

- **NR7 / `volatility_breakout` US500 D1** — ⚠️ **NI VIVANT NI MORT.**
  Le « DSR +2.09 (p=0.0183) » de `run_h3_us500_validate_edge.py` est FAUX :
  equity quotidienne (864 obs) pour 65 trades → n_obs gonflé ×13, et
  `N_TRIALS_CUMUL` codé en dur. Corrigé. DSR honnête PAR TRADE : z=+1.3 à
  +1.5, p=0.06–0.10 → **échoue**.
  MAIS la preuve primaire est la plus solide du projet, au coût MESURÉ :
  **t=3.53 (p=0.0004)**, médiane **+55.5 pips POSITIVE**, kurtosis 2.62,
  5 meilleurs trades = 44 % du PnL, WR 64.6 % — il réussit les trois contrôles
  que le pre-FOMC avait échoués.
  ➡️ Inconfirmable : la fenêtre ≥2024 est brûlée, il faudrait ~474 observations
  là où il y en a 65, et `MaxDD 0.5 %` ne mesure rien (equity à 1 lot en dur).
  **Ne PAS le déclarer vivant. Ne PAS le déclarer mort.** C'est l'illustration
  exacte du §2bis.

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

- ✅ **Les données des screens événementiels sont VERSIONNÉES depuis 2026-08-06**
  (~19.5 Mo) → **`screen_pre_fomc` et `screen_pre_ecb` tournent EN SESSION**,
  sans dépendre du PC du mainteneur :
  `data/raw/US500/US500_H1.csv` · `data/raw/US30/US30_H1.csv` ·
  `data/raw/GER30/GER30_H1.csv` (tous 2012 → 2026-05-19, ~80k barres) et
  `data/raw/economic_calendar/2010..2025.csv` + `data/vendor/`.
  Les exceptions `.gitignore` correspondantes sont écrites en dur — **ne pas les
  casser** (piège : git ne descend jamais dans un dossier exclu, d'où le
  `/data/*` répété à chaque niveau plutôt que `/data/`).
- ⚠️ **Le calendrier économique s'arrête au 2025-12-31** alors que les prix vont
  jusqu'à 2026-05 → **aucun événement 2026 n'est testable**, donc l'OOS vierge
  ≥2026 est INUTILISABLE pour tout screen événementiel. Il faudrait un
  `data/raw/economic_calendar/2026.csv`.
- **Le RESTE des données reste hors repo** (US500 M5 ~704k bougies, EURUSD_H4, …)
  → les screens qui en dépendent (ORB M5, carry, pairs) tournent chez le mainteneur.
- **`app/data/` est COMMITÉ et fonctionnel** depuis 2026-05-29 (loader + registry).
- ⚠️ **Format des CSV de prix : séparateur TABULATION**, colonnes
  `Time Open High Low Close Volume` (`load_asset` le gère ; un `pd.read_csv`
  naïf en virgule renvoie une seule colonne et des dates NaT).
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
   - ℹ️ **Corollaire pour le t-test** (constaté 2026-08-06) : `validate_edge` teste `equity.pct_change()`, donc des rendements **en % d'un compte qui capitalise** — les trades tardifs pèsent moins quand l'equity a monté. Un t-test sur les **pips bruts à poids égal** donne un chiffre différent (US30 : 2.48 vs 2.68 ; US500 : 2.11 vs 2.10, l'equity ayant peu bougé). Les deux sont corrects mais ne répondent pas à la même question — **toujours préciser lequel est cité.**
6. **Anti-look-ahead testé** : pour chaque feature, `feature(df[:n])[-1] == feature(df)[n-1]`. Le décorateur `look_ahead_safe` doit *vérifier*, pas juste marquer.
7. **Résolution intrabar conservatrice** : si TP et SL touchés dans la même barre → **SL gagne** (pas TP).
8. **Un seul regard par hypothèse OOS.** Réagir au résultat = data-snooping = nouvelle hypothèse.

Critères GO (constitution §2, maintenus) : Sharpe WF ≥ 1.0 · DSR > 0 (p<0.05) · MaxDD < 15 % · WR > 30 % · ≥ 30 trades/an. **Tous, sur OOS vierge.**

---

## 6. Backlog de bugs CRITIQUES (audits 2026-05-29, 2026-06-09 et 2026-08-06)

Priorité décroissante. Chacun fausse les résultats ou casse le pipeline.

**Données d'événements (audit 2026-08-06)**
- 🔴 **Réunions FOMC NON PROGRAMMÉES incluses dans le screen pre-FOMC.**
  `load_fomc_announcement_times` filtre sur `event == "FOMC Statement"` sans
  distinguer les réunions **programmées** des décisions d'**urgence**. Les 2
  baisses COVID (**2020-03-03** et **2020-03-15**) passent donc dans le backtest
  alors qu'elles n'ont, par construction, **aucune fenêtre d'anticipation**.
  Impact mesuré : le 2020-03-03 est le **meilleur trade des DEUX actifs**
  (+1096 pips US500 / +1074 US30) = **20-22 % du gain total** ; hors elle,
  US500 passe de t=2.10 à **t=1.84**. ➜ ajouter un filtre `scheduled`.
- 🔴 **Liste de référence de `scripts/verify_fomc_calendar.py` fausse** :
  elle contient `2020-03-18` (réunion **annulée**) et omet les 2 réunions
  d'urgence. Le calendrier LOCAL a raison, le script signale un faux positif.
- ⚠️ **Aucun test placebo dans les screens directionnels.** Une stratégie
  *toujours longue* sur un indice haussier gagne sans aucun edge. Contrôle à
  généraliser : comparer à N tirages de fenêtres de même durée prises au hasard
  sur la même période, mêmes coûts (fait à la main le 2026-08-06 :
  p=0.041 US500 / 0.036 US30).

**Coûts (la cause de mort n°1 du projet)**
- 🔴 **`ASSET_CONFIGS["US30"].spread_pips = 1.5` n'a JAMAIS été relevé.**
  C'est le dernier coût estimé qui porte encore un verdict. Sensibilité mesurée :
  une erreur **×9.1** annule le résultat pre-FOMC US30 — l'erreur constatée sur
  GER30 était **×9.2**. ➜ **capture app XTB requise** (spread, valeur du pip,
  valeur du contrat + taille de lot, swap Achat/Vente, commission).
  **Interdiction de ré-estimer.** Voir `docs/checklist_couts_xtb.md`.
- ⚠️ US500 et GER30 : spreads relevés en **pré-ouverture** (= pire cas, donc
  conservateur et acceptable). 🔜 raffiner par un relevé **en séance**.

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
