# Prompt 00 — Constitution du projet (méta-prompt)

> **Ce document est le contrat de base entre toi (l'IA) et l'utilisateur.**
> **À RELIRE INTÉGRALEMENT au début de chaque session, avant d'exécuter tout autre prompt 01-24.**

---

## 1. Mission globale

Reconstruire un bot de trading ML qui :
1. Détecte un edge statistiquement fiable sur des CFD disponibles chez XTB (indices, métaux, énergies).
2. Génère un signal quotidien (ou selon le TF retenu) avec entrée, stop-loss, take-profit, taille de position.
3. Envoie une alerte Telegram avec ce signal + un prompt copier-coller pour validation humaine via une IA externe.
4. Tourne sur un VPS (ou Docker / GitHub Actions) à intervalle régulier.

**L'objectif n'est PAS de tout réinventer**. Le repo contient déjà une fondation v2 fonctionnelle (`learning_machine_learning_v2/`) qui a trouvé un edge réel (Donchian Breakout US30 D1 + méta-labeling RF, Sharpe walk-forward +8.84). Cette base sera renommée en `app/` au prompt 02 et étendue selon la roadmap v3 (`docs/v3_roadmap.md`).

---

## 2. Critères de succès — TOUS doivent passer avant production

| Critère | Valeur cible | Comment vérifier |
|---|---|---|
| Sharpe walk-forward portfolio | ≥ 1.0 | `python run_v3_walk_forward.py` → rapport `predictions/v3_final.json` |
| DSR (Deflated Sharpe Ratio) | > 0 avec p < 0.05 | Calculé via `analysis/edge_validation.py` |
| Drawdown maximum | < 15 % | Métrique de l'equity curve |
| Win rate | > 30 % | Métrique des trades |
| Trades par an (en moyenne) | ≥ 30 | Évite les stratégies à trop peu de trades |

**Aucun de ces critères n'est négociable.** Si un seul ne passe pas, on revient en arrière (cf. critères go/no-go de chaque prompt).

---

## 3. Paramètres figés du projet

| Paramètre | Valeur |
|---|---|
| Capital de référence | **10 000 €** (variable `CAPITAL_EUR` dans `.env`) |
| Risque par trade | **2 %** du capital (variable `RISK_PER_TRADE` dans `.env`) |
| Split temporel | train ≤ 2022, val = 2023, test ≥ 2024 (FIGÉ — ne jamais modifier) |
| Stratégie de référence | Donchian Breakout (validée H03 / H05) |
| Source données nouveaux actifs | CSV fournis par l'utilisateur dans `data/raw/<ASSET>/<TF>.csv` |
| Langue prompts et docs | **Français** |
| Langue code et identifiants | **Anglais** |
| Python | 3.12+, `from __future__ import annotations` partout |
| Typage | mypy `--strict`, pas de `Any` sauf dans les `Protocol` |

---

## 4. Dossiers et fichiers — règles d'accès

### 🚫 INTERDIT de lire (jamais, sous aucun prétexte)
- `data/`
- `ready-data/`
- `cleaned-data/`

Raison : ces dossiers contiennent des dizaines de milliers de lignes CSV qui pollueraient le contexte LLM. Pour connaître la structure de la donnée, lire UNIQUEMENT le contrat défini dans `app/data/loader.py` (créé au prompt 03).

### 📂 À utiliser comme références
- `docs/v3_roadmap.md` — source de vérité des hypothèses H06–H18
- `docs/v2_hypothesis_01.md` à `docs/v2_hypothesis_05.md` — historique v2 (à recycler dans `JOURNAL.md`)
- `ml_evolution.md` — changelog historique
- `JOURNAL.md` — journal d'exécution (créé au prompt 01, mis à jour à chaque prompt)
- `INVENTORY.md` — inventaire du repo (créé au prompt 01)

### 📂 À créer / modifier
Uniquement ce que le prompt en cours autorise. Tout fichier hors scope = refusé.

---

## 5. Règles d'exécution stricte

### Règle 1 — Lire avant d'agir
À chaque session :
1. Relire `prompts/00_constitution.md` (ce fichier).
2. Lire `JOURNAL.md` pour ne pas répéter une erreur passée.
3. Lire le prompt précédent (`prompts/<N-1>_*.md`) pour le contexte immédiat.

### Règle 2 — Pas d'exécution Python automatique
Tu ne lances **JAMAIS** `python run_*.py` ou `pytest` sans demande explicite de l'utilisateur. Tu peux écrire le code, mais l'exécution = décision humaine.

### Règle 3 — Pas de commit automatique
Tu ne fais **JAMAIS** `git commit` ou `git push` sans l'accord explicite de l'utilisateur.

### Règle 4 — Pas de scope creep
Si tu vois un problème hors du scope du prompt en cours, tu l'**ajoutes dans une section "Hypothèses à explorer ensuite"** du `JOURNAL.md`, tu ne le corriges pas.

### Règle 5 — RTK obligatoire pour commandes longues
Pour toute commande shell susceptible de produire > 20 lignes (`pytest`, `python run_*.py`, `ls -R`, `git log`, etc.), préfixer par `rtk ` :
- ❌ `pytest tests/ -v`
- ✅ `rtk pytest tests/ -v`
- ❌ `git log --oneline`
- ✅ `rtk git log --oneline`

### Règle 6 — Vectorisation pandas obligatoire
Zéro boucle Python sur les rows d'un DataFrame. Priorité absolue à `.shift()`, `.rolling()`, `.where()`, `.cumsum()`, etc. Une boucle row-by-row = code à refuser.

### Règle 7 — Anti-look-ahead
Toute feature à l'instant `t` n'utilise que l'information ≤ `t`. À chaque feature créée, ajouter un test pytest qui vérifie que `feature(df[:n])[-1] == feature(df)[n-1]`.

### Règle 8 — Logging obligatoire à la fin de CHAQUE prompt
À la fin de chaque session, ajouter une entrée dans `JOURNAL.md` :

```markdown
## YYYY-MM-DD — Prompt XX : <titre>
- **Statut** : ✅ Terminé / ⚠️ Partiel / ❌ Bloqué
- **Fichiers modifiés/créés** : liste
- **Résultats clés** : métriques chiffrées si applicable (Sharpe, DSR, etc.)
- **Problèmes rencontrés** : description courte
- **Hypothèses à explorer ensuite** : liste (sera utilisée pour les prompts futurs)
```

### Règle 9 — Un seul regard par hypothèse OOS
Quand tu obtiens un résultat OOS (test 2024-2025), tu le **lis une seule fois**. Tu ne modifies pas les paramètres en réaction (= data snooping). Si tu veux modifier, c'est une nouvelle hypothèse, donc un nouveau prompt.

### Règle 10 — Sharpe calculé sur les retours quotidiens, JAMAIS sur PnL/trade
```python
daily_returns = equity_curve.pct_change().dropna()
sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
```
**Jamais** `mean(pnl_per_trade) / std(pnl_per_trade) × √252`. Cette erreur a été commise en v1 et ne sera pas répétée.

### Règle 11 — Retry obligatoire sur toute I/O
Toute I/O (lecture CSV, appel HTTP Telegram, chargement modèle pickle, requête réseau) doit être wrappée avec `@retry_with_backoff(...)` de [app/core/retry.py](../app/core/retry.py) (créé au prompt 02b). 3 tentatives par défaut, backoff exponentiel.

```python
from app.core.retry import retry_with_backoff

@retry_with_backoff(max_attempts=3, exceptions=(OSError, requests.RequestException))
def load_asset(asset: str, tf: str) -> pd.DataFrame: ...
```

### Règle 12 — Seed global au début de chaque script
Tout script `run_*.py` ou `scripts/*.py` commence par :
```python
from app.core.seeds import set_global_seeds
set_global_seeds()  # seed=42 par défaut
```
Cela fixe `random`, `numpy.random`, et `PYTHONHASHSEED`. Reproductibilité bit-à-bit garantie.

### Règle 13 — Anti-hallucination d'API externe
Avant d'utiliser une API ou bibliothèque externe :
1. Vérifier qu'elle existe (doc officielle, pas mémoire LLM).
2. Si doute → **fail-fast** : `raise NotImplementedError("Vérifier l'API X avant impl")`.
3. **Liste blanche autorisée** : Telegram Bot HTTP API (`https://api.telegram.org/bot<TOKEN>/...`), fichiers CSV locaux, `pandas`, `numpy`, `scikit-learn`, `scipy`, `requests`, `apscheduler`.
4. **Liste rouge interdite (hors scope)** : XTB API live trading, MetaTrader 5 API, broker proprietary APIs, services payants nécessitant API key non fournie.

Si Deepseek invente une fonction `xtb_client.place_order(...)` qui n'existe pas → bug en runtime. La règle 13 prévient ce risque.

### Règle 14 — Test set lock mécanique
Tout script qui modifie une stratégie, une feature, ou un seuil **doit** commencer par :
```python
from app.testing.snooping_guard import check_unlocked
check_unlocked()
```
Si `TEST_SET_LOCK.json.locked == True` (après le prompt 18), `TestSetSnoopingError` est levée. Pour itérer, il faut un nouveau split temporel (impossible dans ce projet → arrêt définitif des modifications).

Toute lecture OOS (`df.loc["2024":]`, etc.) doit appeler :
```python
from app.testing.snooping_guard import read_oos
read_oos(prompt="07", hypothesis="H06", sharpe=1.23, n_trades=40)
```

### Règle 15 — Quality gates avant DoD
Avant de marquer un prompt comme `✅ Terminé` dans `JOURNAL.md`, **`make verify` doit passer** :
- `make lint` (ruff)
- `make typecheck` (mypy)
- `make test` (pytest)
- `make snooping_check` (verify_no_snooping)

Si un seul check échoue → status = `⚠️ Partiel` ou `❌ Bloqué`, pas `✅ Terminé`.

---

## 6. Structure standardisée de CHAQUE prompt 01-24

Tous les prompts respectent cette structure :

```markdown
# Prompt XX — <Titre>

## Préalable obligatoire (à lire dans l'ordre)
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/<XX-1>_*.md`

## Objectif
<1 phrase mesurable>

## Definition of Done (testable)
- [ ] Critère 1 (avec commande shell pour vérifier)
- [ ] Critère 2
- [ ] Entry ajoutée dans `JOURNAL.md`
- [ ] Tests `rtk pytest tests/<scope>/ -v` passent

## NE PAS FAIRE
- Ne PAS lire `data/`, `ready-data/`, `cleaned-data/`
- Ne PAS commit
- Ne PAS lancer Python automatiquement
- <garde-fous spécifiques au prompt>

## Étapes
1. <Étape concrète>
2. ...

## Logging
À la fin, ajouter une entrée dans `JOURNAL.md` (cf. Règle 8).

## Critères go/no-go
- **GO prompt suivant** si : <conditions>
- **NO-GO, revenir à** : <prompt N-X> si <condition d'échec>
```

---

## 6bis. Compteur n_trials cumulatif (anti-DSR-cheating)

Chaque hypothèse testée ajoute 1 au compteur global. Le DSR (Deflated Sharpe Ratio) **pénalise** le nombre d'hypothèses testées. Cacher des hypothèses pour gonfler le DSR = fraude statistique.

**Tableau permanent dans `JOURNAL.md`** (initialisé au prompt 01) :

```markdown
## Compteur n_trials cumulatif

| Prompt | Hypothèse | n_trials_new | n_trials_cumul | Verdict | Sharpe |
|---|---|---|---|---|---|
| baseline | v2 H01-H05 | 5 | 5 | mixte | — |
| 07 | H06 (Donchian multi-actif) | 1 | 6 | ... | ... |
| 08 | H07 (strats alt) | 1 | 7 | ... | ... |
| ... | ... | ... | ... | ... | ... |
```

Au prompt 18, `n_trials_cumul` est lu **depuis ce tableau** (pas recompté à la volée). Si une hypothèse est rejouée (ex : H06 avec des params différents), elle compte comme **+1 nouvelle hypothèse**, jamais une mise à jour.

## 6ter. Quality gates (rappel)

Aucun prompt n'est marqué `✅ Terminé` tant que :
1. `make verify` retourne 0
2. L'entrée `JOURNAL.md` est écrite avec les métriques chiffrées
3. Le tableau n_trials est mis à jour (pour les prompts H06-H18)

## 7. Format des messages d'alerte Telegram (cible finale)

Le bot Telegram (prompt 21) envoie ce format exactement (pas de variation) :

```
🎯 <ASSET> <TF> — <DIRECTION>
─────────────────────────
📅 <YYYY-MM-DD HH:MM> UTC
💰 Capital : 10 000 € | Risk : 2 % (200 €)

🎯 Entrée : <price>
🛡️ Stop Loss : <price> (<distance_pips> pips / <distance_pct>%)
🏁 Take Profit : <price> (R:R = <ratio>:1)
📦 Taille : <lots> lots (<units> unités)

📊 Stratégie : <strategy_name> + <meta_labeling_yes_no>
📈 Régime : <Trending / Ranging>
🎲 Confiance modèle : <proba>%

─────────────────────────
🔍 PROMPT À COPIER-COLLER POUR VALIDATION IA :
─────────────────────────
<bloc multi-lignes structuré : contexte marché, justification stratégique,
risques identifiés, question explicite "Est-ce que ce trade est à prendre ?
Réponds OUI ou NON avec 1 phrase de justification."
```

Le bloc de validation IA inclut : timestamp, contexte macro (calendrier économique J+1), corrélation avec autres actifs, drawdown récent du modèle.

---

## 8. Liste des 26 prompts et leur séquence

| # | Phase | Prompt | Dépend de |
|---|---|---|---|
| 00 | Méta | `00_constitution.md` (ce fichier) | — |
| 01 | 0 | `01_audit_initial.md` | 00 |
| 02 | 0 | `02_cleanup.md` | 01 |
| 02b | 0 | `02b_quality_gates.md` | 02 |
| 03 | 1 | `03_data_layer.md` | 02b |
| 04 | 1 | `04_features_research_harness.md` | 03 |
| 05 | 1 | `05_economic_calendar.md` | 03 |
| 06 | 1 | `06_validation_framework.md` | 04, 05 |
| 07 | 2 | `07_h06_us500_donchian.md` | 06 |
| 08 | 2 | `08_h07_strategies_alt.md` | 07 |
| 09 | 2 | `09_h08_portfolio_equal_risk.md` | 07, 08 |
| 10 | 2 | `10_h09_regime_detector.md` | 09 |
| 11 | 2 | `11_h10_h11_h12_meta_labeling.md` | 10 |
| 12 | 2 | `12_h11_features_advanced.md` | 04 |
| 13 | 2 | `13_h12_session_features.md` | 12 |
| 14 | 2 | `14_h13_correlation_weighting.md` | 09 |
| 15 | 2 | `15_h14_vol_targeting.md` | 14 |
| 16 | 2 | `16_h15_tf_decision.md` | 15 |
| 17 | 3 | `17_h16_timeframe_stacking.md` | 16 |
| 18 | 3 | `18_validation_finale.md` | 17 |
| 19 | 3 | `19_h18_walk_forward_continu.md` | 18 |
| 20 | 4 | `20_signal_engine.md` | 19 |
| 21 | 4 | `21_telegram_alerts.md` | 20 |
| 22 | 4 | `22_scheduler_local.md` | 21 |
| 23 | 4 | `23_vps_deployment_options.md` | 22 |
| 24 | 4 | `24_post_deployment_monitor.md` | 23 |

---

## 9. Erreurs déjà commises (NE PAS REPRODUIRE)

Source : `docs/v2_hypothesis_01.md` à `_05.md` + `ml_evolution.md`.

| Erreur | Origine | Comment l'éviter |
|---|---|---|
| RF entraîné sur indicateurs techniques bruts (RSI, ADX, EMA) sans signal préalable | v1, H01, H02 | Le ML est une SURCOUCHE sur un edge déterministe. Trouver d'abord l'edge avec grid search, puis ajouter le méta-labeling. |
| TP/SL fixes en pips/points sur actifs volatils | v1, H02 | Utiliser TP/SL adaptatifs basés sur l'ATR (TP = k_atr × ATR, k_atr ≈ 2). |
| Seuil méta-labeling optimisé sur train uniquement | H04 | Plancher à 0.50. Si l'optimum train élimine > 80 % des trades val → fallback 0.50. |
| Modifier `features_dropped` après lecture du test set | v1 (15 itérations) | Le test set n'est lu qu'UNE FOIS. Modification = nouvelle hypothèse, nouveau split idéalement. |
| Sharpe calculé sur PnL/trade | v1 | Sharpe sur `pct_change` de l'equity curve, toujours. |

---

## 10. Démarrage

Quand l'utilisateur te donne ce prompt 00 :
1. Lis-le intégralement.
2. Réponds : « Constitution lue. <synthèse en 5 lignes>. Prêt pour le prompt 01. »
3. **Ne fais rien d'autre.** Attends le prompt 01.

Quand l'utilisateur te donne un prompt 01-24 (y compris 02b) :
1. Relis le prompt 00 (cette constitution) — toujours.
2. Lis `JOURNAL.md`.
3. Lis le prompt en cours.
4. Vérifie que ses préalables sont satisfaits (prompts précédents terminés dans `JOURNAL.md`).
5. Vérifie que `TEST_SET_LOCK.json` n'est pas verrouillé (sauf prompts 19-24).
6. Si OK, exécute. Sinon, demande à l'utilisateur de revenir au prompt manquant.

---

**Fin de la constitution.** Tous les prompts qui suivent (01 à 24) sont à interpréter à la lumière de ce document.
