# Journal d'exécution — Refonte v3

> Ce fichier est la mémoire vive du projet. À lire au début de chaque session, à mettre à jour à la fin.

---

## Historique v1 (résumé, archivé)

- EURUSD H1 + RandomForest sur features techniques (RSI, ADX, EMA, etc.).
- 16 itérations de tuning documentées dans [`ml_evolution.md`](ml_evolution.md).
- Verdict final : ❌ NO-GO. Sharpe ≤ 0 sur toutes les années OOS, accuracy ≈ aléatoire (0.332).
- DSR 2025 = −1.97, p(Sharpe>0) = 0.29. Biais directionnel SHORT 75–85%.
- Cause racine : RF sur indicateurs bruts ne contient aucune info prédictive forward. Cible bruitée à 36% NEUTRE.
- Code source v1 supprimé au Prompt 02. Historique conservé dans `ml_evolution.md` et `docs/archive_v1/`.

---

## Historique v2 (H01–H05)

### H01 — RF sur US30 D1 (6 features OHLC)
- **Verdict** : ❌ NO-GO
- **Sharpe OOS** : −1.27
- **Trades** : 66, WR 24.2 %
- **Leçon** : RF seul = pas de signal.

### H02 — RF sur XAUUSD H4
- **Verdict** : ❌ NO-GO
- **Sharpe OOS** : −2.52
- **Trades** : 42, WR 16.7 %
- **Leçon** : RF + TP/SL fixes inadaptés à XAUUSD H4.

### H03 — Grid search déterministe (164 backtests)
- **Verdict** : ✅ GO
- **Stratégie gagnante** : Donchian Breakout (20, 20) sur US30 D1
- **Sharpe OOS** : +3.07
- **Leçon** : L'edge se trouve par grid search systématique, pas par ML.

### H04 — Donchian + méta-labeling RF (CPCV)
- **Verdict** : ✅ GO
- **Sharpe OOS** : +8.61 (moyen CPCV 5.79, std ±10.03)
- **Leçon** : Le ML en SURCOUCHE améliore. Mais instabilité élevée.

### H05 — Walk-forward US30 (Config A vs B)
- **Verdict** : ✅ GO
- **Config B (Donchian + RF méta-labeling)** : Sharpe walk-forward +8.84
- **12 trades sur 30 mois** — peu pour valider robustesse
- **Leçon** : Walk-forward stabilise. Mais nombre de trades insuffisant (critère ≥ 30/an non atteint).

---

## Roadmap v3 cible (résumé de `docs/v3_roadmap.md`)

| Phase | Prompts | Contenu |
|---|---|---|
| Phase 0 — Nettoyage | 01, 02, 02b | Audit, cleanup, quality gates |
| Phase 1 — Data & Features | 03, 04, 05, 06 | Data layer, features harness, calendrier économique, validation framework |
| Phase 2 — Expansion univers | 07, 08, 09 | H06 Donchian multi-actif, H07 stratégies alternatives, H08 portefeuille equal-risk |
| Phase 3 — Régime & filtrage | 10, 11, 12, 13, 14 | H09 régime detector, H10-H12 méta-labeling, H11 features avancées, H12 session, H13 corrélation |
| Phase 4 — Portfolio avancé | 15, 16, 17 | H14 vol targeting, H15 TF décision, H16 timeframe stacking |
| Phase 5 — Validation finale | 18, 19 | Validation finale, H18 walk-forward continu |
| Phase 6 — Production | 20, 21, 22, 23, 24 | Signal engine, Telegram alerts, scheduler, VPS, monitoring |

**Objectif final** : Sharpe walk-forward portfolio ≥ 1.0, DSR > 0 (p < 0.05), DD < 15 %, WR > 30 %, ≥ 30 trades/an.

---

## Compteur n_trials cumulatif

| Prompt | Hypothèse | n_trials_new | n_trials_cumul | Verdict | Sharpe |
|---|---|---|---|---|---|
| baseline | v1 EURUSD H1 (16 itérations) | 16 | 16 | ❌ NO-GO | — |
| baseline | v2 H01 | 1 | 17 | ❌ NO-GO | −1.27 |
| baseline | v2 H02 | 1 | 18 | ❌ NO-GO | −2.52 |
| baseline | v2 H03 | 1 | 19 | ✅ GO | +3.07 |
| baseline | v2 H04 | 1 | 20 | ✅ GO | +8.61 |
| baseline | v2 H05 | 1 | 21 | ✅ GO | +8.84 |
| 03–06 | (phase data, pas d'hypothèse) | 0 | 21 | — | — |
| 07 | H06 (Donchian multi-actif) | — | — | — | — |
| 08 | H07 (strats alt) | — | — | — | — |
| 09 | H08 (portfolio equal-risk) | — | — | — | — |
| 10 | H09 (régime detector) | — | — | — | — |
| 11 | H10-H12 (méta-labeling v3) | — | — | — | — |
| 12 | H11 (features avancées) | — | — | — | — |
| 13 | H12 (session features) | — | — | — | — |
| 14 | H13 (corrélation weighting) | — | — | — | — |
| 15 | H14 (vol targeting) | — | — | — | — |
| 16 | H15 (TF décision) | — | — | — | — |
| 17 | H16 (timeframe stacking) | — | — | — | — |
| 18 | Validation finale | — | — | — | — |
| 19 | H18 (walk-forward continu) | — | — | — | — |

---

## Sessions Deepseek

---

## 2026-05-14 — Prompt 01 : Audit initial

- **Statut** : ✅ Terminé
- **Fichiers créés** : `INVENTORY.md`, `JOURNAL.md`
- **Résultats clés** :
  - 0 CSV trouvés dans `data/raw/` (dossier vide, seul `economic_calendar/` présent)
  - 6 scripts `run_*.py` à la racine (H01–H05 + v3 phase1)
  - 22 rapports/docs dans `docs/` (5 hypothèses v2 + roadmap v3 + 12 specs/rapports step + 2 README)
  - 25 fichiers de test (23 unit, 1 acceptance, 1 conftest)
  - n_trials cumul initialisé à 21 (16 v1 + 5 v2)
- **Problèmes rencontrés** : Aucun CSV d'actif dans `data/raw/` — l'utilisateur devra les fournir avant le prompt 03 (data layer)
- **Hypothèses à explorer ensuite** : (traitées au prompt 02)

## 2026-05-14 — Prompt 02 : Nettoyage et restructuration

- **Statut** : ✅ Terminé
- **Fichiers/dossiers supprimés** : `learning_machine_learning/` (v1), `archive_v1/`, `results/`, `__pycache__/` (tous niveaux), `.pytest_cache/`
- **Fichiers déplacés** : 17 fichiers `docs/step_*.md` → `docs/archive_v1/`
- **Fichiers renommés** : `learning_machine_learning_v2/` → `app/` (via `git mv`)
- **Imports corrigés** : 24 fichiers `.py` (6 scripts racine + 18 internes `app/`)
- **Fichiers markdown mis à jour** : `INVENTORY.md`, `CLAUDE.md`, `README.md`, `.gitignore`, `JOURNAL.md`
- **Tests pytest** : Non exécutés (constitution règle 2 — exécution sur demande)
- **Problèmes rencontrés** : Aucun
- **Structure finale** : `app/`, `docs/`, `docs/archive_v1/`, `prompts/`, `tests/`, `scripts/`, `predictions/`, `data/raw/`
