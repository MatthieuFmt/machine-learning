# Audit v5 — Index des documents

**Date** : 2026-05-19 (audit + plan + exécution des phases 1-5 du correctif)
**Demande utilisateur** : audit complet du pipeline + plan d'amélioration (pas de production)

---

## Documents produits dans cet audit

| Document | Rôle |
|---|---|
| [audit_v4_findings.md](audit_v4_findings.md) | **19 findings classés par gravité** (3 bloqueurs, 7 importants, 9 mineurs). Chaque finding cite `file:line`. C'est la **preuve technique** du diagnostic. |
| [audit_v4_strategies_viability.md](audit_v4_strategies_viability.md) | **Analyse de viabilité réelle** des 6 stratégies "GO". Estimation des Sharpe attendus une fois les bugs corrigés. |
| [plan_v5_correction_critique.md](plan_v5_correction_critique.md) | **Plan d'action** pour corriger les bugs. 5 phases, 5-7 jours de travail. Critères de sortie explicites. |
| [plan_v5_amelioration_strategies.md](plan_v5_amelioration_strategies.md) | **Plan d'extension** (post-correction) : améliorer les 6 existantes + ajouter de nouvelles familles. Exploratoire, sans timing. |
| [audit_v5_execution_status.md](audit_v5_execution_status.md) | **Statut d'exécution** : 15/19 findings résolus, 49 tests neufs, 0 régression. À lire pour savoir où on en est. |

Document complémentaire pré-existant : [audit_final_post_mortem.md](audit_final_post_mortem.md) — couvre les risques *suspectés* à haut niveau. Le présent audit prouve lesquels sont des bugs réels.

---

## Verdict synthétique (2 minutes de lecture)

### Ce qui ne va pas

1. **Le pipeline est cassé sur 3 points méthodologiques** (F1, F2, F3 dans findings) :
   - F1 — En test, le Donchian disparaît. Le ML est appliqué à toutes les barres, la direction vient de SMA slopes. **Ce n'est pas du méta-labeling.**
   - F2 — Sharpe gonflé ×3.5 par incohérence entre sizing à capital fixe et calcul Sharpe en composé.
   - F3 — Le simulateur déterministe est optimiste sur les bougies same-bar (TP-prime au lieu de SL-prime). Inverse de la spec annoncée.

2. **Le verdict NO-GO actuel est probablement un faux négatif**. Le portfolio est inflaté, mais le benchmark Monte Carlo l'est encore plus (à cause de F3). Une fois corrigé, les deux baissent ensemble et le rapport est imprévisible.

3. **Les 6 stratégies "GO" individuellement** ne sont pas des edges démontrés. Estimation post-correction : ~1.0-1.5 Sharpe au mieux, peut-être moins.

### Ce qui va bien

- L'architecture du code est saine (`app/` bien découpé, dataclasses gelées, Protocols).
- Le `snooping_guard.py` mécanique est une bonne idée — il faut juste s'assurer que `n_trials` reflète vraiment `n_reads` (F5).
- Les features de base (`indicators.py`, `regime.py`) semblent corrects, le souci est ailleurs (cross-asset, look-ahead deco sans validation).
- Le post-mortem existant (audit_final_post_mortem.md) avait déjà identifié plusieurs des risques (overfitting ETHUSD, coûts provisoires, taille échantillon). Mon audit ajoute la **preuve dans le code**.

### Ce qu'il faut faire ensuite

**Court terme** : exécuter le plan de correction critique (Phase 1 = bloqueurs). Sans ça, tout nouveau test est invalide.

**Moyen terme** : rejouer la validation avec pipeline corrigé. Décider à ce moment :
- Si **au moins une** stratégie survit → continuer Axe A (améliorer les existantes).
- Si **aucune** → passer à Axe B (nouvelles familles : Bollinger, Keltner, ts_momentum) et Axe C (régime dispatch).

**Long terme** : accepter qu'un résultat "pas d'edge détectable" est valide scientifiquement. Ne pas forcer un Sharpe inventé via bugs.

---

## Points sur lesquels je n'ai pas pu trancher (à exécuter pour confirmer)

Les findings ci-dessous sont **identifiés** mais nécessitent une exécution Python pour vérification finale (je n'exécute pas de code automatiquement) :

1. **F12 — max_dd_pct = -1549 %** : tracer pourquoi `compute_metrics` retourne hors [-100, 0]. Hypothèse : chemin legacy non-A1, ou `pip_value_eur` mal câblé.
2. **F7 — convention timestamp D1/H1** : vérifier sur un CSV réel si les D1 timestamps sont start-of-day ou end-of-day.
3. **F4 — impact réel du look-ahead Stacking** : entraîner le même Stacking avec `TimeSeriesSplit` vs `KFold` et comparer le delta accuracy/Sharpe.
4. **Ampleur de F1+F2 combinés** : refaire le validation_finale avec Option A (méta-labeling fidèle) ET Sharpe linéaire, et comparer les Sharpe affichés vs originaux.

---

## Suggestions hors-scope (à logger si l'utilisateur veut explorer)

- Le portfolio combine 4 couples sur D1 USD (GBPUSD, EURUSD, USDCHF, ETHUSD). La diversification effective est probablement très faible (corrélation > 0.7). Une matrice de corrélation des sleeves serait éclairante.
- La v2 (H05, Sharpe 8.84 sur US30 D1) est probablement **aussi affectée par F2 et F3**. Le résultat fondateur du projet est à re-vérifier.
- 15 fichiers de tests `.bak` indiquent une dette de tests. Si on veut un audit de coverage, c'est un point d'entrée.
- Plusieurs scripts `run_*.py` à la racine (run_pipeline_us30.py, run_walk_forward_us30.py, etc.) coexistent avec `scripts/`. Centralisation possible.

Ces points sont notés dans une section "Hypothèses à explorer ensuite" plutôt que dans le plan principal.
