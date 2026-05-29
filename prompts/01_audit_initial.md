# Prompt 01 — Audit initial et création du journal

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. (Pas de journal encore — ce prompt va le créer)

## Objectif
Faire un audit exhaustif du repo et produire deux fichiers à la racine : `INVENTORY.md` (structure du repo) et `JOURNAL.md` (journal d'exécution initialisé avec l'historique v2).

## Definition of Done (testable)
- [ ] `INVENTORY.md` existe à la racine et liste tous les dossiers de premier niveau avec une description courte de chacun.
- [ ] `JOURNAL.md` existe à la racine et contient les sections : `Historique v1` (résumé 5 lignes), `Historique v2 (H01–H05)` (un bloc par hypothèse avec verdict et métriques), `Roadmap v3 cible` (résumé de `docs/v3_roadmap.md`), `Sessions Deepseek` (vide pour l'instant).
- [ ] La liste des CSV disponibles dans `data/raw/` est dans `INVENTORY.md` — utiliser `ls data/raw/` (sans lire le contenu des CSV).
- [ ] Aucune autre modification du repo.

## NE PAS FAIRE
- Ne PAS lire le contenu de `data/`, `ready-data/`, `cleaned-data/` (uniquement `ls`).
- Ne PAS modifier de fichier de code.
- Ne PAS supprimer quoi que ce soit (cleanup = prompt 02).
- Ne PAS commit.
- Ne PAS lancer Python ou pytest.

## Étapes

### Étape 1 — Recenser la structure
Commande :
```bash
rtk ls -la
```
Lister les dossiers de premier niveau et leur rôle apparent.

### Étape 2 — Recenser les actifs disponibles
Commande :
```bash
rtk ls data/raw/
```
Pour chaque dossier d'actif, lister les fichiers CSV présents et leurs timeframes (D1, H4, H1, etc.) — **sans lire le contenu**.

### Étape 3 — Recenser les scripts à la racine
Commande :
```bash
rtk ls *.py
```
Pour chaque `run_*.py`, lire UNIQUEMENT son docstring de tête (max 30 premières lignes) pour identifier son rôle.

### Étape 4 — Recenser les rapports d'hypothèses
Commande :
```bash
rtk ls docs/
```
Lister les fichiers `v2_hypothesis_*.md` et `v3_*.md`.

### Étape 5 — Construire `INVENTORY.md`

Format :

```markdown
# Inventaire du repo — <YYYY-MM-DD>

## Dossiers de premier niveau
| Dossier | Rôle | Statut (à conserver / à supprimer / à migrer) |
|---|---|---|
| ... | ... | ... |

## Actifs disponibles (data/raw/)
| Actif | Timeframes | Première date | Dernière date |
|---|---|---|---|
| US30 | D1, H4 | (à compléter via inspect scripts si dispo) | ... |

## Scripts run_*.py à la racine
| Script | Hypothèse | Statut |
|---|---|---|
| run_pipeline_us30.py | H01 | Exécuté, NO-GO |
| ... | ... | ... |

## Rapports d'hypothèses (docs/)
| Fichier | Hypothèse | Verdict | Sharpe OOS |
|---|---|---|---|
| v2_hypothesis_01.md | H01 | NO-GO | −1.27 |
| ... | ... | ... | ... |

## Tests existants
| Dossier | Nombre de fichiers | Couverture estimée |
|---|---|---|
| tests/unit | ... | ... |
```

### Étape 6 — Construire `JOURNAL.md`

Format :

```markdown
# Journal d'exécution — Refonte v3

> Ce fichier est la mémoire vive du projet. À lire au début de chaque session, à mettre à jour à la fin.

## Historique v1 (résumé, archivé)
- EURUSD H1 + RandomForest sur features techniques.
- 15 itérations de tuning.
- Verdict final : NO-GO. Sharpe ≤ 0, accuracy < seuil breakeven.
- Cause racine : RF sur indicateurs bruts ne contient aucune info prédictive forward.

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
- **Leçon** : Walk-forward stabilise. Mais nombre de trades insuffisant.

## Roadmap v3 cible (résumé de docs/v3_roadmap.md)

Phase 1 — Expansion univers (H06–H08)
Phase 2 — Régime et filtrage (H09–H12)
Phase 3 — Portfolio avancé (H13–H15)
Phase 4 — Timeframe stacking (H16–H17)
Phase 5 — Walk-forward continu (H18)

Objectif final : Sharpe walk-forward portfolio ≥ 1.0, DSR > 0 (p < 0.05),
DD < 15 %, WR > 30 %, ≥ 30 trades/an.

## Sessions Deepseek

<!-- Chaque exécution de prompt ajoute une section ici -->
```

### Étape 7 — Vérifier
```bash
rtk ls INVENTORY.md JOURNAL.md
```
Les deux fichiers doivent exister.

## Logging
Ajouter en fin de `JOURNAL.md` :

```markdown
## 2026-MM-DD — Prompt 01 : Audit initial
- **Statut** : ✅ Terminé
- **Fichiers créés** : INVENTORY.md, JOURNAL.md
- **Résultats clés** : <nombre> actifs trouvés dans data/raw, <nombre> scripts run_*.py, <nombre> rapports hypothèses
- **Problèmes rencontrés** : <si applicable>
- **Hypothèses à explorer ensuite** : (alimenté au fil de l'eau)
```

## Critères go/no-go
- **GO prompt 02** si : INVENTORY.md et JOURNAL.md sont écrits et reflètent fidèlement le repo.
- **NO-GO, revenir à** : prompt 00 si la constitution n'a pas été lue.
