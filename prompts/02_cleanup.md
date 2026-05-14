# Prompt 02 — Nettoyage et restructuration

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/01_audit_initial.md`

## Objectif
Supprimer le cruft historique (v1, archive_v1, docs obsolètes, caches) et renommer `learning_machine_learning_v2/` → `app/` pour préparer la suite. Le repo doit avoir une structure propre, lisible, sans duplication.

## Definition of Done (testable)
- [ ] `learning_machine_learning_v2/` n'existe plus, remplacé par `app/` (renommage simple, pas de modification de code).
- [ ] `learning_machine_learning/` (v1) est supprimé.
- [ ] `archive_v1/` est supprimé.
- [ ] `results/` (CSV legacy v1) est supprimé.
- [ ] `__pycache__/` à tous les niveaux est supprimé.
- [ ] `.pytest_cache/` est supprimé.
- [ ] `docs/step_*.md` (specs v1 obsolètes) sont déplacés dans `docs/archive_v1/`.
- [ ] `ml_evolution.md` est conservé tel quel.
- [ ] Les imports du code dans `app/` qui référençaient encore `learning_machine_learning_v2` sont corrigés (`from app.X` au lieu de `from learning_machine_learning_v2.X`).
- [ ] `rtk pytest tests/ -v` passe (les tests existants doivent continuer à fonctionner après renommage).
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS toucher au dossier `data/`, `ready-data/`, `cleaned-data/`.
- Ne PAS supprimer `docs/v2_hypothesis_*.md` ni `docs/v3_roadmap.md` (historique précieux).
- Ne PAS supprimer `tests/`.
- Ne PAS supprimer les CSV legacy s'ils sont dans `data/` (uniquement ceux dans `results/`).
- Ne PAS modifier la logique métier dans le code — uniquement les chemins d'import.
- Ne PAS commit.

## Étapes

### Étape 1 — Confirmation utilisateur
Avant toute suppression, **demander à l'utilisateur la confirmation explicite** de la liste exacte des dossiers/fichiers à supprimer. Ne supprimer que ce qui est explicitement validé.

### Étape 2 — Sauvegarde des docs v1
```bash
mkdir -p docs/archive_v1
mv docs/step_*.md docs/archive_v1/
mv docs/v1_*.md docs/archive_v1/  # si existant
```

### Étape 3 — Renommage v2 → app
```bash
git mv learning_machine_learning_v2 app
```

Puis corriger TOUS les imports :
```bash
rtk grep -rn "learning_machine_learning_v2" --include="*.py"
```
Pour chaque fichier listé, remplacer `learning_machine_learning_v2` par `app` (Edit ciblé, pas sed global).

### Étape 4 — Suppression v1 et cruft
```bash
rm -rf learning_machine_learning/
rm -rf archive_v1/
rm -rf results/
```

### Étape 5 — Suppression caches
```bash
rtk find . -type d -name "__pycache__" -exec rm -rf {} +
rtk find . -type d -name ".pytest_cache" -exec rm -rf {} +
rtk find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
```

### Étape 6 — Mettre à jour les scripts à la racine
Pour chaque `run_*.py` à la racine, vérifier qu'il importe `from app.` et non `from learning_machine_learning_v2.`. Corriger via Edit.

### Étape 7 — Mettre à jour `CLAUDE.md` et `README.md`
Remplacer les références à `learning_machine_learning/` et `learning_machine_learning_v2/` par `app/`.

### Étape 8 — Mettre à jour `.gitignore`
S'assurer que `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.env` sont bien ignorés.

### Étape 9 — Tests de non-régression
```bash
rtk pytest tests/ -v --tb=short
```
**Sur demande utilisateur uniquement.** Tous les tests doivent passer.

### Étape 10 — Inventaire final
Régénérer `INVENTORY.md` (Étape 5 du prompt 01) pour refléter la nouvelle structure.

## Logging
```markdown
## 2026-MM-DD — Prompt 02 : Cleanup
- **Statut** : ✅ Terminé
- **Fichiers/dossiers supprimés** : learning_machine_learning/, archive_v1/, results/, __pycache__/, .pytest_cache/
- **Fichiers déplacés** : docs/step_*.md → docs/archive_v1/
- **Fichiers renommés** : learning_machine_learning_v2/ → app/
- **Imports corrigés** : <nombre> fichiers (.py)
- **Tests pytest** : ✅ passent (X tests, 0 failures) / ⚠️ <échecs>
- **Problèmes rencontrés** : ...
```

## Critères go/no-go
- **GO prompt 03** si : `app/` existe, `pytest` passe, INVENTORY.md à jour.
- **NO-GO, revenir à** : ce prompt si tests cassés. Identifier le module incriminé, corriger les imports.
