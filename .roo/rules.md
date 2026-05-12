# Règles de Codage — Pipeline ML Trading

## 1. Typage
- **Toute fonction/méthode** doit avoir des type hints complets sur la signature et le retour.
- Utiliser `from __future__ import annotations` en tête de chaque fichier.
- Les types génériques (`dict`, `list`, `tuple`) doivent avoir leurs paramètres : `dict[str, float]`, `list[int]`, `tuple[str, ...]`.
- Préférer `numpy.ndarray` à `list` pour les tableaux numériques.

## 2. Tests Unitaires
- **Après toute modification de code** dans `learning_machine_learning/`, exécuter :
  ```powershell
  python -m pytest tests/unit/ -v --tb=short
  ```
- Si des tests échouent, corriger avant de continuer.
- Tout nouveau module doit être accompagné d'un fichier de tests dans `tests/unit/`.
- **Obligatoire** : Toute nouvelle classe publique ou fonction exportée doit avoir ses tests unitaires ajoutés **dans le même commit** que le code. Ne jamais différer les tests. Les tests doivent couvrir : cas nominal, cas d'erreur (ValueError), cas limites (zéro, vide), et invariants.

## 3. Vectorisation
- **Zéro** `iterrows()`, `itertuples()`, `apply()` avec lambda.
- Toujours vectoriser avec NumPy : `.values`, `.shift()`, `.rolling()`, opérations broadcast.
- Les calculs financiers (Sharpe, drawdown, volatilité) doivent être en pur NumPy, pas en boucle Python.

## 4. Anti-Data Leakage
- Split temporel strict, jamais de `shuffle`.
- `merge_asof` avec direction `backward` uniquement.
- Toute feature doit être calculée sur des données **antérieures ou contemporaines** à la barre cible, jamais futures.
- Les labels triple barrière doivent utiliser `numpy.roll` ou équivalent sans look-ahead.

## 5. Logging
- Utiliser le logger structuré du projet (`learning_machine_learning.core.logging`).
- Chaque étape du pipeline logge ses dimensions (n_lignes, n_colonnes).
- Toute exception est loggée avec `logger.error("msg", exc_info=True)`.

## 6. Immutabilité des Configs
- Toutes les dataclasses de config sont `frozen=True`.
- Pour dériver une config, utiliser `dataclasses.replace(config, champ=valeur)`.
- Pas de mutation de config au runtime.

## 7. Rapport d'Évolution
- **Toute modification** du pipeline doit être documentée dans `ml_evolution.md`.
- Format obligatoire : date/version, modification, hypothèse, résultats pré-fix, résultats post-fix (**mesurés**, pas `[À MESURER]`), target metrics.
- Mesurer les résultats en relançant `run_pipeline_v1.py` avant d'écrire le log.
