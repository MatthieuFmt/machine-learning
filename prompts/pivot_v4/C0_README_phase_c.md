# Pivot v4 — Phase C : Extension multi-actifs du pipeline ML

> **À LIRE EN PREMIER avant d'exécuter C1, C2, C3, C4 ou C5.**

## Quoi faire

Étendre la Phase A (A5→A9) du pivot v4 aux 18 couples (actif, TF) jamais testés en pipeline ML, **sans toucher au test set ≥ 2024**.

Périmètre des nouveaux couples : BTCUSD, ETHUSD, GBPUSD, USDCHF (tous TF) + EURUSD (D1, H1) + US30 (H1, H4) + XAUUSD (H1, H4).

## Lecture obligatoire avant de commencer (dans cet ordre)

1. [../00_constitution.md](../00_constitution.md) — règles du projet (test set sanctity, RTK, langue, dossiers interdits)
2. [00_README.md](00_README.md) — vue d'ensemble pivot v4 (Phase A originale + Phase B)
3. [../../JOURNAL.md](../../JOURNAL.md) — état d'avancement (lire au moins les sections A1→A9 + B1→B4)
4. [../../CLAUDE.md](../../CLAUDE.md) — architecture du repo, conventions de code
5. [../../docs/cost_audit_v2.md](../../docs/cost_audit_v2.md) — convention pip_size, justification des coûts XTB

## Ordre d'exécution strict

```
C1 → C2 → C3 → C4 → C5
```

| Ordre | Prompt | Objectif | Durée estimée |
|---|---|---|---|
| 1 | [C1_extend_a5_multi_assets.md](C1_extend_a5_multi_assets.md) | ASSET_CONFIGS + inventaire + smoke test superset | 30-60 min |
| 2 | [C2_extend_a6_ranking_multi_assets.md](C2_extend_a6_ranking_multi_assets.md) | Ranking top 15 + bootstrap stability sur 18 couples | 1-2 h compute + revue |
| 3 | [C3_extend_a7_model_selection_multi_assets.md](C3_extend_a7_model_selection_multi_assets.md) | RF vs HGBM vs Stacking en CPCV sur shortlist C2 | 1-2 h compute |
| 4 | [C4_extend_a8_hyperparams_multi_assets.md](C4_extend_a8_hyperparams_multi_assets.md) | Nested CPCV hyperparams + seuil sur shortlist C3 | 2-4 h compute |
| 5 | [C5_extend_a9_pipeline_lock_multi_assets.md](C5_extend_a9_pipeline_lock_multi_assets.md) | Bump version, SHA256, bilan global, recommandations | 30 min |

**Vérification de dépendances** : au début de chaque prompt Cx, vérifier dans `JOURNAL.md` que Cx-1 est ✅ Terminé. Si non → STOP, demander à l'utilisateur.

## Règles strictes (rappel de la constitution)

1. **Pas d'exécution Python automatique.** Tu écris le code, l'utilisateur lance `rtk python ...` lui-même.
2. **Pas de commit git sans accord explicite** de l'utilisateur.
3. **Test set ≥ 2024 jamais lu** sur toute la Phase C. Cutoff strict `2022-12-31 23:59:59 UTC`.
4. **n_trials reste à 28** à la fin de C5 (Phase C entière à 0 trial).
5. **Préfixer toute commande CLI longue par `rtk`** (pytest, python run_*.py, etc.) pour économiser le contexte.
6. **Ne PAS lire `data/`, `ready-data/`, `cleaned-data/`** directement. Uniquement via `app/data/loader.py`.
7. **Ne PAS modifier les 3 entrées A9 d'origine** dans `features_selected.py`, `model_selected.py`, `hyperparams_tuned.py` (US30 D1, EURUSD H4, XAUUSD D1). Chaque prompt a un test qui le vérifie.
8. **Coûts XTB BTCUSD/ETHUSD/GBPUSD/USDCHF** ajoutés en C1 sont PROVISOIRES — l'utilisateur les corrigera après C5 via une session "vérif spreads démo".

## Pendant l'exécution de chaque Cx

Pour chaque prompt :
1. Lire la section "Préalable obligatoire" du prompt et charger les fichiers cités.
2. Lire la section "Definition of Done" pour savoir ce qui doit exister à la fin.
3. Lire la section "NE PAS FAIRE" — ces interdits sont durs.
4. Implémenter étapes 1, 2, 3... dans l'ordre.
5. Faire valider par l'utilisateur avant `rtk make verify`.
6. Une fois ✅ : mettre à jour `JOURNAL.md` avec la section "Logging obligatoire" du prompt.

## En cas de blocage

- Une dépendance Cx-1 manque dans `JOURNAL.md` → STOP, demander à l'utilisateur si elle doit être faite d'abord.
- Un test échoue → ne PAS contourner. Investiguer la cause et fixer.
- Un fichier figé (entrée A9 originale) doit être touché → STOP, c'est un signe d'erreur méthodologique. Demander.
- Données manquantes pour un couple (BTCUSD H1 par exemple) → ne pas bloquer la Phase C : le couple sort avec `status="data_missing"`, on continue.

## À la fin de C5

L'utilisateur reprend la main avec 3 options documentées dans `docs/phase_a_extended_summary.md` :
- **Option A** : Phase B sélective sur 1-3 couples shortlist (+1 à +3 n_trials)
- **Option B** : Vérification spreads démo XTB + correction `ASSET_CONFIGS` (0 n_trial)
- **Option C** : Prompt 18 validation finale sur portfolio existant (+1 n_trial)

**Ne PAS choisir à la place de l'utilisateur.**

## Fin du README Phase C.
**Premier prompt à ouvrir** : [C1_extend_a5_multi_assets.md](C1_extend_a5_multi_assets.md)
