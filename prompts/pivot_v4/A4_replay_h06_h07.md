# Pivot v4 — A4 : Replay H06 + H07 sur train + val (audit informatif)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [A1_audit_simulator.md](A1_audit_simulator.md), [A2_calibration_costs.md](A2_calibration_costs.md), [A3_sharpe_low_frequency.md](A3_sharpe_low_frequency.md) — **TOUS ✅ Terminés**
3. [docs/v3_hypothesis_06.md](../../docs/v3_hypothesis_06.md), [docs/v3_hypothesis_07.md](../../docs/v3_hypothesis_07.md) — anciens résultats
4. [../00_constitution.md](../00_constitution.md) — règle 9 (un seul regard OOS) + règle 14 (test set lock)

## Objectif
Rejouer Donchian (H06) et 4 stratégies alt (H07) avec le simulateur corrigé (A1-A3) **uniquement sur train ≤ 2022 et val 2023**. Ne PAS toucher au test set 2024+ qui est brûlé pour ces hypothèses.

Le but est **purement informatif** : voir si l'edge train/val réapparaît avec coûts corrigés. Ce n'est PAS une nouvelle hypothèse, donc 0 n_trial consommé.

## Type d'opération
🔍 **Audit informatif** — 0 n_trial consommé.

> ⚠️ **C'EST L'ÉTAPE LA PLUS RISQUÉE DU PIVOT V4.** Une lecture accidentelle du test set 2024+ contamine définitivement la décision GO/NO-GO de la Phase B. Lire attentivement la section "NE PAS FAIRE".

## Definition of Done (testable)

- [ ] `scripts/run_pivot_a4_replay.py` (NOUVEAU) est créé. Il :
  - Charge **uniquement les données ≤ 2023-12-31** (filtrage explicite avant tout calcul).
  - Rejoue le grid search Donchian + 4 stratégies alt sur train ≤ 2022.
  - Évalue les meilleurs params sur val 2023.
  - **NE TOUCHE PAS** au test set 2024+. Une assertion `df.index.max() <= "2023-12-31"` doit échouer le script si du test set traîne.
- [ ] `docs/v3_hypothesis_06_replay.md` et `docs/v3_hypothesis_07_replay.md` créés avec :
  - Tableau comparatif : Sharpe train v3 vs v4, Sharpe val v3 vs v4, DD v3 vs v4
  - Verdict informatif : "l'edge train/val [a / n'a pas] résisté à la correction"
- [ ] `predictions/pivot_a4_replay.json` : structure complète des résultats.
- [ ] **Vérification anti-snooping** : `python scripts/verify_no_snooping.py` retourne 0.
- [ ] Le tableau `n_trials` dans `JOURNAL.md` reste à 22 (pas +1).
- [ ] `JOURNAL.md` mis à jour avec l'entrée "audit informatif".

## NE PAS FAIRE

- ❌ **NE PAS LIRE LE TEST SET ≥ 2024** sous aucun prétexte. Filtrer explicitement `df.loc[:"2023-12-31"]` avant tout calcul.
- ❌ Ne PAS appeler `read_oos()` pour ce replay (ce n'est pas une lecture OOS test set).
- ❌ Ne PAS conclure GO/NO-GO sur ces résultats. Ce sont des **indications**, pas des verdicts.
- ❌ Ne PAS modifier les stratégies en réaction (data snooping post-hoc même sur val).
- ❌ Ne PAS incrémenter n_trials.
- ❌ Ne PAS modifier `TEST_SET_LOCK.json`.

## Étapes détaillées

### Étape 1 — Script `scripts/run_pivot_a4_replay.py`

```python
"""Pivot v4 A4 — Replay H06/H07 sur train+val avec simulateur corrigé.

⚠️ AUDIT INFORMATIF : ne touche jamais au test set ≥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Anti-look-ahead enforcement : import en haut
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.data.registry import discover_assets
from app.config.instruments import ASSET_CONFIGS
from app.backtest.deterministic import run_deterministic_backtest
from app.backtest.metrics import compute_metrics
from app.strategies.donchian import DonchianBreakout
from app.strategies.dual_ma import DualMovingAverage
from app.strategies.keltner import KeltnerChannel
from app.strategies.chandelier import ChandelierExit
from app.strategies.parabolic import ParabolicSAR

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"

# ⚠️ INVARIANT CRITIQUE : aucune donnée > 2023-12-31 ne doit entrer dans ce script.
CUTOFF_DATE = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")


def _filter_to_train_val(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre strict aux données ≤ 2023-12-31. Lève si du test set est présent."""
    filtered = df.loc[df.index <= CUTOFF_DATE]
    if filtered.empty:
        raise ValueError("Aucune donnée train/val disponible")
    if filtered.index.max() > CUTOFF_DATE:
        raise AssertionError("Test set leak détecté ! Vérifier le filtrage.")
    return filtered


def replay_donchian_all_assets() -> dict:
    """Rejoue H06 sur train ≤ 2022 et val 2023, agnostic du test set."""
    set_global_seeds()
    assets = discover_assets()
    candidates = [a for a, tfs in assets.items() if "D1" in tfs and a in ASSET_CONFIGS]
    results: dict = {}

    for asset in sorted(candidates):
        df = load_asset(asset, "D1")
        df = _filter_to_train_val(df)
        cfg = ASSET_CONFIGS[asset]

        df_train = df.loc[:TRAIN_END]
        df_val = df.loc[VAL_START:VAL_END]
        if df_train.empty or df_val.empty:
            results[asset] = {"error": "train ou val vide"}
            continue

        best_sharpe_train, best_params = -1e9, None
        for N in [20, 50, 100]:
            for M in [10, 20, 50]:
                strat = DonchianBreakout(N=N, M=M)
                trades_train = run_deterministic_backtest(df_train, strat, cfg)
                m = compute_metrics(trades_train, asset_cfg=cfg, capital_eur=10_000.0)
                if m["sharpe"] > best_sharpe_train:
                    best_sharpe_train = m["sharpe"]
                    best_params = {"N": N, "M": M}

        if best_params is None:
            results[asset] = {"error": "grid search vide"}
            continue

        # Eval val 2023
        strat = DonchianBreakout(**best_params)
        trades_val = run_deterministic_backtest(df_val, strat, cfg)
        m_val = compute_metrics(trades_val, asset_cfg=cfg, capital_eur=10_000.0)

        results[asset] = {
            "best_params": best_params,
            "sharpe_train_v4": float(best_sharpe_train),
            "sharpe_val_v4": float(m_val["sharpe"]),
            "sharpe_method_val": m_val["sharpe_method"],
            "wr_val": float(m_val["win_rate"]),
            "max_dd_pct_val": float(m_val["max_dd_pct"]),
            "trades_val": int(m_val["trades"]),
        }
    return results


def replay_h07_us30() -> dict:
    """Rejoue les 4 strats alt sur US30 D1 train+val."""
    set_global_seeds()
    df = load_asset("US30", "D1")
    df = _filter_to_train_val(df)
    cfg = ASSET_CONFIGS["US30"]
    df_train = df.loc[:TRAIN_END]
    df_val = df.loc[VAL_START:VAL_END]

    strats = {
        "dual_ma":  [DualMovingAverage(fast=f, slow=s) for f in (5, 10, 20) for s in (50, 100, 200)],
        "keltner":  [KeltnerChannel(period=p, mult=m) for p in (10, 20, 50) for m in (1.5, 2.0, 2.5)],
        "chandelier": [ChandelierExit(period=p, k_atr=k) for p in (11, 22, 44) for k in (2.0, 3.0, 4.0)],
        "parabolic":  [ParabolicSAR(step=s, af_max=a) for s in (0.01, 0.02, 0.03) for a in (0.1, 0.2, 0.3)],
    }

    results: dict = {}
    for name, candidate_list in strats.items():
        best_sharpe, best_strat = -1e9, None
        for strat in candidate_list:
            trades = run_deterministic_backtest(df_train, strat, cfg)
            m = compute_metrics(trades, asset_cfg=cfg, capital_eur=10_000.0)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_strat = strat

        if best_strat is None:
            results[name] = {"error": "no candidate"}
            continue

        trades_val = run_deterministic_backtest(df_val, best_strat, cfg)
        m_val = compute_metrics(trades_val, asset_cfg=cfg, capital_eur=10_000.0)
        results[name] = {
            "best_strat": str(best_strat),
            "sharpe_train_v4": float(best_sharpe),
            "sharpe_val_v4": float(m_val["sharpe"]),
            "sharpe_method_val": m_val["sharpe_method"],
            "wr_val": float(m_val["win_rate"]),
            "max_dd_pct_val": float(m_val["max_dd_pct"]),
            "trades_val": int(m_val["trades"]),
        }
    return results


def main() -> int:
    out = {
        "h06_replay": replay_donchian_all_assets(),
        "h07_replay": replay_h07_us30(),
    }
    out_path = Path("predictions/pivot_a4_replay.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Replay terminé. Résultats dans {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 2 — Génération des rapports comparatifs

`docs/v3_hypothesis_06_replay.md` :

```markdown
# Hypothesis 06 — Replay pivot v4 (train + val uniquement)

**Date** : YYYY-MM-DD
**Type** : Audit informatif. **Aucun verdict GO/NO-GO**.
**n_trials** : 22 (inchangé, c'est un replay du même n_trial H06=1).

## Comparaison train + val (test set non touché)

| Actif | Best (N,M) | Sharpe train v3 | Sharpe train v4 | Sharpe val v3 | Sharpe val v4 | DD val v3 | DD val v4 | Méthode Sharpe |
|---|---|---|---|---|---|---|---|---|
| US30 | (100, 10) | +0.35 | ? | +0.58 | ? | (n/a) | ≥ −15 % | weekly |
| XAUUSD | (100, 20) | +1.13 | ? | 0.00 | ? | (n/a) | ≥ −15 % | per_trade |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Interprétation

- Si **Sharpe train v4 ≥ 1.0** sur US30 → l'edge Donchian existait probablement, mais était caché par le simulateur cassé.
- Si **Sharpe train v4 ≥ Sharpe train v3 + 1.0** sur ≥ 3 actifs → confirmation que c'était bien un problème de coûts.
- Si **Sharpe val v4 dérive de Sharpe train v4** → overfitting normal (pas spécifique au pivot).
- Si **Sharpe val v4 reste négatif partout** → l'edge n'existe pas, même sans coûts excessifs.

## Décision

Cette section sert à informer la Phase B, **pas à statuer**. Le test set 2024+ n'est plus disponible pour ces hypothèses (brûlé). Les vrais GO/NO-GO viendront de H_new1, H_new2, H_new3 sur leur propre test set.
```

`docs/v3_hypothesis_07_replay.md` : structure identique.

### Étape 3 — Vérifications anti-snooping renforcées

Avant exécution, vérifier que :

```bash
# Le script DOIT contenir un filtre explicite
grep -n "CUTOFF_DATE" scripts/run_pivot_a4_replay.py
# Output attendu : ≥ 2 lignes (def + usage)

# Aucune ligne ne lit 2024 / 2025
grep -nE '"2024|"2025|"2026' scripts/run_pivot_a4_replay.py
# Output attendu : VIDE
```

### Étape 4 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_pivot_a4_replay.py
```

Sortie attendue :
```
Replay terminé. Résultats dans predictions/pivot_a4_replay.json
```

Aucun trade ni Sharpe ≥ 2024 ne doit apparaître.

### Étape 5 — Lecture & documentation

1. Lire `predictions/pivot_a4_replay.json`.
2. Remplir les tableaux des `_replay.md`.
3. Écrire l'interprétation dans chaque rapport.
4. Mettre à jour `JOURNAL.md`.

### Étape 6 — Validation finale

```bash
python scripts/verify_no_snooping.py  # doit retourner 0
rtk make verify                        # doit passer
```

## Tests unitaires associés

`tests/unit/test_pivot_a4_cutoff.py` (NOUVEAU) :

```python
"""Vérifie que le script de replay ne lit jamais ≥ 2024."""
from pathlib import Path
import re


def test_no_2024_2025_2026_literals():
    """Le script ne doit pas contenir de littéraux d'années test set."""
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    # Exclure les chaînes en commentaires
    code_lines = [line for line in script.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    forbidden = re.findall(r'"(2024|2025|2026)', code)
    assert not forbidden, f"Test set literals found: {forbidden}"


def test_cutoff_constant_present():
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    assert "CUTOFF_DATE" in script
    assert "2023-12-31" in script
```

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A4 : Replay H06/H07 (audit informatif)

- **Statut** : ✅ Terminé
- **Type** : Audit informatif (0 n_trial consommé)
- **Fichiers créés** : `scripts/run_pivot_a4_replay.py`, `tests/unit/test_pivot_a4_cutoff.py`, `docs/v3_hypothesis_06_replay.md`, `docs/v3_hypothesis_07_replay.md`, `predictions/pivot_a4_replay.json`
- **Périmètre** : train ≤ 2022 + val 2023 uniquement. Test set 2024+ NON LU.
- **Résultats clés** :
  - Donchian US30 D1 : Sharpe train v3 +0.35 → v4 ? (méthode Sharpe : weekly)
  - Donchian XAUUSD D1 : Sharpe train v3 +1.13 → v4 ?
  - Strats alt US30 : ...
- **Interprétation** : [edge train/val réapparaît / ne réapparaît pas]
- **n_trials** : 22 (inchangé)
- **verify_no_snooping.py** : ✅ exit 0
- **Notes** : Aucune décision GO/NO-GO. Sert uniquement à informer H_new1 et H_new3.
- **Prochaine étape** : B1 — H_new1 méta-labeling US30 D1 (vraie hypothèse OOS).
```

## Critères go/no-go

- **GO Phase B1** si :
  - Tous les tests A4 passent
  - `verify_no_snooping.py` retourne 0
  - Le test set 2024+ N'A PAS été lu (vérifié par audit de log)
  - Les rapports `_replay.md` sont remplis
- **NO-GO, revenir à** : si une lecture accidentelle du test set est détectée → ces actifs sont définitivement brûlés en OOS, ne plus jamais y toucher.

## Annexes

### A1 — Pourquoi cette étape est risquée

Il est tentant pour Deepseek (ou un humain) de "vouloir voir aussi le test set 2024 pour comparer". **C'est interdit.** Le test set OOS d'un H_i ne se lit qu'une fois. H06 et H07 ont déjà épuisé leur lecture en v3.

Le replay est une **inspection rétrospective** qui n'est valide statistiquement que parce qu'elle n'affecte pas la décision finale. Si on lisait le test set ici, on aurait deux lectures → DSR cumulatif faux → invalidation de toute conclusion.

### A2 — Pourquoi 0 n_trial consommé

Une hypothèse statistique = une décision GO/NO-GO sur un test set. Ici on ne décide rien sur le test set. C'est juste un debug rétrospectif de l'infrastructure.

Si A4 montre que Donchian US30 a un edge train+val avec coûts corrigés, on **n'a pas le droit** de dire "donc Donchian US30 est GO". On peut juste dire "donc l'edge méritait probablement d'être testé avec méta-labeling" → c'est H_new1 qui sera testé sur un test set encore non lu (avec n_trial = 1).

### A3 — Et si A4 montre un edge train+val énorme ?

C'est possible. Dans ce cas, ne pas céder à la tentation de "tester juste pour voir". Le test set 2024+ est brûlé pour H06/H07. On passe directement à H_new1 (méta-labeling) qui est une **nouvelle hypothèse** avec son propre test set.

Si Donchian US30 baseline a Sharpe train +1.5 et val +1.2 → **bonne nouvelle** : l'edge baseline est probablement positif, donc le méta-labeling H_new1 a de bonnes chances.
Si Donchian US30 baseline a Sharpe train −0.5 → **mauvaise nouvelle** : H_new1 a peu de chances. Considérer directement H_new3 (mean-reversion EURUSD).

## Fin du prompt A4.
**Suivant** : [B1_meta_us30.md](B1_meta_us30.md)
