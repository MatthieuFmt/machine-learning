# Pivot v4 — A2 : Calibration des coûts XTB réels

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md) — section "Ordre d'exécution strict — RÉVISÉ"
2. [A1_audit_simulator.md](A1_audit_simulator.md) — **DOIT ÊTRE ✅ Terminé**
3. [A9_pipeline_lock.md](A9_pipeline_lock.md) — **DOIT ÊTRE ✅ Terminé** (ordre révisé : ML d'abord)

> 📌 **Note ordre révisé** : A2 vient maintenant APRÈS le bloc ML (A5-A9). Le pipeline ML est déjà figé ; ici on finalise le simulateur (coûts XTB) avant B1.
3. [../00_constitution.md](../00_constitution.md) — règle 13 (anti-hallucination API)
4. [app/config/instruments.py](../../app/config/instruments.py) — table `ASSET_CONFIGS` à corriger
5. [docs/v3_hypothesis_06.md](../../docs/v3_hypothesis_06.md) — anciens coûts qui ont tué H06

## Objectif
Remplacer les coûts surestimés × 3 à × 80 dans `ASSET_CONFIGS` par les **vrais spreads et slippages XTB** documentés par la plateforme officielle XTB et **les vérifier par capture en compte démo si possible**.

## Type d'opération
🔧 **Bug fix infrastructure** — 0 n_trial consommé.

## Definition of Done (testable)

- [ ] `docs/cost_audit_v2.md` créé avec :
  - Tableau par actif : coût v3 ancien vs spread XTB officiel vs nouveau coût v4
  - Source documentaire (URL XTB ou capture démo MT5 fournie par l'utilisateur)
  - Date de capture
  - Justification du choix de chaque coût
- [ ] [app/config/instruments.py](../../app/config/instruments.py) `ASSET_CONFIGS` mis à jour pour les 7 actifs + EURUSD ajouté.
- [ ] Tests `tests/unit/test_instruments_costs.py` (NOUVEAU) : ≥ 5 tests qui valident :
  - `US30.spread_pips ≤ 2.0`
  - `XAUUSD.total_cost_pips ≤ 1.0` (en USD si pip_size cohérent)
  - `XAGUSD.total_cost_pips ≤ 0.05` (USD)
  - `EURUSD` présent dans `ASSET_CONFIGS`
  - Pour chaque actif : `total_cost_pips / sl_points ≤ 0.10` (coût ≤ 10 % du SL, sinon stratégie impossible)
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` mis à jour

## NE PAS FAIRE

- ❌ Ne PAS inventer des coûts (Règle 13 anti-hallucination). Si tu ne trouves pas la source officielle XTB pour un actif → laisser un commentaire `# TBD: cost source missing for ASSET` et raise `NotImplementedError` à l'usage.
- ❌ Ne PAS modifier `pip_size` et `pip_value_eur` sans documenter pourquoi (impact massif sur le sizing).
- ❌ Ne PAS toucher au test set.
- ❌ Ne PAS commencer A3 avant que cette étape soit ✅.
- ❌ Ne PAS supprimer les anciens coûts du code commit (les commenter à côté pour traçabilité).

## Étapes détaillées

### Étape 1 — Recherche documentaire (manuelle utilisateur ou doc XTB)

**À demander à l'utilisateur** :
1. Capture d'écran de la table des spreads XTB (https://www.xtb.com → Trading conditions → CFD list).
2. Ou export MT5 : "Symbol Specifications" pour chaque actif testé.
3. Heures de capture (les spreads varient : ouverture/clôture, news, weekend).

À défaut, utiliser les valeurs **officielles publiques XTB Standard Account 2025** :

| Actif | Symbole XTB | Spread typique (heures actives) | Pip XTB |
|---|---|---|---|
| US30 (Dow Jones) | `US30` ou `USA30` | **1.5 pts** | 1.0 pt |
| US500 (S&P) | `US500` ou `SP500` | **0.5 pts** | 0.1 pt |
| GER30 (DAX) | `DE30` ou `GER40` | **1.0 pts** | 1.0 pt |
| XAUUSD (Or) | `GOLD` | **0.30 USD** | 0.01 USD |
| XAGUSD (Argent) | `SILVER` | **0.025 USD** | 0.001 USD |
| USOIL (WTI) | `OIL.WTI` | **0.05 USD** | 0.01 USD |
| EURUSD | `EURUSD` | **0.7 pip** | 0.0001 |
| BTCUSD | `BITCOIN` | **30 USD** | 1.0 USD |

> ⚠️ Ces chiffres sont des **ordres de grandeur de référence pour XTB Standard Account**. À confirmer avant prod. Marges XTB Pro Account ou demo Account peuvent différer.

### Étape 2 — Slippage réaliste

Le slippage dépend de :
- La taille de l'ordre (petits ordres = quasi nul)
- L'heure (overlap Londres-NY = liquide → slippage minimal)
- La volatilité (news = slippage élevé)

**Règle d'ordre de grandeur** pour positions ≤ 5 lots, hors news :
- D1 / H4 / H1 sur **majeures liquides** (US30, US500, EURUSD, GOLD) : `slippage ≈ 0.2 × spread`
- Sur **mineures** (XAGUSD, USOIL) : `slippage ≈ 0.5 × spread`
- Sur **crypto** (BTCUSD, ETHUSD) : `slippage ≈ 1.0 × spread`

### Étape 3 — Nouveau `ASSET_CONFIGS`

Remplacer dans `app/config/instruments.py` :

```python
# ─────────────────────────────────────────────────────────────────────────
# ASSET_CONFIGS v4 (pivot) — coûts XTB Standard Account, capture 2026-MM-DD
# Source : docs/cost_audit_v2.md
# ─────────────────────────────────────────────────────────────────────────

ASSET_CONFIGS: dict[str, AssetConfig] = {
    "US30": AssetConfig(
        # v3: spread=3.0 + slippage=5.0 = 8.0  ← surestimation × 5
        # v4: vrai XTB Standard ~1.5
        spread_pips=1.5,
        slippage_pips=0.3,
        commission_pips=0.0,
        pip_size=1.0,
        pip_value_eur=0.92,
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
    ),
    "US500": AssetConfig(
        # v3: 3.5  ← surestimation × 7
        # v4: vrai XTB ~0.5
        spread_pips=0.5,
        slippage_pips=0.1,
        commission_pips=0.0,
        pip_size=0.1,        # ⚠️ pip US500 = 0.1, pas 1.0
        pip_value_eur=0.092,
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
    ),
    "GER30": AssetConfig(
        # v3: 5.0  ← surestimation × 5
        # v4: vrai XTB ~1.0
        spread_pips=1.0,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=1.0,
        pip_value_eur=1.0,
        tp_points=400,
        sl_points=200,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
    ),
    "XAUUSD": AssetConfig(
        # v3: 35.0 USD  ← surestimation × 100 (confusion pip_size)
        # v4: spread XTB ≈ 0.30 USD avec pip_size 0.01 USD = 30 "pips"
        # En cohérence avec ancienne définition: pip_size=1.0 USD → spread = 0.30 USD = 0.30 pip
        spread_pips=0.30,
        slippage_pips=0.05,
        commission_pips=0.0,
        pip_size=1.0,        # 1 pip XTB GOLD = 1 USD pour facilité (à confirmer broker)
        pip_value_eur=0.92,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
    ),
    "XAGUSD": AssetConfig(
        # v3: 45.0  ← surestimation × 1500 (confusion d'unité)
        # v4: spread XTB ≈ 0.025 USD avec pip_size 0.001
        spread_pips=0.025,
        slippage_pips=0.01,
        commission_pips=0.0,
        pip_size=0.001,      # ⚠️ pip XTB SILVER = 0.001 USD
        pip_value_eur=0.92,
        tp_points=300,       # = 0.30 USD soit 1.5 % du prix typique
        sl_points=150,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
    ),
    "USOIL": AssetConfig(
        # v3: 7.0  ← surestimation × 70 (confusion pip_size)
        # v4: spread XTB ≈ 0.05 USD avec pip_size 0.01
        spread_pips=0.05,
        slippage_pips=0.02,
        commission_pips=0.0,
        pip_size=0.01,
        pip_value_eur=0.92,
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
    ),
    "EURUSD": AssetConfig(
        # NOUVEAU en v4 — absent de v3
        spread_pips=0.7,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=0.0001,     # ⚠️ pip forex standard = 0.0001
        pip_value_eur=10.0,  # 1 pip × 1 lot standard (100k) = 10 USD ≈ 9.2 EUR à 1 EUR=1.0875 USD
        tp_points=20,        # 20 pips
        sl_points=10,        # 10 pips
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
    ),
    # ⚠️ BUND désactivé : données indisponibles
    # "BUND": AssetConfig(...),
    # ⚠️ BTCUSD et ETHUSD ajoutables si demande utilisateur
}
```

### Étape 4 — Tests unitaires

`tests/unit/test_instruments_costs.py` :

```python
"""Tests des coûts post pivot v4 A2."""
from __future__ import annotations

import pytest
from app.config.instruments import ASSET_CONFIGS


def test_us30_spread_realistic():
    assert ASSET_CONFIGS["US30"].spread_pips <= 2.0
    assert ASSET_CONFIGS["US30"].total_cost_pips <= 2.5


def test_xauusd_costs_realistic():
    cfg = ASSET_CONFIGS["XAUUSD"]
    # Avec pip_size=1.0, spread doit être ≤ 0.5 (USD)
    assert cfg.spread_pips * cfg.pip_size <= 1.0


def test_xagusd_costs_realistic():
    cfg = ASSET_CONFIGS["XAGUSD"]
    # spread total en USD
    assert cfg.spread_pips * cfg.pip_size <= 0.05


def test_eurusd_present():
    assert "EURUSD" in ASSET_CONFIGS
    cfg = ASSET_CONFIGS["EURUSD"]
    assert cfg.pip_size == 0.0001


@pytest.mark.parametrize("asset", list(ASSET_CONFIGS.keys()))
def test_cost_vs_sl_ratio(asset):
    """Le coût total ne doit pas dépasser 10 % du SL (sinon stratégie impossible)."""
    cfg = ASSET_CONFIGS[asset]
    ratio = cfg.total_cost_pips / cfg.sl_points
    assert ratio <= 0.10, (
        f"{asset}: coût {cfg.total_cost_pips} > 10% du SL {cfg.sl_points}. "
        f"Stratégie mathématiquement impossible."
    )
```

### Étape 5 — Documentation `docs/cost_audit_v2.md`

Format :

```markdown
# Audit des coûts v4 (pivot)

**Date** : YYYY-MM-DD
**Source** : XTB Standard Account, capture utilisateur du YYYY-MM-DD
**Type de compte** : Standard (à vérifier — Pro si différent)

## Comparaison avant/après

| Actif | spread v3 | slippage v3 | total v3 | spread v4 | slippage v4 | total v4 | Facteur correction |
|---|---|---|---|---|---|---|---|
| US30 | 3.0 | 5.0 | 8.0 | 1.5 | 0.3 | 1.8 | ÷ 4.4 |
| US500 | 1.5 | 2.0 | 3.5 | 0.5 | 0.1 | 0.6 | ÷ 5.8 |
| GER30 | 2.0 | 3.0 | 5.0 | 1.0 | 0.2 | 1.2 | ÷ 4.2 |
| XAUUSD | 25 | 10 | 35 | 0.30 | 0.05 | 0.35 | ÷ 100 |
| XAGUSD | 30 | 15 | 45 | 0.025 | 0.01 | 0.035 | ÷ 1285 |
| USOIL | 4 | 3 | 7 | 0.05 | 0.02 | 0.07 | ÷ 100 |
| EURUSD | absent | absent | — | 0.7 | 0.2 | 0.9 | nouveau |

## Justification par actif

### US30
- Source : XTB.com → Trading conditions → CFD → US30
- Spread moyen heures actives : 1.5 pts
- Spread off-hours : jusqu'à 4 pts (à exclure du backtest)
- Slippage estimé pour ordres ≤ 2 lots : 0.3 pts (basé sur liquidité du Dow CFD)

### XAUUSD
- Source : XTB.com → Spot Gold
- Spread moyen : 0.30 USD (parfois 0.50 en news)
- pip_size = 1.0 USD pour rester cohérent avec la définition v3 (un "pip" XTB GOLD = 1 USD à fin de journée).
- ⚠️ Ancien code utilisait "25 pips" = 25 USD → 80× le spread réel.

### XAGUSD
- Source : XTB.com → Spot Silver
- Spread moyen : 0.025 USD (= 2.5 cents)
- pip_size = 0.001 USD pour cohérence (1 "pip" XTB SILVER = 1/10 cent)
- ⚠️ Ancien code utilisait "30 pips" sans pip_size cohérent → blow-up immédiat.

### EURUSD (nouveau)
- Source : XTB.com → Forex
- Spread moyen : 0.7 pip (heures Londres/NY)
- pip_size = 0.0001 (standard forex)
- pip_value_eur = 10 EUR par lot standard (100 000) — taux EUR/USD ≈ 1.0875

## Impact attendu

Sur Donchian US30 D1 (91 trades en H06 test) :
- Coût v3 : 91 × 8 pts × 0.92 € × 2.17 lots = **1 451 €** sur 10 000 € = 14.5 % du capital absorbé en frais
- Coût v4 : 91 × 1.8 pts × 0.92 € × 2.17 lots = **327 €** sur 10 000 € = 3.3 %
- Économie estimée : **−11.2 % du capital** sur 18 mois → **Sharpe brut + ~1.5**

## Limites
- Ces coûts sont des **moyennes**. La réalité oscille selon volatilité, news, heure.
- Le slippage est **stochastique** (cf. prompt 09 du plan v3) — un slippage stochastique uniform[min, max] sera implémenté dans la phase B si nécessaire.
- Ne pas confondre **XTB Standard** (spreads variables, 0 commission) et **XTB Pro** (spreads serrés, commission ~3.5 USD/lot).
- À reconfirmer 1× par trimestre (les coûts XTB peuvent changer).
```

## Tests unitaires associés

Listés en Étape 4. `tests/unit/test_instruments_costs.py` — 4 + N tests parametrisés (1 par actif).

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A2 : Calibration coûts XTB réels

- **Statut** : ✅ Terminé
- **Type** : Bug fix infrastructure (0 n_trial consommé)
- **Fichiers créés** : `docs/cost_audit_v2.md`, `tests/unit/test_instruments_costs.py`
- **Fichiers modifiés** : `app/config/instruments.py` (ASSET_CONFIGS v4)
- **Coûts corrigés** : US30 ÷ 4.4, US500 ÷ 5.8, GER30 ÷ 4.2, XAUUSD ÷ 100, XAGUSD ÷ 1285, USOIL ÷ 100
- **Actifs ajoutés** : EURUSD
- **Tests** : 4 + 7 parametrisés = 11/11 passed
- **make verify** : ✅ passé
- **Impact attendu** : Donchian US30 D1 frais 14.5 % → 3.3 % du capital → Sharpe brut probable +1.5
- **Notes** : Aucune stratégie modifiée, aucune lecture test set.
- **Prochaine étape** : A3 — fix Sharpe stratégies faible fréquence
```

## Critères go/no-go

- **GO Phase A3** si :
  - Tests `test_instruments_costs.py` 11/11 passent
  - `docs/cost_audit_v2.md` complet avec sources
  - L'utilisateur a validé les coûts (capture ou doc XTB officielle)
- **NO-GO, revenir à** : si l'utilisateur ne peut pas fournir de capture → utiliser les valeurs publiques XTB documentées **et** annoter `docs/cost_audit_v2.md` en "⚠️ Coûts non confirmés en démo, à valider avant prod".

## Annexes

### A1 — Pourquoi le facteur × 80 sur XAUUSD ?

Confusion d'unité. Dans v3 :
- `pip_size = 1.0` (1 pip = 1 USD)
- `spread_pips = 25.0` → spread = 25 × 1.0 = 25 USD

Spread XTB réel : 0.30 USD. Soit 80× plus bas.

Source de la confusion : forex/CFD ont des notions différentes du "pip" :
- Forex EURUSD : 1 pip = 0.0001 (4ème décimale)
- Indice US30 : 1 pip = 1 point (entier)
- Or XAUUSD : ambigu — XTB parfois "1 pip = 1 USD" (big figure), parfois "1 pip = 0.01 USD" (pipette)

**Convention adoptée v4** :
- US30, GER30 : pip_size = 1.0 (1 point = 1 USD/EUR)
- US500 : pip_size = 0.1 (le S&P cote au dixième)
- XAUUSD : pip_size = 1.0, spread_pips = 0.30 (spread = 0.30 USD)
- XAGUSD : pip_size = 0.001, spread_pips = 25 (spread = 0.025 USD)
- USOIL : pip_size = 0.01, spread_pips = 5 (spread = 0.05 USD)
- EURUSD : pip_size = 0.0001, spread_pips = 0.7 (standard forex)

### A2 — Comment vérifier en compte démo

1. Ouvrir un compte démo XTB (gratuit, sans engagement).
2. MT4/MT5 → Market Watch → clic droit sur chaque symbole → "Specifications".
3. Noter `Spread` (en pips ou points selon symbole).
4. Comparer avec `ASSET_CONFIGS` v4.
5. Si écart > 30 %, corriger et noter dans `docs/cost_audit_v2.md`.

### A3 — Pourquoi un commission_pips = 0 sur Standard Account

XTB Standard Account : spreads variables, **pas de commission**. Toutes les frictions sont dans le spread + slippage.
XTB Pro Account : spreads serrés (~0.1 pt US30), **commission ~3.5 USD/lot**. À paramétrer si l'utilisateur passe en Pro.

## Fin du prompt A2.
**Suivant** : [A3_sharpe_low_frequency.md](A3_sharpe_low_frequency.md)
