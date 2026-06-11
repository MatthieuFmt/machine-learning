# Checklist — Relever les VRAIS coûts XTB (compte démo, ~15 min)

**Pourquoi** : tous les backtests utilisent des coûts *estimés* (certains marqués
PROVISOIRES dans `app/config/instruments.py`). Deux décisions entières dépendent
des vrais chiffres :
1. **Carry JPY** : l'edge EST le swap. Si le swap réel est plus bas qu'estimé,
   la stratégie meurt. S'il est plus haut, elle s'améliore.
2. **Instruments « sans swap »** : certains CFD basés sur des contrats à terme
   (pétrole, gaz, agricoles…) n'ont pas de frais de nuit quotidiens →
   ré-ouvrirait le suivi de tendance multi-jours à coût faible.

En attendant ces relevés, les screens appliquent une **marge de sécurité ×1.5**
sur les coûts (`--cost-margin`).

---

## Étape 0 — Compte démo

Créer un compte démo XTB (gratuit, sans dépôt) : xtb.com → « Compte démo ».
Ouvrir la plateforme **xStation 5**.

## Étape 1 — Swap réel par instrument (le plus important)

Pour CHAQUE instrument ci-dessous : clic droit sur l'instrument → **« Informations
sur l'instrument »** (ou icône ⓘ). Noter :

| Champ xStation | Quoi noter |
|---|---|
| **Swap long (points)** | frais/crédit par nuit si tu es ACHETEUR |
| **Swap short (points)** | idem si tu es VENDEUR |
| **Triple swap** | quel jour la nuit compte triple (souvent mercredi ou vendredi) |
| **Valeur du point / taille de lot** | pour convertir en € |

Instruments à relever (dans cet ordre de priorité) :

- [ ] **AUDJPY, GBPJPY, EURJPY, USDJPY** (décide du carry)
- [ ] **US500, US30** (coût de la nuit du trade pre-FOMC)
- [ ] **XAUUSD (or)**
- [ ] **EURUSD, GBPUSD**
- [ ] **OIL / OIL.WTI, NATGAS** + 2-3 agricoles (COFFEE, WHEAT…) :
      noter si la fiche dit **« basé sur contrat à terme / rollover »**
      et si le swap est nul ou quasi nul → liste des instruments « sans swap »
- [ ] **BTCUSD, ETHUSD** (swap crypto = souvent énorme, à confirmer)

## Étape 2 — Spread selon l'heure (3 relevés par instrument)

Le spread affiché change selon l'heure. Pour EURUSD, US500, XAUUSD, AUDJPY :

- [ ] Vers **9h00** Paris (session Europe)
- [ ] Vers **15h35** Paris (ouverture US — heure du trade ORB)
- [ ] Vers **23h00** Paris (nuit — heure à ÉVITER)

Noter le spread en pips/points à chaque heure.

## Étape 3 — Tailles minimales

Pour chaque instrument relevé : noter le **lot minimum** (souvent 0.01) et la
**marge requise** pour 0.01 lot. Ça décide de ce qui est jouable avec un petit
capital.

## Étape 4 — Reporter dans le code

Donner les chiffres à l'assistant (ou éditer soi-même
`app/config/instruments.py` : champs `spread_pips`,
`swap_long_pips_per_night`, `swap_short_pips_per_night`) puis :

```bash
rtk pytest tests/unit/test_c1_asset_configs_extended.py
```

Ensuite seulement, relancer les screens avec `--cost-margin 1.0` (coûts réels,
plus besoin de marge).
