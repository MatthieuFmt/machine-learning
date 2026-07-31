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

---

# ✅ RELEVÉS EFFECTUÉS — 2026-07-31 23:07-23:08 (heure française, UTC+2)

Source : captures de l'app mobile XTB, compte du mainteneur.
⚠️ **Heure de relevé = pré-ouverture / clôture NY = le PIRE moment de la journée
pour les spreads.** Les valeurs ci-dessous sont donc des **majorants**.

## Spreads mesurés

| Instrument | Spread relevé | Coût affiché | Estimation du code (avant) | Verdict |
|---|---|---|---|---|
| **US500** | **0,92 point d'indice** (9,2 pips) | 0,20 EUR / 0,005 lot | 0,06 point | 🔴 **sous-estimé ×15** |
| GBPJPY | 3,1 pips | 0,17 EUR / 0,01 lot | 3,1 pips | ✅ exact |
| EURJPY | 2,9 pips | 0,16 EUR / 0,01 lot | 1,9 pips | 🟠 optimiste ×1,5 |
| AUDJPY | **26 pips** | 1,44 EUR / 0,01 lot | 2,2 pips | ⚠️ heure morte (AUD sans séance) |

**Commission : 0,00 EUR confirmée** sur les 4 instruments (compte Standard).

### Comment le spread US500 a été déduit (le « pip » XTB ≠ le pip du code)
```
valeur du contrat 1 636,33 EUR pour 0,005 lot, indice à 7 503
→ 1 POINT d'indice vaut 1636.33/7503 = 0,2181 EUR
→ « valeur du pip » affichée = 0,22 EUR  ⇒ le pip XTB = 1 POINT d'indice
→ spread 0,20 EUR ÷ 0,2181 = 0,92 POINT   (= 9,2 pips internes, pip_size=0,1)
```
➡️ **Impact** : `ASSET_CONFIGS["US500"]` corrigé (spread 0,5 → 9,2 pips).
C'est ce qui rendait l'ORB M5 (~257 allers-retours/an) faussement viable :
257 × 2 × 0,92 ≈ **473 points d'indice de frais par an**, soit ~6 % du notionnel.

## Swaps mesurés

| Instrument | Swap relevé | Estimation du code | Verdict |
|---|---|---|---|
| **US500** | Achat **−0,021167 %/nuit**, Vente −0,001056 % | −16 pips = −1,60 pt | ✅ **juste à 1 % près** |
| AUDJPY / GBPJPY / EURJPY | ❌ **NON RELEVÉ** | +0,9 / +1,0 / +0,3 pips | 🔜 à faire |

`−0,021167 % × 7 503 = −1,59 point/nuit` ⇒ l'estimation −1,60 était bonne.
Pour le pre-FOMC (1 nuit, position longue) : **−0,35 EUR sur 1 636 EUR** = négligeable.

## ⚠️ US500 est un CFD sur FUTURES, pas sur cash

Écran « Informations clés » : *S&P500 index futures contract (CFD)*, avec
**rollover trimestriel** — dernier **17/06/2026**, suivant **16/09/2026**,
levier 1:20 (marge 5 %), taille min **0,005** lot, max 52 lots.

🚨 **Le rollover du 16/09/2026 tombe le jour même de la décision FOMC de
septembre** (réunion 15-16/09). La fenêtre du trade pre-FOMC (J−24h → J−1h)
chevauche donc le rollover. À vérifier avant de jouer cette date : soit passer
sur US30, soit décaler, soit contrôler le traitement du rollover par XTB.

## Ce qui reste à relever

- [ ] **Swaps AUDJPY / GBPJPY / EURJPY** → onglet « Afficher les détails » sur la
      page de l'instrument (celui replié sur les captures du 31/07).
- [ ] **Spread US500 en séance NY** (15h30-22h FR) → la valeur 0,92 est un
      pire-cas ; la vraie valeur de session affinerait le pre-FOMC.
- [ ] **Spread AUDJPY en séance Tokyo** (02h-09h FR) → les 26 pips relevés sont
      une anomalie d'heure creuse.
- [ ] Instruments basés futures sans swap quotidien (OIL, NATGAS, agricoles).
