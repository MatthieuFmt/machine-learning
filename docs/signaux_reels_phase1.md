# Signaux RÉELS trouvés (Phase 1 recherche d'edge) — à conserver

**Dernière mise à jour** : 2026-06-09

> 🚨 **AVERTISSEMENT (audit 2026-06-09) — les DSR/p-values ci-dessous sont CADUCS.**
> Un bug de calcul a été découvert et corrigé : le DSR recevait le Sharpe
> **annualisé** avec n_obs = nb de trades, gonflant le z jusqu'à ×√252 (≈×16)
> (`validate_edge` et `screen_carry._metrics`). Conséquences :
> - **ORB M5 « DSR +11.29 (p=0.000) » est un artefact.** Recalcul à la main :
>   Sharpe/trade ≈ 0.011 → z ≈ 0.6, p ≈ 0.27 → **bruit statistique**.
> - Le « t≈2.5-2.8 » du pre-FOMC et le DSR du carry passaient par le même
>   chemin → **à re-mesurer** avant toute décision.
> Les scripts sont corrigés (DSR canonique + t-test par trade + bootstrap +
> registre n_trials). **UNE re-mesure par signal**, puis mettre à jour ce mémo :
> ```bash
> python scripts/screen_pre_fomc.py --assets US500,US30 --tf H1
> python scripts/screen_orb_fine.py --assets US500 --tf M5 --or-minutes 5
> python scripts/screen_carry.py --assets AUDJPY,GBPJPY,EURJPY --tf D1
> ```

> Mémo de référence : les 3 seuls signaux qui montraient un effet jugé réel
> après validation honnête (fill réaliste, coûts XTB, swap, DSR avec n_trials).
> **Aucun ne franchit seul** la barre constitution (Sharpe ≥ 1 · DSR > 0 p<0.05 ·
> MaxDD < 15 % · WR > 30 % · ≥ 30 trades/an). Mais ils étaient jugés **réels,
> décorrélés et à friction légère** → candidats pour un portefeuille combiné.

---

## Les 3 signaux réels

| Signal | Actif(s) | Sharpe | DSR (preuve) | MaxDD | Fréquence | Friction |
|---|---|---|---|---|---|---|
| **Pre-FOMC drift** | US500 (ou US30) | ~0,7 | t≈2,5-2,8 ; DSR borderline | faible | ~8 trades/an | 1 nuit (peu de swap) |
| **Carry JPY** (panier) | AUDJPY+GBPJPY+EURJPY | ~0,4 | positif faible | **17-30 %** (krachs) | continu | swap = revenu (+0,7 à +3 %/an) |
| **ORB US500 5 min** | US500 | **0,17** | **+11,29 (p=0,000)** ✅ le mieux prouvé | **6,7 %** | ~257 trades/an | **zéro** (intraday, flat la nuit) |

### Détails & reproduction

**1. Pre-FOMC drift** — long l'indice US ~24h avant la décision Fed, sortie ~1h avant.
Effet documenté (Lucca & Moench 2015). Réel mais rare (~8/an) et Sharpe modeste.
```
python scripts/screen_pre_fomc.py --assets US500,US30 --tf H1
```

**2. Carry JPY** — long les paires yen à fort différentiel de taux, le swap paie le portage.
Anomalie FX la plus documentée. Talon d'Achille : krachs de carry (DD 20-30 %).
⚠️ Swaps XTB PROVISOIRES (estimés) — à confirmer en démo.
```
python scripts/screen_carry.py --assets AUDJPY,GBPJPY,EURJPY --tf D1
```

**3. ORB US500 5 min** — Opening Range Breakout : range des 5 premières minutes de
séance NYSE, cassure confirmée à la close, entrée open suivant, stop côté opposé,
sortie le soir (flat la nuit). Effet documenté (Zarattini & Aziz 2023).
**Edge le mieux prouvé du projet** (DSR +11, p=0,000, DD 6,7 %, stable-hausse
0,10→0,22 post-2020), mais Sharpe 0,17 = économiquement minuscule (~+0,6 %/an).
Données : `data/raw/US500/US500_M5.csv` (703 782 bougies 2015-2026).
```
python scripts/download_orb_data.py --asset US500 --tf M5 --start 2015 --end 2026
python scripts/screen_orb_fine.py
```

---

## Idée de portefeuille (piste principale)

Combiner les 3 (pondérés au risque). Diversification réelle surtout entre
**actions US** (pre-FOMC + ORB, partiellement corrélés via le S&P) et **carry yen**
(forex/taux, décorrélé). Estimation : Sharpe **~0,7-0,8** (réel, bas risque, sans
doute sous 1,0 mais potentiellement testable en démo).

---

## Familles MORTES (NO-GO, ne pas refaire à l'identique)

10 familles testées. Tout le reste = pas d'edge ou friction > edge :
tendance, retour-moyenne, filtre de régime (technique D1/H4) ; turn-of-month ;
pré-NFP / pré-CPI (l'effet pre-FOMC ne se généralise pas) ; effet overnight ;
pairs trading or/argent (edge brut +5351 € mangé par spread+swap, NO-GO même
swap=0) ; Asian Range Breakout forex (cassures = fakeouts) ; ORB indices H1
(trop grossier, plat). Détails dans la mémoire projet et les post-mortems.

**Leçon transverse** : le swap CFD tue les holds multi-jours ; l'intraday évite le
swap mais les edges accessibles sont petits ; les seuls effets réels sont faibles.
