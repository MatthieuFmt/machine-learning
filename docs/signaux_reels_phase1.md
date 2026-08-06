# Signaux RÉELS trouvés (Phase 1 recherche d'edge) — à conserver

**Dernière mise à jour** : 2026-08-06

> # ☠️ CE DOCUMENT N'A PLUS DE SURVIVANT (2026-08-06)
>
> Les 2 derniers screens ont tourné sur les vraies bougies. **Aucun des 4
> candidats du projet n'est validé.** Le titre « Signaux RÉELS » est caduc :
> il décrit ce qu'on a cru avoir, pas ce qu'on a.
>
> | Signal | Statut au 2026-08-06 | Motif |
> |---|---|---|
> | ORB US500 M5 | ☠️ **MORT** | t=0,56 · p=0,287 — artefact du bug « DSR ×√252 » |
> | Carry JPY | ☠️ **MORT** | swap réel +0,16 %/an au lieu de +0,7 à +3 % |
> | Tendance crypto | ☠️ **MORTE** | financement XTB mesuré à 35,4 %/an |
> | **Pre-FOMC US500** | ❌ **NO-GO** | médiane/trade **négative**, 76 % du gain sur 5 trades, réunion d'urgence COVID = 20 % du gain |
> | **Pre-FOMC US30** | ⏸️ **NON CONCLUABLE** | repose sur un spread **jamais relevé** ; ×9,1 d'erreur l'annule |
> | **Pre-ECB GER30** | ❌ **NO-GO** | t=0,53 · p=0,298 — rien, coûts mesurés |
>
> Détail complet des mesures : `JOURNAL.md`, section **2026-08-06**.
> Le reste de ce fichier est conservé comme **archive** de l'état antérieur.
> ⚠️ Ne plus s'appuyer sur les verdicts ci-dessous sans lire le bandeau.

---

## 🔻 Re-mesure FINALE sur bougies (2026-08-06) — ce qui remplace tout le reste

Données enfin versionnées (US500/US30/GER30 H1 + calendrier 2010-2025), screens
lancés **une seule fois chacun**, conformément à la règle « un seul regard ».

| | trades | Sharpe | **t/trade (preuve primaire)** | p_bootstrap | DSR | médiane/trade |
|---|---|---|---|---|---|---|
| US500 pre-FOMC | 112 | 0,57 | 2,11 · p=0,018 | 0,028 | 0,16 (p=0,436) | **−16,2 pips** |
| US30 pre-FOMC | 109 | 0,73 | 2,48 · p=0,007 | 0,012 | 0,93 (p=0,177) | +13,4 pips |
| GER30 pre-ECB | 117 | 0,14 | **0,53 · p=0,298** | 0,216 | −1,72 (p=0,957) | — |

### Pourquoi un t-test à p<0,05 ne suffit PAS ici
1. **Médiane US500 négative** (−16,2 pips) : le trade typique perd. WR 48 %.
2. **5 trades = 76 % du gain** (US500) ; 3 trades = 51 %.
3. **Le meilleur trade des 2 actifs est le 2020-03-03**, réunion d'**urgence
   COVID non programmée** → aucune fenêtre d'anticipation possible. 20-22 % du
   gain total. Hors cette réunion, US500 tombe à **t=1,84**.
4. **Queues trop épaisses pour Student** : kurtosis 4,46 (US500) et **10,82**
   (US30) → le t-test sur-rejette, le p nominal est optimiste.
5. **US500 et US30 ne sont pas indépendants** (corrélation ~0,95) : un test, pas deux.

### Le test de décroissance pré/post-2015 est IMPOSSIBLE sur nos données
Affiché : −26,6 pips avant 2015 (24 tr) → +69,4 après (88 tr). **Ce n'est pas un
démenti de Kurov, Wolfe & Gilbert (2021)** : nos prix commencent le 2012-01-16,
alors que Lucca & Moench portaient sur 1994-2011. Le bloc « avant 2015 » ne
couvre que 3 ans, **entièrement postérieurs à l'étude d'origine** → il n'existe
aucune période pré-publication dans l'échantillon. Le `--split-year` ne peut pas
répondre à la question. (Le mémo du 2026-08-01 tablait sur ~40 trades avant
2015 : il y en a 24, et ils perdent de l'argent.)

Année par année (descriptif, aucun test dessus), US500 :
`2012:-7 2013:-25 2014:-48 2015:-24 2016:-10 2017:-0 2018:+42 2019:-47`
`2020:+256 2021:-41 2022:+298 2023:+65 2024:+173 2025:+52`
→ **2012-2019 plat à négatif**, tout le gain vient de 2020/2022/2024. Lecture
compatible avec « effet éteint depuis longtemps, reste de la volatilité ».

### Test placebo (contrôle qui manquait au screen)
La stratégie est **toujours longue** sur des indices qui ont beaucoup monté.
Contre 20 000 tirages de fenêtres de 23 h au hasard, mêmes coûts :

| | pre-FOMC | fenêtre au hasard | p empirique |
|---|---|---|---|
| US500 | +48,9 pips | −10,8 pips | 0,041 |
| US30 | +44,3 pips | −2,5 pips | 0,036 |

Ce n'est donc pas que du beta de marché — mais p≈0,04 sur un effet dont 20 %
provient d'une réunion surprise ne justifie pas d'engager de l'argent.

### 🚨 US30 : le meilleur chiffre repose sur le seul coût jamais mesuré
`ASSET_CONFIGS["US30"].spread_pips = 1.5` — commentaire « vrai XTB ~1.5 pts »,
**sans relevé, sans date, sans capture**. Même classe d'estimation que celles
trouvées fausses **×15 sur US500** et **×9,2 sur GER30**.

| | coût/trade EN PLUS qui ramène t à 1,66 | facteur de spread |
|---|---|---|
| US500 (mesuré, pire cas) | 10,3 pips | ×2,1 |
| US500 hors réunion d'urgence | 3,8 pips | ×1,4 |
| **US30 (estimé)** | 16,8 pips | **×12,2** |
| **US30 hors réunion d'urgence** | 12,1 pips | **×9,1** |

➡️ **×9,1 suffirait à annuler US30. L'erreur mesurée sur GER30 était ×9,2.**
**Rien ne sera conclu sur US30 avant une capture de l'app XTB.**

### Calendrier FOMC : ✅ vérifié, il n'est PAS le problème
`scripts/verify_fomc_calendar.py` → **0 doublon, 0 manquant sur 2010-2018
(période vérifiée)**. Les écarts s'expliquent tous :
- 2020-03-18 absente en local = réunion **annulée** (COVID) → **le local a
  raison, c'est la liste de référence du script qui est fausse** ;
- 2020-03-03 / 2020-03-15 « en trop » = les 2 décisions d'urgence, réelles mais
  **non programmées** → à **exclure** du screen (cf. point 3) ;
- 8 dates de 2026 absentes = le calendrier **s'arrête au 2025-12-31**
  → **l'OOS vierge ≥2026 est inutilisable** pour tout screen événementiel.

---

# 📦 ARCHIVE — état du document avant la re-mesure du 2026-08-06

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

---

## Re-verdict ANALYTIQUE (2026-07-31) — en attendant la re-mesure sur bougies

Les screens exigent les CSV locaux, indisponibles en session cloud (sources de
données bloquées par la politique réseau). Mais le bug corrigé était un bug
d'**algèbre** (Sharpe annualisé → par-période), pas de données : le verdict
corrigé se recalcule depuis les Sharpe/fréquences déjà publiés ci-dessous.

➡️ `python scripts/recheck_signals_from_stats.py` (reproductible, sans données)

| Signal | Sharpe/période | n_obs | **t-test (preuve primaire)** | Verdict |
|---|---|---|---|---|
| **Pre-FOMC** | 0,2475 | 128 | **t = 2,80 · p = 0,0030** | ~~✅ SURVIT~~ → ❌ **INFIRMÉ le 2026-08-06** (voir bandeau) |
| Carry JPY | 0,0252 | 4 032 | t = 1,60 · p = 0,055 | ☠️ **MORT** (voir ci-dessous) |
| ORB US500 M5 | 0,0106 | 2 827 | t = 0,56 · p = 0,287 | ❌ **BRUIT** — confirme l'artefact |

### ☠️ Carry JPY — ENTERRÉ par les relevés de coûts réels (2026-07-31)

Swaps mesurés sur l'app XTB (détail : `docs/checklist_couts_xtb.md`) :

| Paire | Swap long réel | Estimation du code | Carry réel /an |
|---|---|---|---|
| AUDJPY | +0,18 pips | +0,9 (×5 optimiste) | +0,60 % |
| EURJPY | **−0,36 pips** | +0,3 (**signe inverse**) | **−0,73 %** |
| GBPJPY | +0,36 pips | +1,0 (×2,8 optimiste) | +0,62 % |

**Carry réel du panier : +0,16 %/an** contre +0,7 à +3 %/an supposés.
Être long EURJPY **coûte** de l'argent chaque nuit → la prémisse de la stratégie
(« le swap paie le portage ») est fausse chez ce courtier. Il faudrait ~145 ans de
swap pour rembourser une seule mauvaise année (perte max 17-30 %).

➡️ **Deux raisons indépendantes** (statistique + économique). Famille close,
ne pas ré-ouvrir. Les swaps côté SHORT étaient, eux, bien estimés : l'erreur
était unilatérale et favorisait précisément l'hypothèse testée.

**Le t-test est la preuve la plus robuste** : contrairement au DSR, il ne dépend
pas du nombre d'hypothèses testées. Un signal qui échoue au t-test est mort quelle
que soit l'hypothèse retenue sur `n_trials`.

Sur le DSR, la colonne à lire pour ces 3 signaux est **`n_trials = 1`** : les
trois sont des hypothèses **pré-enregistrées** (publiées dans la littérature
AVANT nos tests — Lucca & Moench 2015, carry FX, Zarattini & Aziz 2023), donc non
data-minées par nous. Pre-FOMC y obtient z = +2,75 (p = 0,003), robuste au
scénario « queues épaisses » (z = +2,55). Avec la pénalité cumulée du projet
(n_trials 15-60) il tomberait à p = 0,16-0,34 — mais appliquer notre compteur de
snooping à une hypothèse qu'on n'a pas cherchée serait une double peine injustifiée.

### Ce que ce re-verdict NE tranche PAS
- **La décroissance post-publication du pre-FOMC** (l'effet a-t-il survécu après
  2015 ?) — c'est le test `--split-year` de `screen_pre_fomc.py`, il EXIGE les
  bougies. **C'est le seul contrôle qui reste avant toute décision.**
- Le bootstrap stationnaire et les moments exacts (skew/kurtosis réels).
- Une éventuelle erreur dans le backtest lui-même (coûts, fill) — non ré-auditée ici.

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

---

## 🚨 Pre-FOMC — la littérature répond déjà : effet DISPARU après 2015 (recherche 2026-08-01)

Le test bloquant (`--split-year 2015`) a **déjà été fait par des chercheurs**, sur
un échantillon plus long et plus propre que le nôtre.

**Kurov, Wolfe & Gilbert (2021), « The disappearing pre-FOMC announcement drift »,
*Finance Research Letters* 40** — échantillon étendu à décembre 2019 :
> le drift pre-FOMC **a essentiellement disparu après 2015**, aussi bien pour les
> annonces avec conférence de presse que sans. Explication avancée : baisse de
> l'incertitude après la sortie du taux zéro (décembre 2015).

Découpage rapporté : drift présent sur avril 2011 → décembre 2015, **aucune trace
significative** sur janvier 2016 → décembre 2019.

### Contre-évidence (plus faible)
- Un article de 2024 (*Applied Economics*) conclut à un effet « long-lasting »,
  mais sur un cadrage différent (marchés de volatilité inclus).
- Des sources non académiques (blogs quant) affirment l'effet vivant jusqu'en 2024.
- La NY Fed (Liberty Street, 2018) trouvait des rendements résiduels **uniquement
  pour les réunions avec conférence de presse** — nuance devenue caduque : depuis
  2019 **toutes** les réunions FOMC ont une conférence de presse.

⚖️ Le poids de la preuve penche du côté « effet mort » : la source la plus solide
(revue à comité de lecture, méthodologie explicite) est celle qui conclut à la
disparition.

### 💥 Conséquence directe pour ce projet
Notre mesure (Sharpe 0,70 · t = 2,80 · p = 0,003) porte sur **2010 → 2026**, donc
**à cheval sur les deux régimes**. Si l'effet s'est éteint fin 2015, alors :
- les ~40 trades d'avant 2015 portent tout le signal ;
- les ~88 trades d'après ne rapportent rien ;
- notre t = 2,80 global **mesure un effet historique, pas un effet exploitable**.

C'est le destin classique d'une anomalie publiée — exactement ce que ce projet
passe son temps à traquer, cette fois sur son propre survivant.

**Statut du pre-FOMC : ⏸️ SUSPENDU → 🔻 probablement MORT en pratique.**
Le `--split-year 2015` reste à lancer pour confirmer sur NOS données, mais
l'attente est désormais un échec sur la seconde moitié. Ne PAS engager d'argent
sur la réunion de septembre 2026 avant ce résultat.

> ✅ **FAIT le 2026-08-06 — mais le test ne pouvait pas trancher.** Nos données
> commencent en 2012 : il n'existe aucune période pré-publication à comparer.
> Le découpage a donné l'inverse de l'attendu (−26,6 → +69,4 pips), ce qui **ne
> valide pas** l'effet — le gain est concentré sur 2020/2022/2024 et 2012-2019
> est plat à négatif. Verdict final rendu sur d'autres critères (médiane
> négative, concentration, réunion d'urgence, coûts US30 non mesurés) :
> **NO-GO US500 · non concluable US30**. Voir le bandeau en tête de fichier.

### Effet sur le pre-ECB
Le screen pre-ECB avait été dérivé du mécanisme « announcement premium ». Si ce
mécanisme a été arbitragé aux US, l'hypothèse européenne perd sa force — mais
devient une question distincte et légitime : la zone euro a-t-elle suivi le même
chemin, ou avec du retard ? Le test garde donc son intérêt, avec une attente
revue à la baisse.

Sources : [Kurov, Wolfe & Gilbert — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3134546) ·
[version PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7525326/) ·
[Lucca & Moench, NY Fed SR512](https://www.newyorkfed.org/research/staff_reports/sr512.html) ·
[Liberty Street Economics 2018](https://libertystreeteconomics.newyorkfed.org/2018/11/the-pre-fomc-announcement-drift-more-recent-evidence/)
