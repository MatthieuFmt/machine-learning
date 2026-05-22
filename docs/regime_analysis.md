# Analyse de la distribution des régimes (Phase F4)

**Date** : 2026-05-20
**Module** : [app/features/regime.py](../app/features/regime.py) — `detect_regime()`
**Script** : [scripts/analyze_regime_distribution.py](../scripts/analyze_regime_distribution.py)
**Données** : `data/raw/<ASSET>/*_D1.csv` (Dukascopy, ingestion Phase F3)

---

## 1. Méthodologie

Pour chaque barre on classe le régime selon trois règles déterministes,
appliquées par priorité décroissante :

| Priorité | Label | Condition |
|---|---|---|
| 1 | `vol_high` | `ATR%(14)` > quantile 80% glissant sur 60 barres |
| 2 | `trend`    | `ADX(14)` > 25 (et pas déjà classé vol_high) |
| 3 | `range`    | sinon |

Les barres en warmup (NaN dans ADX, ATR% ou quantile rolling) sont
exclues du décompte. La fenêtre quantile de 60 barres D1 ≈ 3 mois
calendaires, ce qui permet une comparaison locale et non globale du
niveau de volatilité.

Par construction, **`vol_high` capture mécaniquement ~20% des barres**
(top quantile 80%). Les écarts observés (19.6% → 25.4%) viennent du
warmup et des barres où `ATR%` est constant (égalité non stricte avec
le quantile).

---

## 2. Résultats — 14 actifs en D1

| Actif | TF | Barres | Trend % | Range % | Vol_high % | Période |
|---|---|---:|---:|---:|---:|---|
| AUDJPY | D1 | 5251 | 23.7 | 54.9 | 21.4 | 2010-01-01 → 2026-05-18 |
| BTCUSD | D1 | 3042 | **37.1** | 40.5 | 22.4 | 2017-05-23 → 2026-05-13 |
| ETHUSD | D1 | 2868 | **42.4** | 35.4 | 22.2 | 2017-12-11 → 2026-05-13 |
| EURJPY | D1 | 5252 | 27.0 | 51.4 | 21.6 | 2010-01-01 → 2026-05-18 |
| EURUSD | D1 | 5251 | 29.7 | 49.4 | 20.8 | 2010-01-01 → 2026-05-18 |
| GBPJPY | D1 | 5251 | 27.0 | 51.2 | 21.8 | 2010-01-01 → 2026-05-18 |
| GBPUSD | D1 | 5252 | 25.8 | 52.5 | 21.7 | 2010-01-01 → 2026-05-18 |
| GER30 | D1 | 6541 | 20.0 | **57.0** | 23.0 | 2000-01-03 → 2025-12-30 |
| US30 | D1 | 4244 | 22.2 | 53.4 | 24.4 | 2012-04-04 → 2026-05-18 |
| US500 | D1 | 3288 | 20.7 | 53.9 | 25.4 | 2013-05-23 → 2026-05-14 |
| USDCHF | D1 | 4956 | 26.2 | 54.2 | 19.6 | 2010-04-26 → 2026-05-08 |
| USDJPY | D1 | 5016 | 23.4 | 53.4 | 23.2 | 2010-10-01 → 2026-05-18 |
| XAGUSD | D1 | 4680 | 31.4 | 48.6 | 20.0 | 2011-02-17 → 2026-05-14 |
| XAUUSD | D1 | 5153 | 28.5 | 50.1 | 21.4 | 2009-08-06 → 2026-05-08 |

USOIL est sauté (validation `load_asset` échoue — un seul TF disponible
sur cet actif après l'ingestion Phase F3).

---

## 3. Lecture par classe d'actifs

### Cryptos (BTCUSD, ETHUSD)
Profil **fortement tendanciel** : 37-42% de barres en `trend`, contre 20-30%
pour les autres classes. Cohérent avec l'historique de ces actifs marqué
par de longues phases directionnelles (bull runs 2017, 2020-21, 2024-25).
Le `range` est sous-représenté (35-40%) — les phases de consolidation
existent mais sont plus courtes.

**Implication stratégies** : les stratégies de suivi de tendance (Donchian,
EMA cross) ont *a priori* plus de matière sur BTC/ETH que sur le forex.

### Indices actions (GER30, US30, US500)
Profil **dominé par le range** : 53-57% de barres classées `range`, et
seulement 20-22% en `trend`. Le DAX ressort en tête (range 57%), suivi
par le S&P 500 et le Dow Jones. La part `vol_high` y est aussi la plus
élevée du panel (23-25%), traduisant des phases ponctuelles de stress
(corrections, crises) qui sortent du régime nominal.

**Implication stratégies** : les stratégies mean-reverting (BB, RSI
oversold/overbought) sont plus naturelles sur indices que les
stratégies trend-following.

### Forex majors (EURUSD, GBPUSD, USDCHF, USDJPY)
Profil **range modéré** (49-54%), trend moyen (23-30%). EURUSD est le
plus tendanciel des majors (29.7%), USDJPY le moins (23.4%). Distribution
globalement homogène — cohérent avec la nature mean-reverting reconnue
du forex sur D1.

**Implication stratégies** : le forex demande probablement un filtre de
régime *pour éviter* le trend-following en phase `range` (66% du temps
sur GBPUSD si on additionne range + vol_high).

### Forex JPY crosses (EURJPY, GBPJPY, AUDJPY)
Très proches des majors (range 51-55%, trend 24-27%). AUDJPY est le plus
"range-y" (54.9%), ce qui peut refléter sa nature de proxy carry-trade
oscillant entre risk-on et risk-off.

### Métaux (XAUUSD, XAGUSD)
**Plus tendanciels** que le forex (28-31% trend) — XAG ressort en tête
(31.4%, niveau proche des cryptos). Le `range` reste majoritaire (48-50%).
L'argent (XAG) est traditionnellement plus volatil et plus directionnel
que l'or — la mesure le confirme.

---

## 4. Observations transverses

1. **Le `range` est le régime dominant partout sauf en crypto**. Sur 14
   actifs, 12 ont >48% de leurs barres en `range`. Cela suggère qu'un
   filtre de régime appliqué naïvement (n'autoriser que `trend`) divisera
   le nombre de signaux par 3 à 5 sur la plupart des actifs.

2. **Le `vol_high` est mécaniquement borné autour de 20-25%** par la
   définition (top quantile 80%). C'est donc un *re-classement* d'environ
   1/5 des barres — utile pour identifier les périodes de stress, mais
   inutilisable seul comme indicateur de "marché actif".

3. **Pas de différence majeure forex / JPY crosses sur D1**. Pour
   distinguer les régimes JPY, il faudra probablement une feature
   spécifique (corrélation Nikkei, calendrier BoJ, swap rates).

4. **Période d'analyse asymétrique** — GER30 démarre en 2000, BTC en 2017.
   Pour les comparaisons inter-actifs, il faudrait normaliser la période
   ou découper par sous-période (pré/post-2020 par exemple).

---

## 5. Prochaines étapes (au-delà du MVP F4)

- [ ] **Pas de dispatch stratégie pour l'instant** — `detect_regime` est
  livré comme feature, pas comme filtre. La décision d'activer / désactiver
  certaines stratégies par régime se prendra après Phase G (screening).
- [ ] Analyser la distribution sur H4 et H1 (le script accepte
  `--tf H4`/`--tf H1`) pour voir si la résolution change la photo.
- [ ] Sur les actifs avec ≥10 ans d'historique, découper par sous-période
  (2010-2015, 2016-2020, 2021-2025) pour mesurer la stabilité temporelle
  des proportions.
- [ ] Étudier la persistance des régimes : durée moyenne d'un état avant
  bascule, probabilités de transition (matrice 3×3).

---

## 6. Données brutes

CSV : [data/analysis/regime_distribution.csv](../data/analysis/regime_distribution.csv).
Reproduction : `python scripts/analyze_regime_distribution.py --tf D1`.
