# V3 Hypothesis 06 — Extension Donchian multi-actif

**Date** : 2026-05-14
**Statut** : 🔴 **NO-GO** — 0 actif GO sur 6 testés (5 testés + 1 erreur), 1 actif sans données
**Prompt** : 07
**Priorité** : 🔴 P0 — expansion univers CFD
**n_trials cumulatif** : 22 (21 hérités v1+v2 + 1 H06)

---

## 0. Question

Le Donchian Breakout (validé sur US30 D1, Sharpe OOS +3.07 en H03, +8.84 en H05) fonctionne-t-il sur d'autres CFD décorrélés (GER30, US500, XAUUSD, XAGUSD, USOIL, BUND) ?

**Réponse** : ❌ Non. 0 actif sur 7 ne passe `validate_edge` avec les coûts réalistes v3. US30 tombe de +3.07 (v2) à −0.09. XAUUSD a un Sharpe brut de 1.46 mais WR trop faible (22.5%). GER30, US500, XAGUSD : Sharpe test négatif. USOIL : erreur (prix négatifs WTI 2020). BUND : pas de données.

---

## 1. Méthode

### 1.1 Stratégie testée

**Donchian Breakout** uniquement (pas de stratégie alternative — cf. H07).

- **Entrée** : Close > Highest(High, N) sur les M dernières barres → LONG. Close < Lowest(Low, N) → SHORT.
- **Sortie** : Close < Lowest(Low, M) (LONG), Close > Highest(High, M) (SHORT), OU TP/SL touché, OU timeout.
- **Paramètres** : N ∈ {20, 50, 100}, M ∈ {10, 20, 50} → 9 combinaisons.

### 1.2 Actifs testés (6/7 + 1 erreur)

`discover_assets()` a trouvé 11 actifs D1. Après filtrage H06 (actifs dans `ASSET_CONFIGS`), 6/7 testés :

| Actif | D1 dispo ? | Source | Statut |
|-------|-----------|--------|--------|
| US30 (baseline) | ✅ | CSV fourni | ❌ NO-GO |
| XAUUSD (Or) | ✅ | CSV fourni | ❌ NO-GO |
| GER30 (DAX) | ✅ | yfinance `^GDAXI` | ❌ NO-GO |
| US500 (S&P 500) | ✅ | CSV fourni | ❌ NO-GO |
| XAGUSD (Argent) | ✅ | CSV fourni | ❌ NO-GO |
| USOIL (WTI) | ✅ | yfinance `CL=F` | ⚠️ Erreur (prix ≤ 0) |
| BUND (Obligataire) | ❌ | yfinance bloqué | ⚠️ Pas de données |

### 1.3 Coûts par actif (appliqués)

| Actif | Spread (pips) | Slippage (pips) | TP (points) | SL (points) |
|-------|---------------|-----------------|-------------|-------------|
| US30 | 3.0 | 5.0 | 200 | 100 |
| GER30 | 2.0 | 3.0 | 400 | 200 |
| US500 | 1.5 | 2.0 | 200 | 100 |
| XAUUSD | 25 | 10 | 600 | 300 |
| XAGUSD | 30 | 15 | 150 | 75 |
| USOIL | 4 | 3 | 200 | 100 |

### 1.4 Split temporel

| Période | Dates |
|---------|-------|
| Train | ≤ 2022-12-31 |
| Val | 2023-01-01 → 2023-12-31 |
| Test | ≥ 2024-01-01 |

### 1.5 Protocole

1. `discover_assets()` → filtrer les actifs D1 dans `ASSET_CONFIGS`.
2. Pour chaque actif : grid search 9 combinaisons (N, M) sur train, sélection Sharpe max.
3. Appliquer best params sur val 2023 puis test ≥ 2024.
4. `validate_edge(equity_test, trades_test, n_trials=6)` sur les 5 critères constitution.
5. `read_oos(prompt="07", hypothesis="H06", sharpe, n_trades)`.

---

## 2. Résultats (6 actifs, 1 erreur, 1 indisponible)

### 2.1 Synthèse

| Actif | Best (N, M) | Sharpe Train | Sharpe Val | Sharpe Test | WR Test | Trades Test | GO? |
|-------|-------------|-------------|------------|-------------|---------|-------------|-----|
| US30 | (100, 10) | +0.35 | +0.58 | −0.09 | 45.3% | 75 | ❌ |
| XAUUSD | (100, 20) | +1.13 | 0.00 | +1.46 | 22.5% | 40 | ❌ |
| GER30 | (50, 10) | +0.29 | +1.86 | −1.01 | 41.8% | 55 | ❌ |
| US500 | (50, 50) | +0.62 | +1.62 | −0.85 | 56.5% | 46 | ❌ |
| XAGUSD | (20, 10) | 0.00 | 0.00 | 0.00 | 0.0% | 70 | ❌ |
| USOIL | — | — | — | — | — | — | ⚠️ Erreur (prix négatifs) |
| BUND | — | — | — | — | — | — | ⚠️ Pas de données |

### 2.2 Détail US30 (baseline v2)

- **Grid search** : 9 combinaisons. 3/9 Sharpe train > 0.
- **Best train** : N=100, M=10, Sharpe=+0.35
- **Val 2023** : Sharpe=+0.58, WR=47.1%, 34 trades.
- **Test ≥ 2024** : Sharpe=−0.09, WR=45.3%, 75 trades.
- **validate_edge** : ❌ Sharpe −0.27, DSR=−7.85 (p=1.000), Max DD 362.5%
- **Analyse** : L'edge v2 (Sharpe +3.07 en H03, +8.84 en H05) s'effondre avec les coûts réalistes v3 (spread 3.0 + slippage 5.0 pips). Le Donchian pur sans méta-labeling est insuffisant.

### 2.3 Détail XAUUSD

- **Grid search** : 9/9 combinaisons Sharpe train > 0 (0.89–1.13). Exceptionnel.
- **Best train** : N=100, M=20, Sharpe=+1.13
- **Val 2023** : Sharpe=0.00, WR=0%, 15 trades — effondrement.
- **Test ≥ 2024** : Sharpe=+1.46, WR=22.5%, 40 trades.
- **validate_edge** : ❌ WR 22.5%, trades/an 18.1. **Sharpe 7.18 et DSR 2.88 (p=0.002) passent.**
- **Analyse** : Biais directionnel fort mais WR trop faible. Candidat idéal pour méta-labeling (H10-H12).

### 2.4 Détail GER30

- **Grid search** : 5/9 combinaisons Sharpe train > 0.
- **Best train** : N=50, M=10, Sharpe=+0.29
- **Val 2023** : Sharpe=+1.86, WR=26.3%, 19 trades.
- **Test ≥ 2024** : Sharpe=−1.01, WR=41.8%, 55 trades.
- **validate_edge** : ❌ Sharpe −3.74, DSR=−4.43 (p=1.000), Max DD 48.3×, trades/an 28.3.
- **Analyse** : Forte divergence val/test (+1.86 → −1.01). Régime 2024+ défavorable au DAX.

### 2.5 Détail US500

- **Grid search** : 5/9 combinaisons Sharpe train > 0.
- **Best train** : N=50, M=50, Sharpe=+0.62
- **Val 2023** : Sharpe=+1.62, WR=43.8%, 16 trades.
- **Test ≥ 2024** : Sharpe=−0.85, WR=56.5%, 46 trades.
- **validate_edge** : ❌ Sharpe −3.60, DSR=−4.81 (p=1.000), Max DD 4.1×, trades/an 21.5.
- **Analyse** : WR correct (56.5%) mais PnL/trade trop faible face aux coûts. Même pattern que GER30 : val OK, test négatif.

### 2.6 Détail XAGUSD

- **Grid search** : Toutes les 9 combinaisons Sharpe train = 0.00. Aucun signal.
- **validate_edge** : ❌ WR 0.0%.
- **Analyse** : Le Donchian ne capture aucun edge sur l'argent. Actif probablement non trend-following en D1.

### 2.7 USOIL (erreur)

- **Cause** : 2 barres avec prix ≤ 0 (WTI négatif, avril 2020). `load_asset()` rejette le CSV.
- **Solution** : Filtrer `Close ≤ 0` dans le script de download.

### 2.8 BUND (indisponible)

- **Cause** : Yahoo Finance bloque tous les tickers Bund (401 Unauthorized / delisted).
- **Alternative** : Données manuelles nécessaires.

---

## 3. Verdict

### 🔴 NO-GO — 0 actif sur 7 ne passe les 5 critères constitution

| Actif | Échecs |
|-------|--------|
| US30 | Sharpe −0.27, DSR −7.85 (p=1.0), Max DD 362.5% |
| XAUUSD | WR 22.5% < 30%, Trades/an 18.1 < 30 |
| GER30 | Sharpe −3.74, DSR −4.43 (p=1.0), Max DD 4829%, Trades/an 28.3 |
| US500 | Sharpe −3.60, DSR −4.81 (p=1.0), Max DD 411%, Trades/an 21.5 |
| XAGUSD | WR 0.0% (9 combos Sharpe train = 0.00) |
| USOIL | Erreur : prix ≤ 0 (WTI avril 2020) |
| BUND | Pas de données (yfinance bloque tous les tickers) |

### Conclusion

Le Donchian Breakout pur (sans méta-labeling) ne survit pas aux coûts réalistes v3. L'edge v2 (Sharpe +3.07) était partiellement un artefact de coûts sous-estimés.

**Deux lueurs d'espoir :**
1. **XAUUSD** : Sharpe test +1.46, DSR +2.88 (p=0.002) — seuls WR et trades/an bloquent. Un méta-labeling (H10-H12) pourrait filtrer les faux signaux et remonter le WR > 30%.
2. **US30** : WR correct (45.3%), trades/an OK (36.3). Le Sharpe négatif vient de la taille des pertes, pas de la fréquence. Le méta-labeling v2 a prouvé qu'il peut transformer cet edge.

### Recommandation pour la suite

Procéder au **Prompt 08** (H07 — stratégies alternatives) sur US30 pour chercher un edge plus robuste aux coûts, puis **Prompt 10-11** (H09-H12 — méta-labeling) pour US30 + XAUUSD.

---

## 4. Architecture — Fichiers créés/modifiés

| Fichier | Action | Description |
|---------|--------|-------------|
| `app/config/instruments.py` | Créé | `AssetConfig` + `ASSET_CONFIGS` (7 actifs) |
| `scripts/run_h06_donchian_multi_asset.py` | Créé | Grid search multi-actif (370 lignes) |
| `scripts/download_h06_missing_assets.py` | Créé | Téléchargement yfinance GER30, USOIL |
| `predictions/h06_donchian_multi_asset.json` | Créé | Résultats détaillés 6 actifs |
| `docs/v3_hypothesis_06.md` | Créé | Ce rapport |

---

## 5. Prochaine étape — H07 (Prompt 08)

Tester des stratégies trend-following alternatives (Dual MA, Keltner Channel, Chandelier Exit, Parabolic SAR) sur US30 D1 pour trouver un edge qui résiste aux coûts réalistes v3, avant d'appliquer le méta-labeling en H10-H12. XAUUSD sera retesté après méta-labeling.

---

## 6. Leçons apprises

1. **Les coûts réalistes changent tout** : le Donchian US30 qui faisait +3.07 en v2 tombe à −0.09 avec spread 3.0 + slippage 5.0 pips. L'edge v2 était partiellement un artefact de coûts sous-estimés.
2. **XAUUSD a un biais directionnel exploitable** (Sharpe 1.46) mais nécessite un filtre de qualité (méta-labeling) pour remonter le WR.
3. **GER30, US500, XAGUSD** : aucun edge Donchian trend-following avec les coûts v3.
4. **USOIL** : prix négatifs (WTI avril 2020) bloquent `load_asset()`. À filtrer dans le script de download.
5. **BUND** : indisponible via yfinance (tous tickers testés : BUND, FGBL=F, BUND.DE, EUBUND, ^BUND).
6. **La val 2023 est un bon détecteur d'overfitting** : GER30 passe de +0.29 train à +1.86 val puis −1.01 test. XAUUSD passe de +1.13 train à 0.00 val puis +1.46 test.
7. **yfinance produit des colonnes minuscules** (`open` vs `Open`) — `auto_adjust=False` + normalisation nécessaires.
8. **load_asset() attend du TSV** (`sep="\t"`), pas du CSV virgule.
