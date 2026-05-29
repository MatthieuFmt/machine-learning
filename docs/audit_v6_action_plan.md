# Audit v6 — Plan d'action hiérarchisé (priorité × effort × impact)

**Cadrage** : pas de production, pas de timing. Mais budget de temps réaliste. L'objectif est de **maximiser l'information apprise par jour de travail**, en commençant par les corrections structurelles puis les paris à plus haute probabilité.

---

## 1. Vision globale — 3 phases séquentielles

```
Phase F — Foundation             Phase G — Universe expansion       Phase H — Strategy expansion
~1 semaine                       ~2 semaines                        ~3 semaines
─────────────────────             ─────────────────────              ─────────────────────
F1. Swap modélisé                G1. Download Dukascopy 30 actifs   H1. Pre-FOMC drift
F2. Coûts XTB démo               G2. Test pairs JPY                 H2. Asian Range / ORB
F3. Re-download Dukascopy H1     G3. Test Nasdaq + Nikkei           H3. NR4/NR7
F4. Régime detection MVP         G4. Test Brent + NatGas + Copper   H4. Pairs trading
F5. Features macro DXY/VIX       G5. Test cryptos additionnelles    H5. Cross-sectional momentum
```

**Critère go/no-go entre phases** :
- Phase F → G : pipeline corrigé, données fiables.
- Phase G → H : ≥ 3 actifs nouveaux montrent Sharpe train ≥ 0.5 sur stratégies existantes (sinon pivot stratégies avant data).
- Phase H : 1 famille montre Sharpe ≥ 0.7 sur train.

À la fin : décision GO/NO-GO du projet (Sharpe ≥ 1.0 OOS validé statistiquement, ou conclusion honnête "edge inaccessible avec ces moyens").

---

## 2. Phase F — Foundation (~5-7 jours)

### F1. Modéliser le swap overnight 🔴 BLOQUEUR

**Action** :
1. Ajouter à `AssetConfig` (dans [`app/config/instruments.py`](../app/config/instruments.py)) :
   ```python
   swap_long_pips_per_night: float = 0.0
   swap_short_pips_per_night: float = 0.0
   ```
2. Modifier `_simulate_stateful_core()` ([`app/backtest/simulator.py`](../app/backtest/simulator.py)) pour calculer `nights_held` à partir de `entry_time` et `exit_time` :
   ```python
   nights_held = (exit_time.normalize() - entry_time.normalize()).days
   swap_cost = nights_held * (swap_long_pips if direction == 1 else swap_short_pips)
   pips_net -= swap_cost
   ```
3. Idem pour `app/backtest/deterministic.py`.
4. Test unitaire `tests/unit/test_swap_overnight.py` :
   - Trade 1 jour : swap_cost = 1 × swap_pips.
   - Trade overnight Wed→Thu : 1 nuit.
   - Trade Fri→Mon : 3 nuits (charge triple le mercredi pour forex en pratique — à modéliser ou ignorer en V1).
5. Re-runner tous les Sharpe historiques connus (Donchian D1, ETHUSD H1) → mesurer décalage.

**Effort** : 1-2 j.
**Impact** : 🔴 critique méthodologie. Sans ça, tous les futurs Sharpe sont biaisés.
**Probabilité gain edge** : 0 % (c'est un dé-bias, pas un edge), mais nécessaire pour décisions correctes.

**Livrables** :
- Code modifié `app/backtest/{simulator,deterministic}.py` + `app/config/instruments.py`.
- `tests/unit/test_swap_overnight.py`.
- `docs/swap_impact_analysis.md` : tableau impact par actif/stratégie.

### F2. Valider coûts XTB en démo (F10) 🔴 BLOQUEUR

**Action** :
1. Ouvrir compte démo XTB MT5 (gratuit, 5 min).
2. Pour chaque actif `BTCUSD, ETHUSD, GBPUSD, USDCHF, EURUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, US100, JAP225, UK100, FRA40, AUS200, UKOIL, NATGAS, COPPER, COFFEE, PALLADIUM` :
   - Récupérer "Symbol Specifications" : `Spread`, `Swap Long`, `Swap Short`, `Commission`, `Contract Size`, `Min Lot`, `Tick Size`.
   - Noter le spread typique observé (heures actives, hors news).
3. Mettre à jour `ASSET_CONFIGS` dans `app/config/instruments.py`.
4. Supprimer les commentaires `PROVISOIRE` du code.

**Effort** : 1 j.
**Livrable** : `docs/xtb_specs_demo_2026-05.csv` + diff dans `instruments.py`.

### F3. Re-download data via Dukascopy (H1 propre + actifs nouveaux)

**Action** :
1. Installer `dukascopy-python` (`pip install dukascopy-python`).
2. Créer `scripts/download_dukascopy_full.py` :
   ```python
   instruments = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "AUDUSD", "NZDUSD",
                  "USDCAD", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURCHF",
                  "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD",
                  # indices et commodities selon liste F2
                  ]
   timeframes = ["H1", "H4", "D1", "W1"]  # + M15 plus tard si ORB
   from_date = "2010-01-01"  # 16 ans
   ```
3. Sauver dans `data/raw/<asset>/<asset>_<tf>.csv` au format actuel (Tab-separated, colonnes Time/OHLCV).
4. Valider via `app/data/loader.py` (gaps, OHLC consistency).

**Effort** : 1 j (script trivial + 4-8h téléchargement passif).
**Livrable** : data complète pour 30 actifs × 4 TF.

### F4. Régime detection MVP (règles déterministes)

**Action** :
1. Créer `app/regime/detector.py` :
   ```python
   def detect_regime(df: pd.DataFrame) -> pd.Series:
       """Retourne label {trend, range, vol_high} par barre."""
       adx = compute_adx(df, 14)
       atr_pct = compute_atr_pct(df, 14)
       atr_pct_60_pct = atr_pct.rolling(60).quantile(0.8)

       trend = adx > 25
       vol_high = atr_pct > atr_pct_60_pct
       return pd.Series(np.where(vol_high, "vol_high",
                        np.where(trend, "trend", "range")))
   ```
2. Tester sur train EURUSD D1 : visualiser % temps par régime.
3. **Pas encore de dispatch stratégie** — juste la feature.

**Effort** : 1-2 j.
**Livrable** : `app/regime/detector.py`, `tests/unit/test_regime_detector.py`, `docs/regime_analysis.md` (distribution régime par actif).

### F5. Features macro externes (DXY, VIX, yield curve)

**Action** :
1. Créer `app/features/macro_external.py` :
   ```python
   @look_ahead_safe
   def add_external_macro(df: pd.DataFrame, asset: str) -> pd.DataFrame:
       dxy = load_external("DXY", df.index)  # via FRED ou yfinance
       vix = load_external("VIX", df.index)
       slope_10y_2y = load_external("T10Y2Y", df.index)
       df["dxy_zscore_60"] = (dxy - dxy.rolling(60).mean()) / dxy.rolling(60).std()
       df["vix_level"] = vix
       df["vix_zscore_60"] = ...
       df["yield_slope"] = slope_10y_2y
       return df
   ```
2. Source : FRED API (clé gratuite via [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)).
3. Test anti-look-ahead.

**Effort** : 2 j (incluant gestion API + cache local).
**Livrable** : module + 3 features supplémentaires dans superset.

### F6. Bilan Phase F

À la fin :
- Pipeline corrigé (swap + coûts validés).
- Data fiable 30 actifs × 4 TF.
- Régime detector disponible (feature, pas encore filtre).
- Features macro disponibles (DXY, VIX, yield).

Re-runner les 2 stratégies historiquement les moins mauvaises (ETHUSD H1 hgbm, GBPUSD H4 rf) **avec swap correct + coûts validés** → comparer Sharpe pré-fix vs post-fix.

**Critère go/no-go Phase G** :
- ✅ GO si pipeline corrigé sans régression (tous tests passent).
- ❌ Pivot si l'ajout du swap fait chuter tous les Sharpe historiques sous 0 → réfléchir aux stratégies à durée courte (intraday, no overnight).

---

## 3. Phase G — Universe expansion (~10-14 jours)

### G1. Screening étendu D1 + H4 + H1 sur 30 actifs

**Action** :
1. Étendre `scripts/screen_strategies_train.py` aux 30 actifs F3, sur D1 / H4 / H1.
2. Tester les **13 stratégies existantes** + 5 nouveaux ratios TP/SL : `{1:1, 2:1, 3:1, 5:1}` avec SL ∈ `{0.5, 1.0, 1.5} × ATR`.
3. **n_trials cumul** : ce n'est pas une lecture OOS, donc 0 n_trial. C'est du screening train.

**Combinaisons** : 30 actifs × 3 TF × 13 strats × 4 ratios × 3 SL = **14040 backtests train**.

**Effort** : 3-5 j (re-utilise screen existant, adapté).
**Livrable** : `predictions/screen_v6_train.json`.

**Critère** : retenir les combinaisons avec **Sharpe train ≥ 0.6, WR ≥ 35 %, n_trades ≥ 60**. Estimation : ~50-200 candidats.

### G2. Tests OOS sélectifs sur top candidats

Parmi les candidats G1, retenir le **top 10 par delta_sharpe stability** (Sharpe positif sur 2+ sous-échantillons train).

Tester en OOS unique :
- Test ≥ 2024 (24 mois).
- Critères GO : Sharpe ≥ 0.8, delta_sharpe vs train > -0.5, WR ≥ 33 %, n_trades ≥ 30, max_dd ≤ 25 %.

**Effort** : 2-3 j (10 lectures OOS = +10 n_trials cumul → DSR ajusté).
**Livrable** : `predictions/oos_v6_phase_g.json` + analyse markdown.

### G3. Cas particulier : JPY pairs

Tester intentionnellement sur les pairs JPY (USDJPY, EURJPY, AUDJPY, GBPJPY) avec :
- TsMomentum_60 (capture tendance carry).
- Donchian_20/20 (breakout sur tendance forte).
- DualMA 10/50 (trend-following classique).
- **Ratio 3:1 ou 5:1** (TP grand car la tendance court longtemps).

Hypothèse forte : ces pairs ont structurellement un edge trend-following plus marqué que EURUSD/GBPUSD.

**Effort** : 1 j supplémentaire (configs + run).
**Livrable** : `docs/jpy_pairs_analysis.md`.

### G4. Bilan Phase G

À la fin :
- 30 actifs testés sur 13 stratégies × multiples TP/SL.
- Top candidats OOS testés.
- Verdict : combien d'edges Sharpe ≥ 0.8 OOS ?

**Critère go/no-go Phase H** :
- ✅ ≥ 1 sleeve passe (proceed to ML méta-labeling + portfolio).
- ❌ 0 sleeve passe (les stratégies classiques sont mortes même sur univers étendu) → Phase H = stratégies différentes (calendar, ORB, pairs).

**Si succès G** : on a un edge dans le portfolio, le projet est validé scientifiquement. Phase H reste utile pour diversification.

---

## 4. Phase H — Strategy expansion (~15-20 jours)

Toutes les pistes peuvent être faites en parallèle (par familles).

### H1. Pre-FOMC drift (calendar) — priorité 1, effort 1-2 j

**Action** :
1. Vérifier que `app/features/economic.py` charge bien les dates FOMC depuis `data/raw/economic_calendar/`.
2. Stratégie simple : long US500 (et/ou US100) entre `FOMC - 24h` et `FOMC - 1h`. Close à `FOMC` exactement.
3. Backtest train ≤ 2022 (96 FOMC sur 12 ans). Test OOS 2024+ (16 FOMC).
4. Sharpe attendu (selon Lucca-Moench) : > 1.5 sur train, > 1.0 OOS si l'effet persiste.

**Critère GO** : Sharpe OOS ≥ 0.7 (effet réel persistant), p-value bootstrap < 0.10.

### H2. Asian Range Breakout (forex H1) — priorité 1, effort 3-4 j

**Action** :
1. Implémenter `app/strategies/asian_range.py` :
   ```python
   # Tokyo range = [00:00 UTC, 08:00 UTC]
   # Signal à 08:00 UTC :
   #   long si Close(08:00) > High(Tokyo)
   #   short si Close(08:00) < Low(Tokyo)
   # TP = 1.5 × range_Tokyo, SL = 0.5 × range_Tokyo
   # Time-stop à 22:00 UTC (close Londres + NY)
   ```
2. Test sur EURUSD/GBPUSD/USDJPY H1.

**Effort** : 3 j.
**Critère GO** : Sharpe OOS ≥ 0.7 sur ≥ 1 pair.

### H3. NR4/NR7 (Crabel volatility breakout) — priorité 2, effort 1-2 j

**Action** :
1. Implémenter `app/strategies/volatility_breakout.py` :
   ```python
   nr4_today = today_range == min(last 4 days range)
   # Signal lendemain : breakout high/low de NR4.
   # TP = 2× range_NR4, SL = 1× range_NR4
   ```
2. Test sur US30/US500/US100 D1 + H4.

**Effort** : 2 j.

### H4. Pairs trading EURUSD-GBPUSD (cointégration) — priorité 2, effort 5-7 j

**Action** :
1. Implémenter `app/strategies/pairs_trading.py` :
   - Test cointégration Engle-Granger sur train (`statsmodels.tsa.stattools.coint`).
   - Estimer β via OLS.
   - Z-score spread rolling 60 jours.
   - Signal : |z| > 2 entry, |z| < 0.5 exit.
2. Test sur paires candidates :
   - EURUSD-GBPUSD (forex).
   - XAUUSD-XAGUSD (gold-silver).
   - US500-US100 (ES-NQ).
   - UKOIL-USOIL (Brent-WTI) si data dispo.

**Effort** : 5 j.
**Critère GO** : Sharpe OOS ≥ 0.7 sur ≥ 1 pair.

### H5. Cross-sectional momentum (style AQR) — priorité 3, effort 5-7 j

**Action** :
1. Implémenter `app/strategies/cs_momentum.py` :
   - Univers : 22 actifs (forex + indices + commodities + crypto).
   - Chaque mois : compute return 12M skip-1M.
   - Long top quintile (4-5 actifs), short bottom quintile.
   - Rebalance mensuel.
2. Refactor du backtest pour supporter multi-asset simultané (le simulateur stateful actuel suppose un actif).

**Effort** : 7-10 j (changement architectural).
**Critère GO** : Sharpe OOS ≥ 0.7.

### H6. Bilan Phase H

À la fin :
- 4-5 familles testées en OOS.
- Métrique par famille : Sharpe OOS, n_trades/an, DSR, corrélation avec autres familles.

**Critère décision finale** :
- Au moins 1 famille avec Sharpe OOS ≥ 1.0 et DSR > 0 (p < 0.05) → **projet validé**.
- 2-3 familles décorrelées avec Sharpe OOS ≥ 0.7 → **portfolio Sharpe ≥ 1.0 atteignable**.
- 0 famille passe → **conclusion honnête** : pas d'edge détectable avec les méthodes courantes sur ce périmètre.

---

## 5. Calendrier prévisionnel (estimation)

| Semaine | Phase | Livrables |
|---|---|---|
| 1 | F1, F2 | Swap modélisé, coûts XTB validés |
| 2 | F3, F4, F5 | Data complète, régime detector, features macro |
| 3-4 | G1, G2, G3 | Screening 30 actifs, top OOS, JPY pairs |
| 5 | G4 | Bilan Phase G, décision orientation |
| 6 | H1, H2 | Pre-FOMC drift, Asian Range OOS |
| 7 | H3, H4 (start) | NR4/NR7, pairs trading kickoff |
| 8 | H4 (suite) | Pairs trading OOS |
| 9-10 | H5 | Cross-sectional momentum (si pertinent) |
| 11 | H6 | Bilan final + décision projet |

**Effort total** : ~10-11 semaines à temps partiel (1-2h/jour), ou ~5-6 semaines à temps plein.

---

## 6. Budget n_trials cumul

État au début de Phase F : **n_trials_cumul = 28+** (selon JOURNAL.md C5).

| Phase | n_trials ajoutés |
|---|---|
| F | 0 (corrections + screening pur train) |
| G2 | +10 (top 10 OOS) |
| G3 | +4 (JPY pairs OOS) |
| H1 | +2 (pre-FOMC sur 1-2 indices) |
| H2 | +3 (Asian Range sur 3 pairs) |
| H3 | +2 (NR4/NR7 sur 2 indices) |
| H4 | +4 (4 pairs cointégrées) |
| H5 | +1 (CS momentum) |

**Total final estimé** : 28 + 26 = **54 n_trials** cumul.

DSR avec N=54, Sharpe observed 1.5 → threshold ≈ 2.1 (assez restrictif). Avec Sharpe observed 1.2 → threshold ≈ 1.8 (marginal). C'est pour ça qu'il **vaut mieux trouver 2-3 sleeves Sharpe 1.0 décorrélées** qu'un seul à Sharpe 1.5.

---

## 7. Gestion des risques / pièges connus

### 7.1 Pièges méthodologiques à éviter

| Piège | Prévention |
|---|---|
| Re-tuning sur OOS si décevant | Règle 9 constitution : **1 lecture OOS / hypothèse**. Snooping_guard activé. |
| Cherry-picking dans 30 actifs G2 | Documenter **tous** les tests avant lecture OOS (déclaration ex-ante). |
| Survival bias données | Dukascopy = vrai historique non révisé. OK. |
| Look-ahead via features macro | DXY/VIX en daily disponibles uniquement après close. Shift(1) obligatoire. F8 validation. |
| Sur-fit aux coûts | Sensitivity analysis spread × 1.5 obligatoire avant chaque GO. |

### 7.2 Pièges techniques

| Piège | Prévention |
|---|---|
| Swap charges weekend (3× nuit le mercredi) | Implémenter dans F1 (au moins en V2). |
| Data Dukascopy gap weekend | Pipeline `is_normal_gap()` gère déjà. |
| Cointégration spurious | Test ADF sur résidus + walk-forward test de cointégration. |
| Régime change in-sample → OOS | Stress test 2020-Q1 COVID systématique. |

### 7.3 Quand abandonner

Abandonner le projet si **après Phase F + G + H1+H2+H4** (12 semaines) :
- 0 sleeve passe Sharpe ≥ 0.7 OOS.
- Tous les pairs trading montrent rupture de cointégration en OOS.
- Toutes les calendar effects montrent décroissance d'efficacité (post-discovery decay).

C'est un résultat valide : **"L'edge retail accessible via XTB avec capital 10k €, sans HFT, sans options, sans data alt payante, n'est pas détectable statistiquement avec les méthodes courantes."**

C'est rare mais possible. Cela impose alors un pivot fondamental (capital plus important, ou autre modèle économique).

---

## 8. Outils et ressources à acquérir

| Outil | Coût | Effort install |
|---|---|---|
| Compte démo XTB MT5 | gratuit | 30 min |
| `dukascopy-python` | gratuit | 5 min |
| Clé FRED API | gratuit | 5 min (`fred.stlouisfed.org`) |
| LightGBM | gratuit | 5 min (`pip install lightgbm`) |
| `statsmodels` (cointégration) | déjà installé | — |
| `arch` (GARCH si besoin) | gratuit | 5 min |
| Calendrier économique propre | déjà dans `data/raw/economic_calendar/` | — |

Pas de ressource payante nécessaire.

---

## 9. Premier pas concret (à exécuter dès demain)

**Tâche atomique** : démarrer la **Phase F1 (swap modélisé)**.

```bash
# 1. Créer la branche
rtk git checkout -b audit-v6-phase-f

# 2. Ouvrir et modifier
# - app/config/instruments.py (ajouter swap fields)
# - app/backtest/simulator.py (intégrer swap_cost dans _simulate_stateful_core)
# - app/backtest/deterministic.py (idem)
# - tests/unit/test_swap_overnight.py (nouveau)

# 3. Valider
rtk pytest tests/unit/test_swap_overnight.py -v
rtk pytest tests/unit/test_simulator.py -v  # non-régression
rtk pytest tests/unit/test_deterministic_sl_prime.py -v  # non-régression

# 4. Re-runner les screens existants avec swap modélisé
rtk python scripts/diagnose_donchian_atr_grid.py
# Comparer la sortie au diagnostic original
```

Cette tâche prend 1-2 jours, ne consume aucun n_trial OOS, et **élimine un biais structurel** dans toutes les analyses passées et futures.

---

## 10. Conclusion

Le projet a fait un excellent travail méthodologique (49 tests neufs, 15/19 findings résolus) mais a exploré un **espace de recherche très restreint** : 9 actifs × 13 stratégies × 1 ratio TP/SL × 2 TF, soit ~900 combinaisons sur **~14000 raisonnables** dans l'univers retail XTB.

Le plan ci-dessus structure une **exploration de × 15** sur 10-11 semaines à temps partiel, avec :
- Corrections bloquantes (Phase F).
- Expansion univers (Phase G) — pari sur la diversité des actifs.
- Expansion stratégies (Phase H) — pari sur des familles différentes.

À chaque phase, un critère go/no-go chiffré décide de la suite. À la fin, **soit on a trouvé un edge** (1 sleeve Sharpe ≥ 1 OOS ou un portfolio Sharpe ≥ 1.5), **soit la conclusion "pas d'edge accessible" est solidement fondée**.

Le pipeline est prêt. La question n'est plus "le pipeline est-il correct ?" mais "**a-t-on cherché là où ça pouvait se trouver ?**" — et la réponse actuelle est non.
