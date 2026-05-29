# Step 07 — Validation cross-actif (GBPUSD / USDJPY / XAUUSD)

**Catégorie** : Validation (test final de robustesse)
**Priorité** : 🟡 Basse (gate final, ne pas faire avant que les autres steps aient produit un edge)
**Effort estimé** : 2-3 jours
**Dépendances** : steps 01-06 doivent avoir abouti à une config "production candidate"

---

## 1. Hypothèse mathématique

### Principe

Un edge **réel** sur EURUSD H1 doit, à minima, **transférer** (modestement, mais positivement) sur d'autres paires forex majeures **sans tuning par actif**. Si l'edge ne transfère pas, deux explications :

1. **Overfit instrument-spécifique** : le modèle a appris des particularités d'EURUSD non généralisables ⇒ l'edge est illusoire.
2. **Vraie spécificité d'EURUSD** : possible (cf. liquidité, micro-structure différente) mais rare. Dans le doute, considérer (1).

### Formalisation

Soit $\theta^* = \arg\max_\theta \text{Sharpe}_{\text{OOS-EURUSD}}(\theta)$ la config optimale trouvée sur EURUSD. On évalue $\theta^*$ (figé) sur 3 autres actifs $A \in \{GBPUSD, USDJPY, XAUUSD\}$ :

$$\text{Test passé} \iff \#\{A : \text{Sharpe}_A(\theta^*) > 0 \text{ et } DSR_A > 0\} \ge 2$$

**Hypothèse $H_1$** : si l'edge est réel et lié à la microstructure forex globale (RSI, ADX, sessions, calendrier macro USD/EUR), il transfère partiellement sur GBPUSD et XAUUSD (corrélés avec USD strength), moins sur USDJPY (dynamique propre BoJ).

**Hypothèse $H_0$ (à rejeter)** : le Sharpe positif sur EURUSD est l'effet du data-snooping sur 15+ itérations — sur les 3 autres actifs, Sharpe ≈ 0 ou négatif.

---

## 2. Méthodologie d'implémentation

### Fichiers concernés

- **Étendre** [`learning_machine_learning/config/instruments.py`](../learning_machine_learning/config/instruments.py) avec :

  ```
  @dataclass(frozen=True)
  class GbpUsdConfig(InstrumentConfig):
      name = "GBPUSD"
      pip_size = 0.0001
      pip_value_eur = 1.18  # taux EUR/GBP indicatif
      timeframes = frozenset({"H1", "H4", "D1"})
      primary_tf = "H1"
      macro_instruments = frozenset({"EURUSD", "USDCHF"})
      features_dropped = EurUsdConfig.features_dropped  # IDENTIQUE — pas de tuning
  
  @dataclass(frozen=True)
  class UsdJpyConfig(InstrumentConfig):
      name = "USDJPY"
      pip_size = 0.01  # JPY exception
      pip_value_eur = 0.0085  # 1 pip USDJPY ≈ 0.0085 EUR par lot
      ...
  
  @dataclass(frozen=True)
  class XauUsdConfig(InstrumentConfig):
      name = "XAUUSD"
      pip_size = 1.0  # or, 1$ = 1 unité
      pip_value_eur = 0.92
      macro_instruments = frozenset({"EURUSD", "USDCHF"})
      ...
  ```

- **Vérifier la présence des données** dans `cleaned-data/` : il faut au moins `{instrument}_H1_cleaned.csv`, `{instrument}_H4_cleaned.csv`, `{instrument}_D1_cleaned.csv` pour chaque actif testé. Si absent, étape préalable d'ingestion via `data/loader.py`.

- **Créer** [`learning_machine_learning/pipelines/gbpusd.py`](../learning_machine_learning/pipelines/), `usdjpy.py`, `xauusd.py` — chacun ~10 lignes hérité de `BasePipeline`, identique à [`pipelines/eurusd.py`](../learning_machine_learning/pipelines/eurusd.py) sauf le nom d'instrument.

- **Mettre à jour** [`config/registry.py`](../learning_machine_learning/config/) pour enregistrer les 3 nouveaux instruments.

- **Créer** `run_pipeline_multi_asset.py` à la racine :
  - Itère sur les 4 instruments (EURUSD, GBPUSD, USDJPY, XAUUSD)
  - Lance le pipeline complet (avec config v16+ courante) pour chacun
  - Aggrège les métriques dans un tableau side-by-side
  - Génère `predictions/multi_asset_comparison.md`

- **Créer** [`learning_machine_learning/analysis/cross_asset.py`](../learning_machine_learning/analysis/) :
  - `compute_pnl_correlation(trades_by_asset: dict[str, pd.DataFrame]) -> pd.DataFrame` : matrice de corrélation des PnL daily entre actifs
  - `aggregate_cross_asset_metrics(metrics_by_asset: dict) -> dict` : E[Sharpe], min/max, # actifs avec Sharpe > 0

### Choix techniques

- **Spread historique par actif** : récupérer les spreads moyens réels par paire (GBPUSD ~0.8 pips, USDJPY ~0.9 pips, XAUUSD ~25 pips) et calibrer `commission_pips`, `slippage_pips` dans `BacktestConfig` par instrument (via une dataclass héritante ou un override). NOTE : c'est techniquement un tuning par actif, mais **inévitable** sinon le backtest est faux.
- **TP/SL en pips** : conserver les ratios (3:1 actuel) mais ajuster les valeurs absolues à l'ATR moyen de chaque actif. Pour XAUUSD, TP=30 pips = 30$ — absurde. Pour cohérence, exprimer TP/SL en multiples d'ATR.
- **Identique partout (pas d'overfit)** :
  - `features_dropped` strictement identique à `EurUsdConfig`
  - `RF_PARAMS` ou backend GBM identique
  - Seuils méta calibrés calculés via le mode `breakeven` (formule analytique, pas sweep par actif)

### Anti-leak / précautions
- **Pas de "cherrypicking" des actifs gagnants** : commencer par fixer la liste {EUR, GBP, USDJPY, XAU} ex ante. Pas de "on essaie 10 actifs et on garde les 3 qui marchent".
- **Pas de recalibration features** : la liste `features_dropped` a été optimisée sur EURUSD. La garder figée pour les autres actifs (sinon, on fait du tuning par actif déguisé).
- **Données indépendantes** : les CSV historiques de GBPUSD/USDJPY ne doivent pas être manipulés post-hoc. Sourcer depuis le même provider que EURUSD pour cohérence.

---

## 3. Métriques de validation

### Métriques cibles
| Métrique | Référence EURUSD v16 | Objectif step_07 |
|---|---|---|
| Sharpe OOS 2025 GBPUSD | NA | **> 0** |
| Sharpe OOS 2025 USDJPY | NA | **> 0** (ou Sharpe > -0.3 = acceptable) |
| Sharpe OOS 2025 XAUUSD | NA | **> 0** |
| Nombre d'actifs avec Sharpe > 0 | NA | **≥ 2 sur 4 (incluant EURUSD)** |
| DSR multi-actif (agrégé) | NA | **> 0** |

### Métriques secondaires

- **Corrélation des PnL daily entre actifs** : devrait être **faible à modérée** (0.2-0.5). Si > 0.8 → tous les actifs gagnent/perdent ensemble = pas de diversification, et probablement un biais commun (ex : USD strength). Si < 0 → divergence pathologique.
- **Sharpe pondéré par capital** : moyenne pondérée par taille de position implicite — utile pour un portfolio futur.
- **Drawdown max agrégé** : si on tradait les 4 actifs en parallèle avec position égale, quel serait le DD max ? (Test de portfolio basique.)

### Critère d'arrêt (= go/no-go production)

- **Sharpe > 0 sur ≥ 2 actifs / 4** ⇒ **edge probablement réel** ⇒ GO production avec le portfolio multi-actifs.
- **Sharpe > 0 uniquement sur EURUSD** ⇒ edge spécifique à l'instrument. Acceptable mais risqué — exiger ≥ 1 an de live paper trading avant capital réel.
- **Sharpe < 0 sur 3 actifs / 4** ⇒ **preuve d'overfit EURUSD** ⇒ retour au labo, reconsidérer cible (step_01) ou abandonner.

---

## 4. Risques & dépendances

- **R1 — Données non disponibles** : si `cleaned-data/{GBPUSD,USDJPY,XAUUSD}_*.csv` n'existent pas, prévoir 1 jour additionnel pour ingestion (source : MT5 export, Dukascopy, ou TradingView download).
- **R2 — Spécificités USDJPY** : pip_size 0.01 (vs 0.0001 pour les autres). Vérifier que tous les calculs (pips_to_return, TP/SL pip-relative) utilisent correctement `instrument.pip_size` et ne hard-codent pas 0.0001. Auditer [`backtest/simulator.py`](../learning_machine_learning/backtest/simulator.py), [`backtest/metrics.py`](../learning_machine_learning/backtest/metrics.py).
- **R3 — XAUUSD comme test extrême** : l'or n'est pas une paire forex au sens strict (instrument synthétique vs USD). Volatilité différente, fondamentaux différents (taux US, risk-off). Si Sharpe XAU > 0, fort signal. Si négatif uniquement sur XAU, OK — toujours considérer 2 forex sur 3 comme succès.
- **R4 — Multi-actif n'est pas un portfolio** : ce test évalue la **généralisation** de la stratégie, pas la performance d'un portfolio diversifié. Pour un vrai portfolio, il faudrait ajouter une couche d'allocation (vol-targeting, risk-parity) — hors scope de step_07.
- **R5 — Coût compute** : 4 × pipeline complet (~30s chacun) × CPCV (200 splits) = 16 000 minutes = **infaisable sans parallélisation**. Mitigation : pour cette étape, accepter un seul split train ≤ 2023 / val 2024 / test 2025 (au lieu de CPCV intégral) si CPCV trop lent.

---

## 5. Références

- Harvey, C. R., Liu, Y., & Zhu, H. (2016). *... and the Cross-Section of Expected Returns*. Review of Financial Studies. (sur l'inflation des découvertes par data-snooping)
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). *Pseudo-Mathematics and Financial Charlatanism*. Notices of the AMS.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 11 (The Dangers of Backtesting).
- Code à étendre : [`learning_machine_learning/config/instruments.py`](../learning_machine_learning/config/instruments.py), [`pipelines/eurusd.py`](../learning_machine_learning/pipelines/eurusd.py) (template).
