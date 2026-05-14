# Step 04 — Features de session (Tokyo / Londres / NY / overlaps)

**Catégorie** : Feature engineering
**Priorité** : 🟠 Moyenne
**Effort estimé** : 1-2 jours
**Dépendances** : step_03 (si GBM avec gestion native des catégorielles, meilleure intégration)

---

## 1. Hypothèse mathématique

### Observations EURUSD H1

La microstructure de l'EURUSD varie systématiquement selon les sessions de trading :

| Session (GMT) | Heures | Volatilité réalisée moyenne | Pattern |
|---|---|---|---|
| Asie (Tokyo) | 23:00-08:00 | Faible | Range, spreads larges |
| Europe (Londres) | 07:00-16:00 | Élevée | Trending, breakouts |
| US (NY) | 12:00-21:00 | Très élevée | Volatil, retournements |
| **Overlap Londres-NY** | **12:00-16:00** | **Pic de liquidité** | **Mouvements directionnels les plus fiables** |
| Faible liquidité | 22:00-01:00 | Min | Spreads max, slippage |

Les features actuelles utilisent `Hour_Sin/Hour_Cos` (cyclique) — droppées dans [`EurUsdConfig.features_dropped`](../learning_machine_learning/config/instruments.py) car permutation importance non significative. Le `SessionFilter` ([backtest/filters.py:95](../learning_machine_learning/backtest/filters.py)) est binaire (exclut 22h-01h) — perd 95 % de l'information.

### Formalisation

Soit $s(t) \in \{Tokyo, Londres, NY, LdN\_overlap, low\_liq\}$ la session de la barre $t$. On définit :

$$\text{ATR\_session\_zscore}(t) = \frac{\text{ATR}_{14}(t) - \mu_{s(t)}}{\sigma_{s(t)}}$$

où $\mu_{s(t)}, \sigma_{s(t)}$ sont la moyenne et l'écart-type de l'ATR **conditionnels à la session** (calculés sur train uniquement).

**Hypothèse $H_1$** : la distribution de $P(Y = +1 | X)$ varie significativement selon $s(t)$ ⇒ ajouter `session_id` comme feature catégorielle capture une partie de cette structure.

**Hypothèse $H_2$** : les patterns techniques (RSI, ADX) ont une signification différente selon la session (ex : un RSI 70 en session asiatique = sur-achat probable / en overlap LdN-NY = trend confirmé). Les **interactions** `RSI × session_id` peuvent être discriminantes.

---

## 2. Méthodologie d'implémentation

### Fichiers concernés

- **Étendre** [`learning_machine_learning/features/regime.py`](../learning_machine_learning/features/regime.py) avec :
  - `compute_session_id(index: pd.DatetimeIndex) -> pd.Series[int]` : retourne {0=Tokyo, 1=Londres, 2=NY, 3=LdN_overlap, 4=low_liq}
  - `compute_session_volatility_zscore(atr_norm: pd.Series, session_id: pd.Series, train_mask: pd.Series) -> pd.Series` : ATR normalisé par les stats train-only de la session courante
  - `compute_session_open_range(high: pd.Series, low: pd.Series, session_id: pd.Series) -> pd.Series` : cumul (H-L) depuis l'ouverture de session
  - `compute_relative_position_in_session(index: pd.DatetimeIndex, session_id: pd.Series) -> pd.Series` : (minutes_écoulées / durée_totale_session) ∈ [0, 1]

- **Modifier** [`learning_machine_learning/features/pipeline.py:99-106`](../learning_machine_learning/features/pipeline.py) pour ajouter le bloc de session après les autres features de régime H1.

- **Modifier** [`learning_machine_learning/features/pipeline.py:147-152`](../learning_machine_learning/features/pipeline.py) (`colonnes_finales`) : ajouter `session_id`, `ATR_session_zscore`, `session_open_range`, `relative_position_in_session`.

- **Décision encoding** : tester deux approches via flag config `session_encoding ∈ {"ordinal", "one_hot"}` dans `EurUsdConfig` :
  - Ordinal : 1 colonne `session_id ∈ [0, 4]` (compact, mais introduit un ordre artificiel)
  - One-hot : 5 colonnes binaires (plus de paramètres mais sémantiquement correct)
  - Avec GBM (LightGBM, step_03), l'ordinal avec `categorical_feature=['session_id']` est optimal.

### Choix techniques

- **Bornes des sessions** : utiliser UTC strict. Les bornes 7h/16h pour Londres correspondent à l'heure d'été — un "session_id" simple ignore le DST. Pour rester robuste, baser sur des heures UTC fixes (07:00-16:00) — accepter l'imprécision DST comme bruit acceptable.
- **Overlap LdN-NY** : 12:00-16:00 UTC est plus fiable que les bornes traditionnelles (chevauchement actif des deux pools de liquidité).
- **`session_volatility_zscore`** : NÉCESSITE des stats train-only. La fonction doit recevoir un `train_mask` ou être appelée en deux temps (fit train → apply all). Implémenter en classe `SessionVolatilityScaler` (analogue à StandardScaler).
- **`session_open_range`** : reset à chaque changement de session. Utiliser `groupby(session_id).cummax() - groupby(session_id).cummin()` après bornage par session.
- **Interactions explicites** : optionnel — si on garde RF, ajouter colonnes `RSI_14 × session_id_dummy`. Si GBM, laisser le modèle apprendre les interactions implicitement.

### Anti-leak / précautions
- **`SessionVolatilityScaler` train-only** : tester unitairement que `fit(X_train)` et `transform(X_test)` ne touchent jamais aux stats du test (test : changer drastiquement la distribution test, vérifier que le scaler s'en moque).
- **Pas de leak via `relative_position_in_session`** : cette feature ne regarde que le passé local de la session courante — OK.
- **`session_open_range` à $t=0$ de session** : NaN ou 0 ? Choisir 0 (range vide) et documenter.

---

## 3. Métriques de validation

### Métriques cibles
| Métrique | Baseline v15 | Objectif step_04 |
|---|---|---|
| Permutation importance `session_id` | NA | **> 0.005** (significatif) |
| Permutation importance `ATR_session_zscore` | NA | **> 0.005** |
| Accuracy OOS 2025 | 0.332 | **> 0.34** (gain marginal mais réel) |
| Sharpe OOS 2025 | +0.04 | **> 0.30** |
| Sharpe par session (overlap LdN-NY) | NA | **> 1.0** (sessions où le modèle excelle) |

### Métriques secondaires
- **Accuracy stratifiée par session** : reporter accuracy OOS conditionnelle à chaque session — devrait varier (sinon les features de session n'apportent rien).
- **Distribution des trades par session** : avant/après. Si tous les trades restent concentrés en NY, le modèle ne tire pas parti de la session info.
- **WR par session** : doit révéler 1-2 sessions clairement profitables et 1-2 perdantes — base pour un filtre conditionnel ultérieur.

### Critère d'arrêt
- Si **permutation_importance(session_id) < 0.003 ET Sharpe OOS gain < +0.10**, considérer que la session n'apporte rien (ou que le modèle ne sait pas l'exploiter — probablement à recouper avec step_03 GBM).
- Sinon → conserver les features et passer à step_05.

---

## 4. Risques & dépendances

- **R1 — Multicolinéarité avec `Hour_Sin/Cos`** : déjà droppés, donc pas de conflit direct, mais surveiller la stabilité de la feature importance.
- **R2 — Distribution non-stationnaire des sessions** : volume Forex a augmenté en Asie depuis 2020 (montée des trading desks asiatiques). Les statistiques de session calculées sur 2010-2018 peuvent ne plus être pertinentes en 2024-2025. Mitigation : recalibrer `SessionVolatilityScaler` à chaque retrain (intégrer step_02 walk-forward).
- **R3 — DST (Daylight Saving Time)** : ignoré dans l'implémentation simple. En période de transition (mars/octobre), les sessions Londres/NY décalent d'1h. Acceptable comme bruit, mais surveiller via reporting Sharpe mensuel.
- **R4 — Encoding catégoriel et GBM** : LightGBM gère bien `categorical_feature`, XGBoost mal (depuis v1.5 OK avec `enable_categorical=True`). Vérifier le backend choisi en step_03.

---

## 5. Références

- Bauwens, L., Hafner, C. M., & Laurent, S. (2012). *Handbook of Volatility Models and their Applications*, ch. 6 (Intraday volatility).
- Ranaldo, A. (2009). *Segmentation and time-of-day patterns in foreign exchange markets*. Journal of Banking & Finance.
- Code existant à étendre : [`learning_machine_learning/features/regime.py`](../learning_machine_learning/features/regime.py), [`backtest/filters.py:95`](../learning_machine_learning/backtest/filters.py) (SessionFilter actuel à compléter par les nouvelles features).
