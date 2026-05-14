# Prompt 18 — Validation finale GO/NO-GO production

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (état complet recherche v3)
3. `prompts/17_h16_timeframe_stacking.md`

## Objectif
**Décision GO/NO-GO production**. Vérifier que la configuration finale (sleeves retenus + filtres régime + méta-labeling + pondération portfolio + TF + stacking) respecte **TOUS** les 5 critères de la constitution :

1. Sharpe walk-forward portfolio ≥ 1.0
2. DSR > 0 et p < 0.05
3. Max DD < 15 %
4. WR > 30 %
5. ≥ 30 trades/an

Si UN SEUL critère ne passe pas → NO-GO. Itérer en revenant au prompt approprié.

## Definition of Done (testable)
- [ ] `scripts/run_validation_finale.py` charge la config retenue (typée `ProductionConfig` depuis `app/config/models.py`), rejoue le portfolio complet sur test 2024-2025, calcule `validate_edge` avec `n_trials = valeur lue depuis le tableau JOURNAL.md` (cf. section 6bis constitution — pas recompté à la volée).
- [ ] **Comparaison vs 2 benchmarks naïfs OBLIGATOIRE** :
  1. **Buy-and-Hold equal weight** : portfolio qui achète et garde les actifs retenus à parts égales. Le portfolio stratégique doit produire `Sharpe ≥ Sharpe_BH + 0.3`.
  2. **Monte Carlo random** : 1 000 simulations de signaux aléatoires (proba d'entrée = proba d'entrée du modèle, direction = bernoulli 50/50). Le portfolio doit être > P95 des 1000 random Sharpes.
- [ ] Si l'un des 2 benchmarks bat le portfolio → **NO-GO** (le portfolio n'apporte pas d'edge réel).
- [ ] **Avant le verdict** : `python scripts/verify_no_snooping.py` retourne 0.
- [ ] **Si verdict GO** : appel `from app.testing.snooping_guard import lock; lock(prompt="18")` → `TEST_SET_LOCK.json` est verrouillé. Aucune modification de stratégie n'est plus possible (sauf nouveau split temporel).
- [ ] Génère un rapport `predictions/validation_finale.json` :
  ```json
  {
    "config": {...},
    "metrics": {...},
    "go": true/false,
    "reasons": [...],
    "n_trials": 18,
    "date": "..."
  }
  ```
- [ ] Génère un rapport humain `docs/v3_final_report.md` :
  - Récap stratégies / sleeves retenus
  - Tableau des 5 critères : valeur observée vs cible
  - Verdict GO / NO-GO
  - Si NO-GO : prompt(s) à reprendre
- [ ] Si GO : `JOURNAL.md` reçoit « ✅ Edge validé. Passage en Phase 4 (production). »
- [ ] Si NO-GO : `JOURNAL.md` documente les critères ratés et l'itération à faire.

## NE PAS FAIRE
- Ne PAS modifier la config en réaction au résultat (data snooping). Si NO-GO, revenir AVANT et changer l'hypothèse, ne pas tweaker post-hoc.
- Ne PAS passer en production si UN SEUL critère ne passe pas.
- Ne PAS calculer le Sharpe sur PnL/trade (Règle 10 constitution).
- Ne PAS modifier `n_trials` pour faire passer le DSR.
- Ne PAS sauter la comparaison benchmark — un Sharpe ≥ 1.0 qui ne bat pas B&H = stratégie inutile.
- Ne PAS appeler `lock()` avant d'avoir reçu confirmation utilisateur explicite (le lock est irréversible).

## Étapes

### Étape 1 — Charger la config finale
Lecture de `JOURNAL.md` pour récupérer :
- Sleeves retenus (par exemple : `["donchian_US30", "donchian_GER30", "chandelier_US500"]`)
- Filtre régime activé ou non
- Méta-labeling : aucun / per_asset / multi_asset
- Pondération : equal_risk / correlation_aware
- Vol targeting : oui / non
- TF : D1 / H4 / H1
- Stacking : oui / non

### Étape 2 — Reproduire le pipeline complet
Un seul backtest end-to-end qui chaîne tous les composants retenus.

### Étape 3 — `validate_edge` avec n_trials cumulatif
```python
n_trials = 5 + n_hypotheses_v3_run  # depuis JOURNAL.md
report = validate_edge(equity_portfolio, all_trades, n_trials)
```

### Étape 4 — Benchmarks naïfs
```python
def buy_and_hold_benchmark(assets: list[str], start: str, end: str) -> pd.Series:
    """Equity B&H equal weight des actifs retenus, frais inclus à l'achat."""
    returns = pd.DataFrame()
    for a in assets:
        df = load_asset(a, "D1").loc[start:end]
        returns[a] = df["close"].pct_change()
    portfolio = returns.mean(axis=1)
    return (1 + portfolio).cumprod()

def random_monte_carlo(n_iter: int = 1000, signal_freq: float = 0.05) -> np.ndarray:
    """Simule 1000 stratégies random : entrée bernoulli(signal_freq), direction 50/50."""
    sharpes = np.zeros(n_iter)
    for i in range(n_iter):
        rng = np.random.default_rng(seed=i)
        # ... génère signaux random, applique simulator avec MÊMES coûts ...
        sharpes[i] = sharpe_ratio(equity.pct_change().dropna())
    return sharpes

# Verdict benchmarks
sr_portfolio = report.metrics["sharpe"]
sr_bh = sharpe_ratio(bh_equity.pct_change().dropna())
mc_sharpes = random_monte_carlo()
p95_random = np.percentile(mc_sharpes, 95)

bench_ok = (sr_portfolio >= sr_bh + 0.3) and (sr_portfolio > p95_random)
```

### Étape 5 — Rapport humain
Tableau clair :

| Critère | Cible | Observé | Verdict |
|---|---|---|---|
| Sharpe | ≥ 1.0 | 1.3 | ✅ |
| DSR | > 0, p < 0.05 | 0.7 (p=0.02) | ✅ |
| Max DD | < 15 % | 12 % | ✅ |
| WR | > 30 % | 34 % | ✅ |
| Trades/an | ≥ 30 | 42 | ✅ |
| **Beat B&H+0.3** | Sharpe > 0.8 | 1.3 vs B&H 0.5 | ✅ |
| **Beat P95 random** | Sharpe > 0.9 | 1.3 vs P95 0.6 | ✅ |
| **GLOBAL** | — | — | **✅ GO** |

### Étape 6 — Lock du test set (si GO)
```bash
python scripts/verify_no_snooping.py  # doit retourner 0
```
Puis (sur confirmation utilisateur) :
```python
from app.testing.snooping_guard import lock
lock(prompt="18")  # IRRÉVERSIBLE
```

### Étape 5 — Décision

- **GO** → continuer prompt 19.
- **NO-GO** :
  - Sharpe insuffisant → prompt 14 (vol targeting) ou prompt 11 (méta-labeling)
  - DD trop grand → prompt 15 (vol targeting) ou prompt 14 (corrélation)
  - WR insuffisant → prompt 10 (régime) ou prompt 11 (méta-labeling)
  - Trades/an insuffisant → prompt 16 (changer TF) ou prompt 08 (ajouter stratégies)
  - DSR non significatif → revoir n_trials, peut-être réduire les hypothèses testées (paradoxe : plus on cherche, plus DSR pénalise)

## Critères go/no-go
- **GO prompt 19** si : ✅ GO sur les 5 critères. **Validation utilisateur explicite obligatoire avant prompt 19.**
- **NO-GO** : revenir au prompt indiqué. NE PAS forcer le passage.
