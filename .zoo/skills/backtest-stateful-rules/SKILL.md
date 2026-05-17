---
name: backtest-stateful-rules
description: Règles backtest stateful — un seul trade à la fois, coûts explicites, pas d'oracle.
---

# Backtest Stateful Rules — Simulation Réaliste

## Règle cardinale

**UN seul trade ouvert à la fois. La sortie est le PREMIER événement parmi TP, SL, ou timeout. Les coûts de transaction (spread + commission + slippage) sont explicites et déduits du PnL. Pas d'information future dans la boucle de simulation.**

## Instructions

### 1. Architecture stateful

```python
# ✅ CORRECT : un seul trade à la fois
i = 0
while i < n:
    if signal[i] != 0:
        # Ouvre un trade
        for j in range(1, window + 1):
            if tp_hit or sl_hit:
                break
        i = exit_idx  # reprend après la sortie
    i += 1

# ❌ INCORRECT : vectorisé (plusieurs trades simultanés)
trades = df[df["Signal"] != 0]  # suppose qu'on peut tout trader en même temps
```

### 2. Modèle de coûts obligatoire

```python
# Coûts explicites, jamais ignorés
spread_cost_base = commission_pips + slippage_pips  # ex: 0.5 + 1.0 = 1.5 pips

# Coût total = spread réel (variable) + coûts fixes
spread_cost = spreads[i] / 10.0 + spread_cost_base  # spread raw → pips

# PnL net = PnL brut * weight - coûts
pips_brut = tp_pips - spread_cost  # si TP touché
pips_net = pips_brut * weight     # sizing appliqué
```

### 3. Gestion du timeout (B2 fix)

```python
# Timeout : PnL basé sur le Close à expiration (PAS sur le TP/SL)
# → C'est le prix réel auquel on peut sortir
exit_idx = min(i + window, n - 1)
exit_price = closes[exit_idx]
if signal == 1:  # LONG
    pips_brut = (exit_price - entry_price) / pip_size - spread_cost
else:             # SHORT
    pips_brut = (entry_price - exit_price) / pip_size - spread_cost
```

### 4. Ordre de priorité TP/SL dans la même barre

```python
# Si High touche TP ET Low touche SL dans la MÊME barre :
# → Vérifier le prix d'ouverture de la barre pour déterminer l'ordre
# → Si Open est plus proche de SL → SL touché en premier
# Approche simplifiée et conservatrice : SL prime sur TP
if curr_low <= sl:
    # SL touché — perte
    break
elif curr_high >= tp:
    # TP touché — gain
    break
```

### 5. Métriques minimales à calculer

```python
# Métriques obligatoires après simulation :
metrics = {
    "n_trades": len(trades_df),
    "win_rate": (trades_df["result"] == "win").mean() * 100,
    "profit_factor": abs(wins["Pips_Nets"].sum() / losses["Pips_Nets"].sum())
                     if len(losses) > 0 else float("inf"),
    "pnl_total_pips": trades_df["Pips_Nets"].sum(),
    "pnl_moyen_par_trade": trades_df["Pips_Nets"].mean(),
    "max_drawdown_pips": compute_max_drawdown(trades_df["Pips_Nets"].cumsum()),
    "expectancy": trades_df["Pips_Nets"].mean(),  # espérance par trade
}
```

### 6. Filtres de régime — injection, pas hardcoding

```python
# ✅ CORRECT : filtres injectés comme dépendance
filter_pipeline = FilterPipeline([TrendFilter(), VolFilter(), SessionFilter()])
trades_df, n_signaux, filtres = simulate_trades(df, weight_func=..., filter_pipeline=filter_pipeline)

# ❌ INCORRECT : filtres hardcodés dans simulate_trades
if df["ADX_14"] < 25:
    mask_long = False
```

## Checklist finale

1. Un seul trade ouvert à la fois (stateful)
2. Coûts explicites (spread + commission + slippage)
3. Timeout → Close réel (pas TP/SL)
4. SL prioritaire sur TP si même barre
5. Sizing injecté comme fonction (weight_func)
6. Pas de `df.shift(-1)` dans la boucle de simulation
