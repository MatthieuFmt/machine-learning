# Guide — Chercher une stratégie fiable (Phase 1)

> Guide pas-à-pas pour débutant. Objectif : trouver une stratégie de trading qui
> survit à une validation honnête, AVANT de risquer le moindre euro.
> **À lire avant `prompts/00_constitution.md` si tu veux juste lancer la recherche.**

## 0. Le principe (à comprendre une fois)

La plupart des « stratégies rentables » qu'on trouve en ligne sont des illusions :
elles paraissent gagnantes sur le passé mais perdent en réel, parce que le backtest
était trop optimiste (coûts ignorés, look-ahead) ou parce qu'on a testé 1000 idées
et gardé celle qui marchait par hasard (data-snooping).

Ce projet applique une discipline stricte pour éviter ces deux pièges :
- **Backtest honnête** : entrée à l'ouverture de la barre suivante (on ne peut pas
  trader dans le passé), coûts XTB réels (spread + slippage + commission) et swap.
- **Validation hors-échantillon (OOS)** : on choisit la stratégie sur une période
  ancienne (in-sample), puis on la teste UNE seule fois sur une période récente
  jamais regardée. On compte chaque essai (`n_trials`) pour pénaliser le hasard (DSR).

**Résultat le plus probable et NORMAL : aucune stratégie ne passe.** Ce n'est pas un
échec, c'est de l'honnêteté. On itère alors sur de nouvelles idées. Une seule
stratégie qui passe vraiment vaut mille faux positifs.

## 1. Installer (une fois)

```bash
pip install -r requirements.txt
```

## 2. Récupérer les données

Le dépôt ne contient AUCUNE donnée. Il faut peupler `data/raw/` avec des CSV au
format `data/raw/<ACTIF>/<NOM>_<TF>.csv` (séparateur tabulation, colonnes
`Time, Open, High, Low, Close, Volume`).

Le téléchargeur Dukascopy existe :

```bash
python scripts/download_dukascopy_full.py        # voir les options en tête du fichier
```

> ⚠️ Le téléchargement de données ne fonctionne PAS dans l'environnement cloud de
> Claude (réseau bloqué pour Yahoo/Dukascopy). Lance-le sur **ta machine**.

Actifs disponibles chez XTB et déjà configurés (coûts réels) :
`US30, US500, GER30, XAUUSD, XAGUSD, USOIL, EURUSD, GBPUSD, USDCHF, BTCUSD, ETHUSD, USDJPY`.

**Conseil (basé sur l'historique du projet)** : commence par les actifs les plus
tendanciels — **BTCUSD, ETHUSD, XAUUSD** en **D1** — c'est là que le trend-following
a le plus de chances. Le forex et les indices sont surtout en range (hostiles).

## 3. Lancer la recherche

```bash
# Tout ce qui est dans data/raw/, en D1 et H4 :
python scripts/screen_edge.py

# Cibler les actifs tendanciels en daily :
python scripts/screen_edge.py --assets BTCUSD,ETHUSD,XAUUSD --timeframes D1

# Choisir la frontière in-sample / out-of-sample (défaut 2024-01-01) :
python scripts/screen_edge.py --oos-start 2024-01-01
```

Sortie : un tableau classé (GO d'abord, puis par DSR) + un CSV
(`predictions/edge_screen_results.csv`). Exemple de ligne :

```
❌ NO-GO  XAUUSD/D1  [Donchian55 TP45/SL22]  IS Sharpe=0.89 (23 tr) |
          OOS Sharpe=0.31 DSR=-1.20 (p=0.78) WR=41% DD=12% (18 tr) [n_trials=14]
```

## 4. Lire le verdict

Une stratégie est **GO** seulement si TOUS ces critères passent sur l'OOS :

| Critère | Seuil | Pourquoi |
|---|---|---|
| Sharpe | ≥ 1.0 | Rendement/risque suffisant |
| DSR (Deflated Sharpe) | > 0 et p < 0.05 | Significatif APRÈS correction du nombre d'essais |
| Max Drawdown | < 15 % | Perte maximale supportable |
| Win rate | > 30 % | Évite les stratégies dégénérées |
| Trades/an | ≥ 30 | Assez de données pour conclure |

Le **DSR** est le critère clé : il dégrade le Sharpe en fonction de `n_trials`
(nombre de configurations testées). Plus tu testes d'idées, plus la barre monte.
C'est ce qui empêche de « trouver » un edge par pur hasard.

## 5. Et si rien ne passe ?

C'est le cas attendu au début. Options, par ordre de valeur :
1. **Nouvelles familles de stratégies** jamais testées sérieusement (voir
   `CLAUDE.md` §2) : carry sur paires JPY, Opening Range Breakout, pairs
   trading / cointégration, effets de calendrier (pre-FOMC).
2. **Filtrage de régime** : ne trader le trend que quand le marché EST tendanciel.
3. Élargir l'univers d'actifs.

Chaque nouvelle idée s'ajoute via `app/strategies/` puis se teste avec le même
`screen_edge.py` — la discipline reste identique.

## 6. Quand (et SEULEMENT quand) un edge passe

1. **Valider en compte DÉMO XTB** plusieurs semaines (les prix/spreads XTB diffèrent
   des données Dukascopy).
2. Brancher l'alerte Telegram (specs déjà écrites dans `prompts/20-24`).
3. Ne passer en réel qu'après confirmation démo, avec un risque par trade fixe (2 %).

> Règle d'or : **aucun euro réel tant qu'une stratégie n'a pas passé l'OOS PUIS la
> démo.** Un bot qui n'alerte jamais vaut mieux qu'un bot qui fait perdre de l'argent.
