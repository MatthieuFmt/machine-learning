# Stratégie manuelle Forex Swing — fichiers TradingView

## Contenu

| Fichier | Rôle |
|---|---|
| `indicateur_1_boussole_tendance.pine` | v1 d'origine (conservée) |
| `indicateur_2_signal_entree.pine` | v1 d'origine (conservée) |
| **`indicateur_1_boussole_tendance_v2.pine`** | Boussole D1 + **alertes de changement de régime** + mesure d'« étirement » (distance EMA200 en ATR) |
| **`indicateur_2_signal_entree_v2.pine`** | Signal H4 avec **filtre Daily intégré** (plus d'erreur d'alignement), signaux **à la clôture seulement**, **alertes avec Entrée/SL/TP**, SL « structure », pip automatique, niveau break-even +1R, filtre de session, tableau-checklist |
| **`strategie_backtest.pine`** | Version « strategy() » → onglet **Strategy Tester** de TradingView : tu vois toi-même WR, gain moyen, courbe de capital |
| `strategie.html` | Guide pas-à-pas + calculateur de taille de position |

## Installation (2 min par fichier)

1. TradingView → graphique → onglet **« Éditeur Pine »** (en bas).
2. Coller le contenu du fichier → **« Ajouter au graphique »**.
3. Pour les alertes : icône réveil ⏰ → « Créer une alerte » → Condition =
   l'indicateur v2 → choisir « Signal ACHAT (clôture H4) » etc. → Notification
   app mobile. **Une fois l'alerte créée, plus besoin de surveiller le
   graphique.**

## ⚠️ Honnêteté d'abord

Cette famille de stratégie (suivi de tendance + repli D1/H4 forex) a été testée
en backtest sur ce repo et n'a **jamais** montré d'avantage statistique net de
frais. Avant de risquer du vrai argent :

1. Lance le backtest Python (chiffres avec coûts + swap réels) :
   ```bash
   python scripts/screen_trend_pullback.py --assets EURUSD,GBPUSD,USDJPY,XAUUSD
   ```
2. Regarde le Strategy Tester TradingView (`strategie_backtest.pine`).
3. Si tu trades quand même : compte démo d'abord, risque ≤ 1 %, journal
   obligatoire (la valeur de cette stratégie est d'apprendre la discipline,
   pas de gagner de l'argent).

## Différence v1 → v2 la plus importante

En v1, c'est TOI qui devais vérifier que la flèche H4 allait dans le sens du
Daily — principal risque d'erreur. En v2, la flèche n'apparaît **que** si le
Daily (de la veille, déjà clôturé — anti-triche) est d'accord. Un seul
graphique, zéro erreur d'alignement, zéro repaint.
