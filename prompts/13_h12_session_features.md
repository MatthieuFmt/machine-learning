# Prompt 13 — H12 bis : Features de session (Tokyo / Londres / NY)

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/12_h11_features_advanced.md`
4. `docs/v3_roadmap.md`

## Objectif
Ajouter des features de session de trading (Tokyo, Londres, NY, overlap) ET des features de jour de la semaine. Tester l'impact sur le méta-labeling. **Applicable surtout au timeframe intraday (H4/H1)** — sur D1 l'apport est limité.

## Definition of Done (testable)
- [ ] `app/features/sessions.py` contient :
  - `session_tokyo(index, weekday_filter=True)` : 1 si timestamp dans 00:00–09:00 UTC
  - `session_london(index)` : 1 si 07:00–16:00 UTC
  - `session_ny(index)` : 1 si 13:00–22:00 UTC
  - `session_overlap_london_ny(index)` : 1 si 13:00–16:00 UTC
  - `day_of_week(index)` : 0–4 (lundi-vendredi), 5/6 ignorés sur D1
  - `is_monday(index)`, `is_friday(index)` : booléens
  - `days_to_month_end(index)` : nombre de jours jusqu'à fin du mois
  - `days_to_quarter_end(index)` : idem trimestre
- [ ] Tests : chaque feature retourne la bonne valeur sur cas synthétiques (timestamps UTC connus).
- [ ] `scripts/run_h12_bis_session_features.py` :
  - Sur les sleeves intraday (H4/H1 si dispo), ajouter ces features au RF méta-labeling
  - Comparer Sharpe avec/sans
- [ ] Rapport `docs/v3_hypothesis_12_bis.md`.

## NE PAS FAIRE
- Ne PAS supposer un fuseau horaire local — tout en UTC.
- Ne PAS ajouter ces features sur D1 (1 barre/jour, l'info session est dégénérée).
- Ne PAS oublier les jours fériés (impact marginal, à documenter mais pas à implémenter ici).

## Étapes

### Étape 1 — Features
```python
def session_london(index: pd.DatetimeIndex) -> pd.Series:
    h = index.hour
    return ((h >= 7) & (h < 16)).astype(int)


def is_monday(index: pd.DatetimeIndex) -> pd.Series:
    return (index.weekday == 0).astype(int)
```

### Étape 2 — Test sur sleeves intraday
Si TF retenu (cf. prompt 16) inclut H4 ou H1, intégrer. Sinon, documenter que le prompt n'apporte rien sur D1 et passer.

## Critères go/no-go
- **GO prompt 14** systématique. Si TF = D1 only : ce prompt produit juste les fonctions, sans impact métier.
