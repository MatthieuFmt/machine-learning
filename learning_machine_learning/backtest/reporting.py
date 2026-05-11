"""Reporting — génération de rapports Markdown et sauvegarde des trades."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def save_trades_detailed(
    trades_df: pd.DataFrame,
    annee: int,
    df: pd.DataFrame | None = None,
    output_dir: str | Path = "results",
) -> str | None:
    """Persiste la liste des trades enrichis pour analyse ex-post.

    Args:
        trades_df: DataFrame de trades.
        annee: Année du backtest.
        df: DataFrame d'entrée avec features et probas (optionnel).
        output_dir: Répertoire de sortie.

    Returns:
        Chemin du fichier sauvegardé, ou None si trades_df vide.
    """
    if trades_df.empty:
        return None

    out = trades_df.copy()

    if df is not None:
        proba_cols = ["Confiance_Hausse_%", "Confiance_Neutre_%", "Confiance_Baisse_%"]
        feature_cols = [
            c
            for c in df.columns
            if c
            not in (
                [
                    "High",
                    "Low",
                    "Close",
                    "Spread",
                    "Signal",
                    "Weight",
                    "proba_max",
                    "Prediction_Modele",
                    "Close_Reel_Direction",
                    "Filter_Rejected",
                ]
                + proba_cols
            )
        ]
        enrich_cols = feature_cols + proba_cols
        enrich_cols = [c for c in enrich_cols if c in df.columns]
        enrich = df.loc[df.index.intersection(out.index), enrich_cols].copy()
        enrich = enrich.rename(
            columns={
                "Confiance_Hausse_%": "proba_hausse",
                "Confiance_Neutre_%": "proba_neutre",
                "Confiance_Baisse_%": "proba_baisse",
            }
        )
        out = out.join(enrich, how="left")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"Trades_Detailed_{annee}.csv"
    out.to_csv(path)
    logger.info("Trades détaillés sauvegardés : %s", path)
    return str(path)


def save_report_md(
    metrics: dict,
    annee: int,
    output_dir: str | Path = "predictions",
    version: str | None = None,
    notes: str | None = None,
    n_signaux: int | None = None,
    config_block: str = "",
) -> str:
    """Génère un rapport Markdown à partir du dict de métriques.

    Args:
        metrics: Dict retourné par compute_metrics().
        annee: Année testée.
        output_dir: Répertoire de sortie racine.
        version: Sous-dossier optionnel (ex: 'ratio_1_1').
        notes: Notes additionnelles en fin de rapport.
        n_signaux: Nombre de signaux générés (pour ratio signaux/trades).
        config_block: Description texte de la configuration.

    Returns:
        Chemin du fichier généré.
    """
    target_dir = Path(output_dir)
    if version:
        target_dir = target_dir / version
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"Rapport_Performance_{annee}.md"

    n_trades = metrics.get("trades", 0)
    profit_net = metrics.get("profit_net", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    dd = metrics.get("dd", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    sharpe_per_trade = metrics.get("sharpe_per_trade", 0.0)
    total_return_pct = metrics.get("total_return_pct", 0.0)
    max_dd_pct = metrics.get("max_dd_pct", 0.0)
    bh_pips = metrics.get("bh_pips", 0.0)
    bh_return_pct = metrics.get("bh_return_pct", 0.0)
    alpha_pips = metrics.get("alpha_pips", 0.0)
    alpha_return_pct = metrics.get("alpha_return_pct", 0.0)
    esperance = profit_net / n_trades if n_trades else 0.0

    lines = [
        "# 📈 Rapport de Performance",
        f"**Année testée :** {annee}",
    ]
    if version:
        lines.append(f"**Version :** {version}")
    if config_block:
        lines.append(f"**Configuration :** {config_block}")

    lines += [
        "",
        "## 📊 Stratégie",
        "| Métrique | Valeur |",
        "| :--- | :--- |",
        f"| Nombre de Trades | {n_trades} |",
        f"| Win Rate | {win_rate:.2f}% |",
        f"| Résultat Net | **{profit_net:.1f} pips** ({total_return_pct:+.2f}%) |",
        f"| Max Drawdown | {dd:.1f} pips ({max_dd_pct:.2f}%) |",
        f"| Espérance par trade | {esperance:.2f} pips/trade |",
        f"| Sharpe (returns annualisés) | {sharpe:.2f} |",
        f"| Sharpe per-trade | {sharpe_per_trade:.2f} |",
    ]

    if n_signaux is not None:
        ratio = n_signaux / n_trades if n_trades else 0
        lines.append(
            f"| Signaux générés / trades exécutés | {n_signaux} / {n_trades} (×{ratio:.2f}) |"
        )

    lines += [
        "",
        "## 📊 Benchmark Buy & Hold",
        "| Métrique | Valeur |",
        "| :--- | :--- |",
        f"| Buy & Hold Net | {bh_pips:+.1f} pips ({bh_return_pct:+.2f}%) |",
        f"| Alpha (stratégie − B&H) | **{alpha_pips:+.1f} pips ({alpha_return_pct:+.2f}%)** |",
    ]

    if notes:
        lines += ["", "## 📝 Notes", notes]

    lines += ["", "*Généré automatiquement par backtest.reporting.save_report_md*"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Rapport sauvegardé : %s", path)
    return str(path)
