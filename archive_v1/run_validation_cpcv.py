"""Script orchestrateur CPCV + DSR — Step 02.

Exécute le framework de validation robuste :
1. Charge les données EUR/USD via EurUsdPipeline.
2. Génère les splits CPCV sur la période OOS (2024-2025).
3. Exécute le backtest sur chaque split (parallélisé joblib).
4. Agrège les métriques et calcule le DSR distributionnel.
5. Exécute la validation principale (train ≤ 2023, test = 2025).
6. Sauvegarde CSV + rapport Markdown.
7. Affiche le verdict GO/NO-GO.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from learning_machine_learning.analysis.cpcv import (
    aggregate_cpcv_metrics,
    generate_cpcv_splits,
    run_cpcv_backtest,
)
from learning_machine_learning.analysis.edge_validation import (
    deflated_sharpe_ratio_from_distribution,
    validate_edge_distribution,
)
from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.config.instruments import InstrumentConfig
from learning_machine_learning.config.model import ModelConfig
from learning_machine_learning.config.registry import ConfigRegistry
from learning_machine_learning.core.logging import get_logger
from learning_machine_learning.model.training import (
    _FILTER_ONLY_COLS,
    train_model,
    train_regressor,
)
from learning_machine_learning.pipelines.eurusd import EurUsdPipeline

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Paramètres CPCV
# ═══════════════════════════════════════════════════════════════════════════
N_GROUPS: int = 48          # 2 semaines/groupe sur 24 mois OOS
K_TEST: int = 12            # 6 mois de test par split
N_SAMPLES: int = 200        # Combinaisons aléatoires
PURGE_HOURS: int = 48       # Cohérent avec ModelConfig.purge_hours
OOS_START: str = "2024-01-01"
OOS_END: str = "2025-12-31"
RANDOM_STATE: int = 42


def main() -> None:
    """Point d'entrée principal du script CPCV."""
    t0 = datetime.now()
    logger.info("=" * 70)
    logger.info("Step 02 — CPCV + DSR Validation Framework — EURUSD H1")
    logger.info("=" * 70)

    # ── Registry ─────────────────────────────────────────────────────────
    registry = ConfigRegistry()
    entry = registry.get("EURUSD")
    instrument_cfg: InstrumentConfig = entry.instrument
    model_cfg: ModelConfig = entry.model
    backtest_cfg: BacktestConfig = entry.backtest
    paths = entry.paths
    paths.ensure_dirs()

    target_mode = instrument_cfg.target_mode
    logger.info(
        "Config: target_mode=%s, tp=%.1f, sl=%.1f, commission=%.1f, slippage=%.1f",
        target_mode, backtest_cfg.tp_pips, backtest_cfg.sl_pips,
        backtest_cfg.commission_pips, backtest_cfg.slippage_pips,
    )

    # ── 1. Charger les données ───────────────────────────────────────────
    logger.info("Chargement des données EUR/USD...")
    pipeline = EurUsdPipeline()
    data = pipeline.load_data()
    ml_data_full = pipeline.build_features(data)
    logger.info(
        "ML-ready: %d barres, %d colonnes",
        len(ml_data_full), len(ml_data_full.columns),
    )

    # ── 2. Déterminer X_cols ────────────────────────────────────────────
    drop_set: set[str] = (
        {"Time", "Target"}
        | _FILTER_ONLY_COLS
        | set(instrument_cfg.features_dropped)
    )
    X_cols = [c for c in ml_data_full.columns if c not in drop_set]
    logger.info("X_cols: %d features (drop_set=%s)", len(X_cols), sorted(drop_set))

    # ── 3. Période OOS pour CPCV ────────────────────────────────────────
    oos_ml_data = ml_data_full.loc[OOS_START:OOS_END]
    if len(oos_ml_data) == 0:
        logger.error("Aucune donnée OOS pour %s → %s.", OOS_START, OOS_END)
        return

    n_months = len(oos_ml_data) / (24 * 365.25 / 12)
    logger.info(
        "Période OOS CPCV: %s → %s (%d barres, %.1f mois)",
        oos_ml_data.index.min().date(),
        oos_ml_data.index.max().date(),
        len(oos_ml_data),
        n_months,
    )

    # ── 4. Générer splits CPCV ──────────────────────────────────────────
    oos_index = pd.DatetimeIndex(oos_ml_data.index)
    cpcv_splits = list(generate_cpcv_splits(
        index=oos_index,
        n_groups=N_GROUPS,
        k_test=K_TEST,
        purge_hours=PURGE_HOURS,
        n_samples=N_SAMPLES,
        random_state=RANDOM_STATE,
    ))
    logger.info(
        "Splits CPCV générés: %d / demandés %d", len(cpcv_splits), N_SAMPLES,
    )

    # ── 5. Model factory (n_jobs=1 pour éviter fork-bomb) ───────────────
    if target_mode == "forward_return":
        gbm_params = dict(model_cfg.gbm_params)

        def model_factory(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
            return train_regressor(X_train, y_train, gbm_params)
    else:
        rf_params = dict(model_cfg.rf_params)
        rf_params["n_jobs"] = 1  # ⚠️ fork-bomb prevention avec joblib.Parallel

        def model_factory(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
            return train_model(X_train, y_train, rf_params)

    # ── 6. Exécuter CPCV ────────────────────────────────────────────────
    logger.info(
        "Lancement CPCV backtest (%d splits, n_jobs=-1)...", len(cpcv_splits),
    )
    ohlcv_h1: pd.DataFrame = data["h1"]

    results_df = run_cpcv_backtest(
        ml_data=oos_ml_data,
        ohlcv_h1=ohlcv_h1,
        splits=iter(cpcv_splits),
        model_factory=model_factory,
        backtest_cfg=backtest_cfg,
        instrument_cfg=instrument_cfg,
        X_cols=X_cols,
        target_mode=target_mode,
        confidence_threshold=backtest_cfg.confidence_threshold,
        continuous_signal_threshold=backtest_cfg.continuous_signal_threshold,
        n_jobs=-1,
    )

    # ── 7. Agrégation ───────────────────────────────────────────────────
    cpcv_agg = aggregate_cpcv_metrics(results_df)
    sharpe_dist: np.ndarray = results_df["sharpe"].values.astype(np.float64)

    # ── 8. DSR distributionnel ──────────────────────────────────────────
    observed_sr = cpcv_agg["sharpe"]["mean"]
    dsr_result = deflated_sharpe_ratio_from_distribution(
        observed_sr=observed_sr,
        sharpe_distribution=sharpe_dist,
    )
    logger.info(
        "DSR distributionnel: DSR=%.4f, SR₀*=%.4f, E[SR]=%.4f±%.4f",
        dsr_result["dsr"], dsr_result["sr0_star"],
        cpcv_agg["sharpe"]["mean"], cpcv_agg["sharpe"]["std"],
    )

    # ── 9. Split principal (train≤2023, test=2025) ──────────────────────
    logger.info(
        "Split principal: train≤%d, test=2025...", model_cfg.train_end_year,
    )
    principal_metrics: dict[int, Any] = {}
    try:
        principal_model, principal_X_cols = pipeline.train_model(ml_data_full)
        principal_preds = pipeline.evaluate_model(
            principal_model, ml_data_full, principal_X_cols,
        )
        principal_trades, principal_metrics = pipeline.run_backtest(
            principal_preds, ml_data_full, ohlcv_h1,
        )

        trades_2025 = principal_trades.get(2025, pd.DataFrame())
        if not trades_2025.empty and "Pips_Nets" in trades_2025.columns:
            edge_result = validate_edge_distribution(
                trades_df=trades_2025,
                backtest_cfg=backtest_cfg,
                sharpe_distribution=sharpe_dist,
                n_trials_searched=N_SAMPLES,
                n_bootstrap=10_000,
                random_state=RANDOM_STATE,
            )
            logger.info(
                "Split principal 2025: Sharpe=%.4f, PSR(Bailey)=%.4f, DSR=%.4f",
                edge_result.get("bootstrap_sharpe", {}).get(
                    "observed", float("nan"),
                ),
                edge_result.get("psr_bailey", float("nan")),
                (
                    edge_result.get("cpcv_dsr", {}).get("dsr", float("nan"))
                    if isinstance(edge_result.get("cpcv_dsr"), dict)
                    else float("nan")
                ),
            )
        else:
            edge_result: dict[str, Any] = {
                "error": "Aucun trade sur 2025", "n_trades": 0,
            }
            logger.warning("Split principal: aucun trade sur 2025.")
    except Exception as exc:
        logger.error("Split principal: erreur — %s", exc)
        edge_result = {"error": str(exc)}

    # ── 10. Sauvegarde CSV ──────────────────────────────────────────────
    csv_path = paths.predictions / "cpcv_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Résultats CPCV sauvegardés: %s", csv_path)

    # ── 11. Rapport Markdown ────────────────────────────────────────────
    md_path = paths.predictions / "cpcv_report.md"
    _write_report(
        md_path=md_path,
        cpcv_agg=cpcv_agg,
        dsr_result=dsr_result,
        edge_result=edge_result,
        principal_metrics=principal_metrics,
        backtest_cfg=backtest_cfg,
        model_cfg=model_cfg,
        n_groups=N_GROUPS,
        k_test=K_TEST,
        n_samples=N_SAMPLES,
        purge_hours=PURGE_HOURS,
        target_mode=target_mode,
        n_splits_actual=len(results_df),
        elapsed=(datetime.now() - t0).total_seconds(),
    )
    logger.info("Rapport CPCV sauvegardé: %s", md_path)

    # ── 12. Verdict ─────────────────────────────────────────────────────
    _print_verdict(cpcv_agg, dsr_result)
    logger.info("Terminé en %.0fs.", (datetime.now() - t0).total_seconds())


def _write_report(
    *,
    md_path: Any,
    cpcv_agg: dict[str, Any],
    dsr_result: dict[str, Any],
    edge_result: dict[str, Any],
    principal_metrics: dict[int, Any],
    backtest_cfg: BacktestConfig,
    model_cfg: ModelConfig,
    n_groups: int,
    k_test: int,
    n_samples: int,
    purge_hours: int,
    target_mode: str,
    n_splits_actual: int,
    elapsed: float,
) -> None:
    """Génère le rapport Markdown CPCV."""
    lines: list[str] = []
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _s(key: str, default: str = "N/A") -> str:
        d = cpcv_agg.get("sharpe", {})
        val = d.get(key, default)
        if isinstance(val, float):
            return f"{val:.4f}" if not np.isnan(val) else "NaN"
        return str(val)

    def _d(key: str, default: str = "N/A") -> str:
        val = dsr_result.get(key, default)
        if isinstance(val, float):
            return f"{val:.4f}" if not np.isnan(val) else "NaN"
        return str(val)

    def _e(key: str, default: str = "N/A") -> str:
        val = edge_result.get(key, default)
        if isinstance(val, dict):
            return "…"
        if isinstance(val, float):
            return f"{val:.4f}" if not np.isnan(val) else "NaN"
        return str(val)

    # ── En-tête ──────────────────────────────────────────────────────────
    lines.append("# Rapport CPCV — EURUSD H1")
    lines.append("")
    lines.append(f"**Date** : {date_str}")
    lines.append(f"**Temps d'exécution** : {elapsed:.0f}s")
    lines.append("")

    # ── Configuration ────────────────────────────────────────────────────
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Paramètre | Valeur |")
    lines.append("|---|---|")
    lines.append(f"| n_groups | {n_groups} |")
    lines.append(f"| k_test | {k_test} |")
    lines.append(f"| n_samples (demandé) | {n_samples} |")
    lines.append(f"| n_splits (réalisés) | {n_splits_actual} |")
    lines.append(f"| purge_hours | {purge_hours}h |")
    lines.append(f"| target_mode | {target_mode} |")
    lines.append(
        f"| Model | RandomForest "
        f"(n={model_cfg.rf_n_estimators}, depth={model_cfg.rf_max_depth}) |"
    )
    lines.append(
        f"| TP/SL | {backtest_cfg.tp_pips:.1f} / {backtest_cfg.sl_pips:.1f} pips |"
    )
    lines.append(f"| Commission | {backtest_cfg.commission_pips:.2f} pips |")
    lines.append(f"| Slippage | {backtest_cfg.slippage_pips:.2f} pips |")
    filter_desc = (
        "Momentum+Vol+Session" if backtest_cfg.use_momentum_filter
        else "Session seul"
    )
    lines.append(f"| Filtres | {filter_desc} |")
    lines.append("")

    # ── Métriques CPCV ───────────────────────────────────────────────────
    lines.append("## Métriques CPCV (distribution sur les splits)")
    lines.append("")
    lines.append("| Métrique | Valeur |")
    lines.append("|---|---|")
    lines.append(
        f"| Splits valides | {cpcv_agg.get('n_splits_valid', 'N/A')} "
        f"/ {cpcv_agg.get('n_splits', 'N/A')} |"
    )
    lines.append(f"| % splits profitables | {cpcv_agg.get('pct_profitable', 'N/A')}% |")
    lines.append(f"| E[Sharpe] | {_s('mean')} |")
    lines.append(f"| σ[Sharpe] | {_s('std')} |")
    lines.append(f"| Sharpe médian | {_s('median')} |")
    lines.append(f"| Sharpe min | {_s('min')} |")
    lines.append(f"| Sharpe max | {_s('max')} |")
    lines.append(f"| Sharpe CI 95% | [{_s('ci_95_lower')}, {_s('ci_95_upper')}] |")
    nt = cpcv_agg.get("n_trades", {})
    lines.append(
        f"| Trades/split (moyenne) | "
        f"{nt.get('mean', 'N/A')} ± {nt.get('std', 'N/A')} |"
    )
    lines.append(f"| Trades totaux | {nt.get('total', 'N/A')} |")
    lines.append("")

    # ── DSR distributionnel ──────────────────────────────────────────────
    lines.append("## DSR Distributionnel (López de Prado 2014)")
    lines.append("")
    lines.append("| Métrique | Valeur |")
    lines.append("|---|---|")
    lines.append(f"| DSR | {_d('dsr')} |")
    lines.append(f"| PSR(SR*=0) | {_d('psr_zero')} |")
    lines.append(f"| SR₀* (seuil déflaté) | {_d('sr0_star')} |")
    lines.append(f"| E[max SR] sous H₀ | {_d('e_max_sr')} |")
    lines.append(f"| Var[SR] CPCV | {_d('var_sr')} |")
    lines.append(f"| N splits pour DSR | {_d('n_splits')} |")
    lines.append(f"| % profitables (DSR) | {_d('pct_profitable')}% |")
    lines.append(f"| E[SR] distrib | {_d('mean_sr')} |")
    lines.append(f"| σ[SR] distrib | {_d('std_sr')} |")
    lines.append(f"| SR min distrib | {_d('min_sr')} |")
    lines.append(f"| SR max distrib | {_d('max_sr')} |")
    lines.append(f"| SR médian distrib | {_d('median_sr')} |")
    lines.append(
        f"| CI 95% distrib | "
        f"[{_d('ci_95_lower')}, {_d('ci_95_upper')}] |"
    )
    lines.append("")

    # ── Split principal ──────────────────────────────────────────────────
    lines.append("## Split Principal (train≤2023, test=2025)")
    lines.append("")

    if "error" in edge_result and edge_result.get("n_trades", 1) == 0:
        lines.append(f"**Erreur** : {edge_result.get('error', 'N/A')}")
        lines.append("")
    else:
        bs = edge_result.get("bootstrap_sharpe", {})
        lines.append("| Métrique | Valeur |")
        lines.append("|---|---|")
        lines.append(f"| N trades | {edge_result.get('n_trades', 'N/A')} |")
        lines.append(
            f"| Sharpe observé | "
            f"{_fmt_val(bs.get('observed', float('nan')))} |"
        )
        lines.append(
            f"| p(Sharpe>0) bootstrap | "
            f"{_fmt_val(bs.get('p_value_gt_0', float('nan')))} |"
        )
        lines.append(f"| PSR (Bailey) | {_e('psr_bailey')} |")

        cpcv_dsr = edge_result.get("cpcv_dsr")
        if isinstance(cpcv_dsr, dict):
            lines.append(
                f"| DSR distributionnel | "
                f"{_fmt_val(cpcv_dsr.get('dsr', float('nan')))} |"
            )

        br = edge_result.get("breakeven", {})
        lines.append(
            f"| Breakeven WR | "
            f"{_fmt_pct(br.get('wr_pct'))} |"
        )
        lines.append(
            f"| WR observé | "
            f"{_fmt_pct(br.get('observed_wr_pct'))} |"
        )
        ts = edge_result.get("t_statistic", {})
        lines.append(
            f"| t-statistique | "
            f"{_fmt_val(ts.get('t_stat'))} "
            f"(p={_fmt_val(ts.get('p_value'))}) |"
        )
        lines.append("")

        # Métriques de performance annuelle
        if 2025 in principal_metrics:
            m = principal_metrics[2025]
            lines.append("### Performance 2025")
            lines.append("")
            lines.append("| Métrique | Valeur |")
            lines.append("|---|---|")
            lines.append(f"| Sharpe | {_fmt_val(m.get('sharpe'))} |")
            lines.append(
                f"| Return total (%) | {_fmt_val(m.get('total_return_pct'))} |"
            )
            lines.append(
                f"| Max drawdown (%) | {_fmt_val(m.get('max_dd_pct'))} |"
            )
            lines.append(f"| Win rate (%) | {_fmt_val(m.get('win_rate'))} |")
            lines.append(f"| N trades | {m.get('trades', 'N/A')} |")
            lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────
    lines.append("## Verdict")
    lines.append("")

    dsr = dsr_result.get("dsr", float("nan"))
    pct_prof = cpcv_agg.get("pct_profitable", 0.0)
    mean_sr = cpcv_agg.get("sharpe", {}).get("mean", float("nan"))

    dsr_ok = not np.isnan(dsr) and dsr > 0
    profitable_ok = not np.isnan(pct_prof) and pct_prof > 60.0
    sharpe_ok = not np.isnan(mean_sr) and mean_sr > 0

    lines.append("| Critère | Seuil | Valeur | OK? |")
    lines.append("|---|---|---|---|")
    dsr_str = f"{dsr:.4f}" if not np.isnan(dsr) else "NaN"
    lines.append(
        f"| DSR > 0 | > 0 | {dsr_str} | {'✅' if dsr_ok else '❌'} |"
    )
    pct_str = f"{pct_prof:.1f}%" if not np.isnan(pct_prof) else "NaN"
    lines.append(
        f"| % profitables > 60% | > 60% | {pct_str} | "
        f"{'✅' if profitable_ok else '❌'} |"
    )
    sr_str = f"{mean_sr:.4f}" if not np.isnan(mean_sr) else "NaN"
    lines.append(
        f"| E[Sharpe] > 0 | > 0 | {sr_str} | "
        f"{'✅' if sharpe_ok else '❌'} |"
    )
    lines.append("")

    go = dsr_ok and profitable_ok and sharpe_ok
    verdict = (
        "🟢 **GO — Phase 2-3**"
        if go
        else "🔴 **NO-GO — Améliorer l'edge avant de continuer**"
    )
    lines.append(f"### {verdict}")
    lines.append("")

    # ── Écrire ───────────────────────────────────────────────────────────
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _print_verdict(
    cpcv_agg: dict[str, Any],
    dsr_result: dict[str, Any],
) -> None:
    """Affiche le verdict GO/NO-GO dans les logs."""
    dsr = dsr_result.get("dsr", float("nan"))
    pct_prof = cpcv_agg.get("pct_profitable", 0.0)
    mean_sr = cpcv_agg.get("sharpe", {}).get("mean", float("nan"))

    dsr_ok = not np.isnan(dsr) and dsr > 0
    profitable_ok = not np.isnan(pct_prof) and pct_prof > 60.0
    sharpe_ok = not np.isnan(mean_sr) and mean_sr > 0
    go = dsr_ok and profitable_ok and sharpe_ok

    logger.info("=" * 70)
    if go:
        logger.info("🟢 VERDICT: GO — L'edge passe les seuils CPCV.")
    else:
        logger.info("🔴 VERDICT: NO-GO — L'edge ne passe pas les seuils.")
    logger.info(
        "DSR=%.4f (>0=%s), %%profitables=%.1f%% (>60%%=%s), E[SR]=%.4f (>0=%s)",
        dsr if not np.isnan(dsr) else -999.0,
        dsr_ok,
        pct_prof if not np.isnan(pct_prof) else -1.0,
        profitable_ok,
        mean_sr if not np.isnan(mean_sr) else -999.0,
        sharpe_ok,
    )
    logger.info("=" * 70)


def _fmt_pct(val: Any) -> str:
    """Formate un pourcentage."""
    if val is None:
        return "N/A"
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(fv):
        return "NaN"
    if abs(fv) < 10:
        return f"{fv * 100:.1f}%"
    return f"{fv:.1f}%"


def _fmt_val(val: Any) -> str:
    """Formate une valeur numérique."""
    if val is None:
        return "N/A"
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(fv):
        return "NaN"
    return f"{fv:.4f}"


if __name__ == "__main__":
    main()
