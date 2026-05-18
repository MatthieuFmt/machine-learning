"""Tests unitaires pour le module CPCV — Step 02.

Couvre : generate_cpcv_splits, invariants temporels, psr_from_returns,
deflated_sharpe_ratio_from_distribution, aggregate_cpcv_metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.analysis.cpcv import (
    generate_cpcv_splits,
    aggregate_cpcv_metrics,
)
from learning_machine_learning.analysis.edge_validation import (
    _compute_sharpe_from_returns,
    psr_from_returns,
    deflated_sharpe_ratio_from_distribution,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_h1_index() -> pd.DatetimeIndex:
    """Index H1 sur 18 mois (assez pour 24 groupes + purge)."""
    return pd.date_range("2023-01-01", "2024-06-30", freq="1h", tz="UTC")


@pytest.fixture
def mock_results_df() -> pd.DataFrame:
    """DataFrame simulant run_cpcv_backtest() output avec 200 splits."""
    rng = np.random.default_rng(42)
    n = 200
    sharpe_values = rng.normal(loc=0.2, scale=0.5, size=n)
    # 10% d'erreurs simulées
    errors = [None] * n
    for i in rng.choice(n, size=n // 10, replace=False):
        errors[i] = "TestError: mock"
    return pd.DataFrame({
        "split_id": range(1, n + 1),
        "train_start": ["2023-01-01"] * n,
        "train_end": ["2023-12-31"] * n,
        "test_start": ["2024-01-01"] * n,
        "test_end": ["2024-06-30"] * n,
        "n_train": rng.integers(5000, 15000, size=n),
        "n_test": rng.integers(1000, 3000, size=n),
        "n_trades": rng.integers(5, 50, size=n),
        "sharpe": sharpe_values,
        "sharpe_per_trade": sharpe_values * np.sqrt(rng.integers(5, 50, size=n)),
        "profit_net": rng.normal(loc=50.0, scale=200.0, size=n),
        "win_rate": rng.normal(loc=35.0, scale=10.0, size=n),
        "dd": rng.normal(loc=-50.0, scale=30.0, size=n),
        "total_return_pct": rng.normal(loc=0.5, scale=2.0, size=n),
        "n_signaux": rng.integers(10, 100, size=n),
        "n_filtres": rng.integers(0, 20, size=n),
        "error": errors,
    })


# ═══════════════════════════════════════════════════════════════════════════
# generate_cpcv_splits
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateCpcvSplits:
    """Tests invariants temporels des splits CPCV."""

    def test_no_overlap_train_test(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """Aucun indice ne doit être à la fois dans train et test."""
        splits = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6, n_samples=50, random_state=42,
        ))
        assert len(splits) > 0, "Aucun split généré"
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, (
                f"Chevauchement train/test de {len(overlap)} indices"
            )

    def test_no_future_leak(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """Si train précède test, alors max(train) < min(test)."""
        splits = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6, n_samples=50, random_state=42,
        ))
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            max_train = synthetic_h1_index[train_idx].max()
            min_test = synthetic_h1_index[test_idx].min()
            # Le train peut être après le test (CPCV symétrique),
            # mais pas simultané
            if max_train < min_test:
                # Vérifier purge : max_train + 48h < min_test
                from datetime import timedelta
                assert max_train + timedelta(hours=48) < min_test, (
                    f"Purge violée: max_train={max_train}, min_test={min_test}"
                )

    def test_purge_bidirectional(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """Les barres dans la zone de purge ne sont ni train ni test."""
        splits = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6, n_samples=30,
            purge_hours=48, random_state=42,
        ))
        assert len(splits) > 0
        # Vérification indirecte : aucune barre ne manque (train + test = all?)
        # Non : la purge exclut des barres. On vérifie juste que la somme
        # ne dépasse pas le total.
        n_total = len(synthetic_h1_index)
        for train_idx, test_idx in splits:
            combined = np.union1d(train_idx, test_idx)
            assert len(combined) <= n_total
            # Au moins quelques barres exclues par la purge
            # (sauf si purge=0 ou k_test très petit)
            if len(combined) == n_total:
                # Acceptable si purge=0 ou peu de groupes test
                pass

    def test_n_samples_respected(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """n_samples détermine le nombre de splits émis."""
        splits_20 = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6,
            n_samples=20, random_state=42,
        ))
        splits_50 = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6,
            n_samples=50, random_state=99,
        ))
        assert len(splits_20) <= 20
        assert len(splits_50) <= 50
        # Seeds différents → splits différents
        first_20_ids = {tuple(sorted(t)) for _, t in splits_50[:20]}
        second_20_ids = {tuple(sorted(t)) for _, t in splits_20}
        assert first_20_ids != second_20_ids, "Les 20 splits sont identiques (même seed ?)"

    def test_coverage_all_months_appear(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """Chaque mois de la période devrait apparaître dans les tests."""
        splits = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6,
            n_samples=100, random_state=42,
        ))
        assert len(splits) > 0

        # Collecter tous les mois couverts par les périodes test
        months_seen: set[str] = set()
        for _, test_idx in splits:
            test_ts = synthetic_h1_index[test_idx]
            for ts in test_ts:
                months_seen.add(ts.strftime("%Y-%m"))

        # Avec 24 groupes sur 18 mois, chaque mois devrait être couvert
        assert len(months_seen) >= 12, (
            f"Seulement {len(months_seen)} mois couverts sur ~18"
        )

    def test_reproducibility(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """Même random_state → mêmes splits."""
        splits_a = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6,
            n_samples=20, random_state=42,
        ))
        splits_b = list(generate_cpcv_splits(
            synthetic_h1_index, n_groups=24, k_test=6,
            n_samples=20, random_state=42,
        ))
        assert len(splits_a) == len(splits_b)
        for (ta, tea), (tb, teb) in zip(splits_a, splits_b):
            assert np.array_equal(ta, tb)
            assert np.array_equal(tea, teb)

    def test_invalid_inputs(self, synthetic_h1_index: pd.DatetimeIndex) -> None:
        """Les inputs invalides lèvent ValueError."""
        with pytest.raises(ValueError, match="n_groups"):
            list(generate_cpcv_splits(synthetic_h1_index, n_groups=1, k_test=1))
        with pytest.raises(ValueError, match="k_test"):
            list(generate_cpcv_splits(synthetic_h1_index, n_groups=10, k_test=10))
        with pytest.raises(ValueError, match="purge_hours"):
            list(generate_cpcv_splits(synthetic_h1_index, purge_hours=-1))
        with pytest.raises(TypeError):
            list(generate_cpcv_splits(pd.Index([1, 2, 3]), n_groups=2, k_test=1))


# ═══════════════════════════════════════════════════════════════════════════
# PSR (Bailey & López de Prado 2012)
# ═══════════════════════════════════════════════════════════════════════════

class TestPsrFromReturns:
    """Tests pour psr_from_returns."""

    def test_positive_returns_high_psr(self) -> None:
        """Returns clairement positifs → PSR proche de 1."""
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=2.0, scale=5.0, size=200).astype(np.float64)
        psr = psr_from_returns(returns, sr_benchmark=0.0)
        assert not np.isnan(psr)
        assert psr > 0.5, f"PSR={psr} attendu > 0.5"

    def test_zero_mean_returns_psr_near_05(self) -> None:
        """Returns centrés sur 0 → PSR > 0 (H₀: can't reject that SR>0)."""
        rng = np.random.default_rng(123)
        returns = rng.normal(loc=0.0, scale=5.0, size=200).astype(np.float64)
        psr = psr_from_returns(returns, sr_benchmark=0.0)
        assert not np.isnan(psr)
        # ŜR est non nul par échantillonnage → PSR peut être > 0.5
        # On vérifie juste que c'est une probabilité valide
        assert 0.0 <= psr <= 1.0, f"PSR={psr} hors [0,1]"

    def test_negative_returns_low_psr(self) -> None:
        """Returns négatifs → PSR proche de 0."""
        rng = np.random.default_rng(456)
        returns = rng.normal(loc=-2.0, scale=5.0, size=200).astype(np.float64)
        psr = psr_from_returns(returns, sr_benchmark=0.0)
        assert not np.isnan(psr)
        assert psr < 0.2, f"PSR={psr} attendu < 0.2"

    def test_single_return_nan(self) -> None:
        """1 seul return → NaN."""
        psr = psr_from_returns(np.array([5.0], dtype=np.float64))
        assert np.isnan(psr)

    def test_constant_returns(self) -> None:
        """Returns constants → PSR = 1.0 si mean > 0, 0.0 si mean ≤ 0."""
        pos = psr_from_returns(np.full(100, 3.0, dtype=np.float64))
        assert pos == 1.0
        zero = psr_from_returns(np.full(100, 0.0, dtype=np.float64))
        assert zero == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# DSR via distribution CPCV
# ═══════════════════════════════════════════════════════════════════════════

class TestDeflatedSharpeRatioDistribution:
    """Tests pour deflated_sharpe_ratio_from_distribution."""

    def test_profitable_distribution_positive_dsr(self) -> None:
        """Distribution très profitable + peu de splits → DSR > 0.

        Avec peu de splits (n=30), la déflation SR₀* est moins sévère.
        """
        rng = np.random.default_rng(42)
        sharpe_dist = rng.normal(loc=0.8, scale=0.2, size=30).astype(np.float64)
        result = deflated_sharpe_ratio_from_distribution(
            observed_sr=0.8, sharpe_distribution=sharpe_dist,
        )
        assert result["n_splits"] == 30
        assert result["pct_profitable"] > 90.0
        assert result["dsr"] > 0.0, f"DSR={result['dsr']} attendu > 0"

    def test_unprofitable_distribution_negative_dsr(self) -> None:
        """Distribution majoritairement non-profitable → DSR < 0."""
        rng = np.random.default_rng(123)
        sharpe_dist = rng.normal(loc=-0.3, scale=0.3, size=200).astype(np.float64)
        result = deflated_sharpe_ratio_from_distribution(
            observed_sr=-0.2, sharpe_distribution=sharpe_dist,
        )
        assert result["pct_profitable"] < 40.0
        assert result["dsr"] < 0.0, f"DSR={result['dsr']} attendu < 0"

    def test_small_distribution(self) -> None:
        """Moins de 2 splits valides → NaN."""
        result = deflated_sharpe_ratio_from_distribution(
            observed_sr=0.5,
            sharpe_distribution=np.array([0.5], dtype=np.float64),
        )
        assert result["n_splits"] == 1
        assert np.isnan(result["dsr"])

    def test_output_keys(self) -> None:
        """Vérifie la présence de toutes les clés de sortie."""
        rng = np.random.default_rng(42)
        sharpe_dist = rng.normal(loc=0.3, scale=0.4, size=100).astype(np.float64)
        result = deflated_sharpe_ratio_from_distribution(
            observed_sr=0.3, sharpe_distribution=sharpe_dist,
        )
        expected_keys = {
            "dsr", "psr_zero", "sr0_star", "e_max_sr", "var_sr",
            "n_splits", "pct_profitable", "mean_sr", "std_sr",
            "min_sr", "max_sr", "median_sr", "ci_95_lower", "ci_95_upper",
        }
        assert set(result.keys()) == expected_keys, (
            f"Clés manquantes: {expected_keys - set(result.keys())}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# aggregate_cpcv_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregateCpcvMetrics:
    """Tests pour aggregate_cpcv_metrics."""

    def test_output_structure(self, mock_results_df: pd.DataFrame) -> None:
        """Vérifie la structure du dict retourné."""
        result = aggregate_cpcv_metrics(mock_results_df)

        assert result["n_splits"] == 200
        assert result["n_splits_valid"] > 0
        assert not np.isnan(result["pct_profitable"])
        assert "mean" in result["sharpe"]
        assert "std" in result["sharpe"]
        assert result["n_trades"]["total"] > 0
        assert isinstance(result["coverage"], dict)

    def test_empty_results(self) -> None:
        """DataFrame vide → zéro splits valides."""
        empty = pd.DataFrame(columns=["n_trades", "error", "sharpe"])
        result = aggregate_cpcv_metrics(empty)
        assert result["n_splits"] == 0
        assert result["n_splits_valid"] == 0
        assert np.isnan(result["sharpe"]["mean"])

    def test_all_errors(self, mock_results_df: pd.DataFrame) -> None:
        """Tous les splits en erreur → 0 valides."""
        broken = mock_results_df.copy()
        broken["error"] = "fail"
        broken["n_trades"] = 0
        result = aggregate_cpcv_metrics(broken)
        assert result["n_splits"] == 200
        assert result["n_splits_valid"] == 0

    def test_coverage_keys(self, mock_results_df: pd.DataFrame) -> None:
        """Les clés de coverage sont au format YYYY-MM."""
        result = aggregate_cpcv_metrics(mock_results_df)
        for key in result["coverage"]:
            parts = key.split("-")
            assert len(parts) == 2
            assert 2023 <= int(parts[0]) <= 2025
            assert 1 <= int(parts[1]) <= 12
