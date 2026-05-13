"""Tests unitaires pour les features de session (Step 04).

Couvre : compute_session_id, compute_session_open_range,
compute_relative_position_in_session, SessionVolatilityScaler,
et l'intégration dans build_ml_ready().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.regime import (
    compute_session_id,
    compute_session_open_range,
    compute_relative_position_in_session,
    SessionVolatilityScaler,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def hourly_index_48h() -> pd.DatetimeIndex:
    """Index UTC sur 48h — couvre toutes les sessions 2x."""
    return pd.date_range(
        "2024-01-15 00:00",
        periods=48,
        freq="h",
        tz="UTC",
    )


@pytest.fixture
def session_id_48h(hourly_index_48h: pd.DatetimeIndex) -> pd.Series:
    """Session IDs pour un index 48h."""
    return compute_session_id(hourly_index_48h)


# ── compute_session_id ────────────────────────────────────────────────

class TestComputeSessionId:
    """compute_session_id — mapping heure UTC → session_id."""

    def test_returns_series_with_same_index(
        self, hourly_index_48h: pd.DatetimeIndex,
    ) -> None:
        sid = compute_session_id(hourly_index_48h)
        assert isinstance(sid, pd.Series)
        assert len(sid) == len(hourly_index_48h)
        assert sid.index.equals(hourly_index_48h)

    def test_dtype_int8(self, hourly_index_48h: pd.DatetimeIndex) -> None:
        sid = compute_session_id(hourly_index_48h)
        assert sid.dtype == np.int8

    @pytest.mark.parametrize(
        "hour,expected",
        [
            (0, 4),   # Low liq
            (1, 0),   # Tokyo
            (6, 0),   # Tokyo
            (7, 1),   # London
            (11, 1),  # London
            (12, 3),  # Overlap
            (15, 3),  # Overlap
            (16, 2),  # NY
            (20, 2),  # NY
            (21, 0),  # Tokyo (gap)
            (22, 4),  # Low liq
            (23, 4),  # Low liq
        ],
    )
    def test_individual_hours(
        self, hour: int, expected: int,
    ) -> None:
        idx = pd.date_range(
            f"2024-01-15 {hour:02d}:00",
            periods=1,
            freq="h",
            tz="UTC",
        )
        sid = compute_session_id(idx)
        assert sid.iloc[0] == expected, (
            f"Heure {hour:02d}:00 → attendu {expected}, reçu {sid.iloc[0]}"
        )

    def test_overlap_beats_london(self) -> None:
        """12:00 doit être 3 (Overlap), pas 1 (London)."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 12:00", tz="UTC"),
        ])
        sid = compute_session_id(idx)
        assert sid.iloc[0] == 3

    def test_overlap_beats_ny(self) -> None:
        """15:00 doit être 3 (Overlap), pas 2 (NY)."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 15:00", tz="UTC"),
        ])
        sid = compute_session_id(idx)
        assert sid.iloc[0] == 3

    def test_low_liq_beats_tokyo(self) -> None:
        """23:00 doit être 4 (Low liq), pas 0 (Tokyo)."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 23:00", tz="UTC"),
        ])
        sid = compute_session_id(idx)
        assert sid.iloc[0] == 4

    def test_all_five_sessions_present(self, hourly_index_48h: pd.DatetimeIndex) -> None:
        sid = compute_session_id(hourly_index_48h)
        unique = set(sid.unique())
        assert unique == {0, 1, 2, 3, 4}, (
            f"Sessions manquantes: {unique}"
        )


# ── compute_session_open_range ────────────────────────────────────────

class TestComputeSessionOpenRange:
    """compute_session_open_range — range cumulé depuis ouverture de session."""

    def test_returns_series(self, hourly_index_48h: pd.DatetimeIndex) -> None:
        n = len(hourly_index_48h)
        rng = np.random.default_rng(42)
        high = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum(), index=hourly_index_48h)
        low = high - 0.0005
        sid = compute_session_id(hourly_index_48h)
        sor = compute_session_open_range(high, low, sid)
        assert isinstance(sor, pd.Series)
        assert len(sor) == n

    def test_never_negative(self, hourly_index_48h: pd.DatetimeIndex) -> None:
        n = len(hourly_index_48h)
        rng = np.random.default_rng(43)
        high = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum(), index=hourly_index_48h)
        low = high - rng.uniform(0.0002, 0.001, n)
        sid = compute_session_id(hourly_index_48h)
        sor = compute_session_open_range(high, low, sid)
        assert (sor >= 0).all()

    def test_resets_on_session_change(
        self, hourly_index_48h: pd.DatetimeIndex,
    ) -> None:
        """Le range doit reset à chaque changement de session.

        Vérifie qu'après un changement de session, le range repart
        du range de la première barre (high-low local), qui doit être
        inférieur au max cumulé de la session précédente.
        """
        n = len(hourly_index_48h)
        rng = np.random.default_rng(44)
        base = 1.1000 + rng.normal(0, 0.001, n).cumsum()
        # High et Low réalistes : Low toujours < High
        spread = rng.uniform(0.0002, 0.001, n)
        high = pd.Series(base + spread / 2, index=hourly_index_48h)
        low = pd.Series(base - spread / 2, index=hourly_index_48h)
        sid = compute_session_id(hourly_index_48h)
        sor = compute_session_open_range(high, low, sid)

        # Identifier les changements de session
        changes = sid.diff().fillna(1).ne(0)
        change_idx = changes[changes].index

        # Vérifier qu'à chaque reset, le range est ≤ range de la 1ère barre
        # (donc nettement plus petit que le max cumulé de la session précédente)
        for i in range(1, len(change_idx)):
            before_segment = sor.loc[change_idx[i - 1]:change_idx[i - 1] + pd.Timedelta(hours=1)]
            first_bar_range = (high.loc[change_idx[i]] - low.loc[change_idx[i]])
            new_range = sor.loc[change_idx[i]]
            # Le range de la 1ère barre doit être égal au high-low local
            assert new_range == pytest.approx(first_bar_range, rel=1e-9)


# ── compute_relative_position_in_session ──────────────────────────────

class TestComputeRelativePositionInSession:
    """compute_relative_position_in_session — progression ∈ [0,1]."""

    def test_bounds_zero_one(self, hourly_index_48h: pd.DatetimeIndex) -> None:
        sid = compute_session_id(hourly_index_48h)
        rpos = compute_relative_position_in_session(hourly_index_48h, sid)
        assert (rpos >= 0.0).all(), f"min={rpos.min()}"
        assert (rpos <= 1.0).all(), f"max={rpos.max()}"

    def test_starts_near_zero(self) -> None:
        """Début de session London (7h) doit être proche de 0."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 07:00", tz="UTC"),
        ])
        sid = compute_session_id(idx)
        rpos = compute_relative_position_in_session(idx, sid)
        assert rpos.iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_ends_near_one_london(self) -> None:
        """Fin de session London (11h, dernière barre avant Overlap) doit être proche de 1."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 11:00", tz="UTC"),
        ])
        sid = compute_session_id(idx)
        rpos = compute_relative_position_in_session(idx, sid)
        # London: 7h→11h = 4h écoulées sur 5h → 0.8
        assert rpos.iloc[0] == pytest.approx(4.0 / 5.0, rel=1e-9)

    def test_midnight_wrap_tokyo(self) -> None:
        """Tokyo 01:00 (start=23, wrap) → (1+24-23)/9 = 2/9."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 01:00", tz="UTC"),
        ])
        sid = compute_session_id(idx)
        rpos = compute_relative_position_in_session(idx, sid)
        assert rpos.iloc[0] == pytest.approx(2.0 / 9.0, rel=1e-9)

    def test_monotonic_within_session(
        self, hourly_index_48h: pd.DatetimeIndex,
    ) -> None:
        """Dans une session donnée, la position relative doit croître."""
        sid = compute_session_id(hourly_index_48h)
        rpos = compute_relative_position_in_session(hourly_index_48h, sid)

        # Trouver un segment London (sid=1) continu
        mask = sid == 1
        segments = mask.ne(mask.shift()).cumsum()[mask]
        # Prendre le premier segment
        first_seg = segments[segments == segments.iloc[0]]
        rpos_seg = rpos[first_seg.index]
        assert rpos_seg.is_monotonic_increasing


# ── SessionVolatilityScaler ───────────────────────────────────────────

class TestSessionVolatilityScaler:
    """SessionVolatilityScaler — standardisation par session, fit train-only."""

    def test_fit_transform_centers_by_session(self) -> None:
        """Les z-scores par session doivent être centrés (~0) après fit."""
        n = 200
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(45)
        sid = compute_session_id(idx)
        atr = pd.Series(
            0.002 + rng.normal(0, 0.0005, n),
            index=idx,
        )

        scaler = SessionVolatilityScaler()
        z = scaler.fit_transform(atr, sid)

        # Moyenne globale proche de 0
        assert z.mean() == pytest.approx(0.0, abs=0.2)

    def test_different_sessions_different_means(self) -> None:
        """Chaque session doit avoir une moyenne de z-score ≈ 0."""
        n = 500
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(46)
        sid = compute_session_id(idx)
        # ATR varie par session
        atr_means = {0: 0.001, 1: 0.003, 2: 0.004, 3: 0.005, 4: 0.0008}
        atr = pd.Series(0.0, index=idx)
        for s, mu in atr_means.items():
            mask = sid == s
            atr[mask] = mu + rng.normal(0, mu * 0.1, mask.sum())

        scaler = SessionVolatilityScaler()
        z = scaler.fit_transform(atr, sid)

        for s in [0, 1, 2, 3, 4]:
            z_s = z[sid == s]
            if len(z_s) > 10:
                assert z_s.mean() == pytest.approx(0.0, abs=0.3), (
                    f"Session {s}: mean z-score = {z_s.mean():.3f}"
                )

    def test_no_leak_fit_train_only(self) -> None:
        """Les stats du train ne changent pas après transform sur test modifié."""
        n = 200
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(47)

        # Train : ATR ~ 0.002, Test : ~ 0.010 (5x plus volatil)
        atr_train = pd.Series(rng.normal(0.002, 0.0005, 100), index=idx[:100])
        atr_test_normal = pd.Series(rng.normal(0.010, 0.002, 100), index=idx[100:])
        sid_train = compute_session_id(idx[:100])
        sid_test = compute_session_id(idx[100:])

        scaler = SessionVolatilityScaler()
        scaler.fit(atr_train, sid_train)

        # Capturer les stats après fit
        stats_after_fit = {k: v for k, v in scaler._stats.items()}

        # Transformer des données test (distribution différente)
        z1 = scaler.transform(atr_test_normal, sid_test)

        # Les stats internes ne doivent PAS avoir changé
        assert scaler._stats == stats_after_fit, (
            "Les stats du scaler ont changé après transform() — LEAK détecté!"
        )

        # Les z-scores doivent être ≠ 0 (car distribution différente du train)
        assert abs(z1.mean()) > 0.1, (
            f"Z-scores anormalement centrés: mean={z1.mean():.4f}"
        )

    def test_missing_session_fallback_global(self) -> None:
        """Session absente du train → fallback sur stats globales."""
        n = 60
        idx = pd.date_range("2023-01-15 07:00", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(48)
        # Seulement London (7h-11h) et Overlap (12h-15h)
        atr = pd.Series(rng.normal(0.003, 0.0005, n), index=idx)
        sid = compute_session_id(idx)
        # Tokyo, NY, LowLiq absents du dataset

        scaler = SessionVolatilityScaler()
        scaler.fit(atr, sid)
        z = scaler.transform(atr, sid)

        assert not z.isna().any(), "Fallback global a produit des NaN"
        # London et Overlap devraient être bien centrés
        for s in [1, 3]:
            mask = sid == s
            if mask.sum() > 5:
                assert abs(z[mask].mean()) < 0.5

    def test_zero_division_prevention(self) -> None:
        """ATR constant → sigma ≈ 0 → ne doit pas crash (1e-10 de protection)."""
        n = 50
        idx = pd.date_range("2023-01-15 07:00", periods=n, freq="h", tz="UTC")
        atr = pd.Series(np.full(n, 0.003), index=idx)
        sid = compute_session_id(idx)

        scaler = SessionVolatilityScaler()
        z = scaler.fit_transform(atr, sid)

        assert not z.isna().any()
        # Avec ATR constant, z-score doit être ~0 (pas d'infini)
        assert np.isfinite(z).all()
        assert abs(z.mean()) < 1e-6

    def test_fallback_default_values(self) -> None:
        """Scaler sans fit doit utiliser mu=0, sigma=1 par défaut."""
        n = 20
        idx = pd.date_range("2023-01-15 07:00", periods=n, freq="h", tz="UTC")
        atr = pd.Series(np.linspace(0.001, 0.005, n), index=idx)
        sid = compute_session_id(idx)

        scaler = SessionVolatilityScaler()
        # Pas de fit → mu=0, sigma=1
        z = scaler.transform(atr, sid)

        # Le z-score doit être proche de la valeur brute (mu=0, sigma=1)
        np.testing.assert_allclose(z.values, atr.values, rtol=0.01)


# ── Intégration pipeline ──────────────────────────────────────────────

class TestPipelineIntegration:
    """Vérifie que build_ml_ready inclut les colonnes session."""

    def test_build_ml_ready_adds_session_columns(self) -> None:
        """Les colonnes session doivent être présentes dans la sortie."""
        from learning_machine_learning.config.instruments import EurUsdConfig
        from learning_machine_learning.features.pipeline import build_ml_ready

        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(49)
        close = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum(), index=idx)
        h1 = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 0.0002,
            "Low": close - 0.0002,
            "Close": close,
            "Spread": 0.0002,
            "Volume": 1000,
            # merge_features exige Time timezone-naive
            "Time": idx.tz_localize(None),
        }, index=idx)

        instrument = EurUsdConfig()
        # Utiliser une feature_dropped vide pour garder toutes les colonnes
        instrument = instrument.__class__(
            features_dropped=(),
            target_mode="triple_barrier",
        )

        ml = build_ml_ready(
            instrument=instrument,
            data={"H1": h1},
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
        )

        assert "session_id" in ml.columns
        assert "ATR_session_zscore" in ml.columns
        assert "session_open_range" in ml.columns
        assert "relative_position_in_session" in ml.columns

    def test_one_hot_encoding_columns_present(self) -> None:
        """Avec encoding='one_hot', les 4 dummies doivent être présentes."""
        from learning_machine_learning.config.instruments import EurUsdConfig
        from learning_machine_learning.features.pipeline import build_ml_ready

        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(50)
        close = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum(), index=idx)
        h1 = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 0.0002,
            "Low": close - 0.0002,
            "Close": close,
            "Spread": 0.0002,
            "Volume": 500,
            "Time": idx.tz_localize(None),
        }, index=idx)

        instrument = EurUsdConfig(
            features_dropped=(),
            target_mode="triple_barrier",
            session_encoding="one_hot",
        )

        ml = build_ml_ready(
            instrument=instrument,
            data={"H1": h1},
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
        )

        for col in ["session_London", "session_NY", "session_Overlap", "session_LowLiq"]:
            assert col in ml.columns, f"Colonne one-hot manquante: {col}"

        # Vérifier que les dummies sont binaires
        for col in ["session_London", "session_NY", "session_Overlap", "session_LowLiq"]:
            valid = ml[col].dropna()
            assert valid.isin({0, 1}).all(), (
                f"{col} n'est pas binaire: {valid.unique()}"
            )

    def test_ordinal_encoding_no_dummies(self) -> None:
        """Avec encoding='ordinal', pas de dummies, seulement session_id."""
        from learning_machine_learning.config.instruments import EurUsdConfig
        from learning_machine_learning.features.pipeline import build_ml_ready

        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(51)
        close = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum(), index=idx)
        h1 = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 0.0002,
            "Low": close - 0.0002,
            "Close": close,
            "Spread": 0.0002,
            "Volume": 500,
            "Time": idx.tz_localize(None),
        }, index=idx)

        instrument = EurUsdConfig(
            features_dropped=(),
            target_mode="triple_barrier",
            session_encoding="ordinal",
        )

        ml = build_ml_ready(
            instrument=instrument,
            data={"H1": h1},
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
        )

        assert "session_id" in ml.columns
        assert "session_London" not in ml.columns
        assert "session_NY" not in ml.columns

    def test_train_end_separation(self) -> None:
        """train_end=None vs train_end='2023-12-31' donne des z-scores différents."""
        from learning_machine_learning.config.instruments import EurUsdConfig
        from learning_machine_learning.features.pipeline import build_ml_ready

        n = 500
        idx = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
        # train_end doit être dans la plage des données pour que le split ait un effet
        train_end_ts = idx[250]  # mi-parcours, ≈ 2023-06-11
        rng = np.random.default_rng(52)
        close = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum(), index=idx)
        h1 = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 0.0002,
            "Low": close - 0.0002,
            "Close": close,
            "Spread": 0.0002,
            "Volume": 500,
            "Time": idx.tz_localize(None),
        }, index=idx)

        instrument = EurUsdConfig(
            features_dropped=(),
            target_mode="triple_barrier",
        )

        ml_all = build_ml_ready(
            instrument=instrument,
            data={"H1": h1.copy()},
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            train_end=None,  # fit sur tout → look-ahead
        )
        ml_split = build_ml_ready(
            instrument=instrument,
            data={"H1": h1.copy()},
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            train_end=train_end_ts,  # fit train-only, pas de look-ahead
        )

        # build_ml_ready retourne un index tz-naive → conversion nécessaire
        train_end_ts_naive = train_end_ts.tz_localize(None)
        post_train_all = ml_all.index > train_end_ts_naive
        post_train_split = ml_split.index > train_end_ts_naive
        z_all = ml_all.loc[post_train_all, "ATR_session_zscore"]
        z_split = ml_split.loc[post_train_split, "ATR_session_zscore"]
        # Comparer sur l'intersection (après dropna, les DataFrames ont moins de lignes)
        common_idx = z_all.index.intersection(z_split.index)
        assert len(common_idx) > 10, "Pas assez d'observations post-train communes"
        assert not np.allclose(
            z_all[common_idx].values, z_split[common_idx].values, equal_nan=True
        ), (
            "train_end=None et train_end='2023-12-31' devraient donner des z-scores différents"
        )
