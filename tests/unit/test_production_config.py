import pytest
from app.config.models import Sleeve, ProductionConfig


def test_sleeve_ok():
    Sleeve(asset="US30", tf="D1", strategy_name="donchian", strategy_params={"N": 20, "M": 20})


def test_sleeve_bad_threshold():
    with pytest.raises(ValueError):
        Sleeve(asset="US30", tf="D1", strategy_name="d", strategy_params={}, meta_threshold=0.9)


def test_config_validation():
    s = Sleeve(asset="US30", tf="D1", strategy_name="d", strategy_params={"N": 20})
    with pytest.raises(ValueError, match="target_vol_annual"):
        ProductionConfig(
            version="v3.0", sleeves=(s,), portfolio_weighting="equal_risk",
            vol_targeting=True, target_vol_annual=2.0, leverage_cap=2.0, retrain_months=6,
        )
