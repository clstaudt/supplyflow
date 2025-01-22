from datetime import timedelta

from supplyflow.forecasting import forecast_stock_levels


def test_foreacst_stock_levels(demo_stochastic_supply_chain):
    horizon = timedelta(days=30)
    stock_level_forecast = forecast_stock_levels(
        supply_chain=demo_stochastic_supply_chain,
        horizon=horizon,
    )
    # forecast is as long as the horizon
    assert stock_level_forecast.shape[0] == (
        horizon // demo_stochastic_supply_chain.freq
    )
