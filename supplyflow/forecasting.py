"""Functionality related to forecasting using the supply chain models."""

from pandera.typing import DataFrame
import copy
from datetime import timedelta

from .core import SupplyChain


def forecast_stock_levels(
    supply_chain: SupplyChain,
    horizon: timedelta,
) -> DataFrame:
    """Forecast the stock levels for a given horizon."""
    if supply_chain.forecast_source is None:
        raise ValueError(
            "To forecast the stock levels, the SupplyChain needs to have a forecast source"
        )

    if supply_chain.policy is None:
        raise ValueError(
            "To forecast the stock levels, the SupplyChain needs to have a policy"
        )

    supply_chain_ = copy.deepcopy(supply_chain)
    # treat forecast as actual demand
    supply_chain_.demand_source = supply_chain_.forecast_source
    h = horizon // supply_chain_.freq
    supply_chain_.run(iterations=h)

    data = supply_chain_.history_data
    stock_level_forecast = data["level"].iloc[-h:].to_frame()
    return stock_level_forecast
