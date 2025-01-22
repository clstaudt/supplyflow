"""Supply chain metrics."""

from datetime import timedelta
from pandera.typing import DataFrame

from supplyflow.core import SupplyChain


def demand_coverage(
    order,
    demand,
):
    """The demand coverage is defined as the fraction of expected demand over
    lead time that an order covers.

    Args:
        order (_type_): _description_
        demand (_type_): _description_
    """
    raise NotImplementedError()



def period_service_level(
    stock_history: DataFrame,
    period: timedelta,
) -> float:
    """The empirical probability that no out-of-stock situation
    occurs in the given period."""
    stock_history[["out_of_stock"]] = (
        stock_history[["stock"]]
        .apply(lambda x: x == 0)
    )
    period_service_level = 1 - (
        stock_history.resample(period)["out_of_stock"].any()
    ).mean()
    return period_service_level