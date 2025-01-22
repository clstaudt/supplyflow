"""Formulary: A collection of formulas - a place for math and stats which does not depend on supplyflow.core"""


from typing import Tuple
from scipy import stats
import numpy
import math


def normal_distribution(
    mu,
    sigma,
):
    return stats.norm(loc=mu, scale=sigma)


def optimal_order_up_to_levels(
    demand_distribution,
    cycle_service_level_target: float,
) -> Tuple[float, float]:
    """
    For a given demand distribution and service level target, calculate the required level stock level as cycle stock and safety stock.
    """
    try:
        assert 0.0 <= cycle_service_level_target <= 1
    except AssertionError:
        raise ValueError(
            f"service level target must be in [0,1] - got {cycle_service_level_target} instead"
        )

    mu_d, sigma_d = demand_distribution.mean(), demand_distribution.std()
    # calculate cycle stock
    cycle_stock = math.ceil(mu_d)

    # caclculate safety stock
    alpha = cycle_service_level_target
    z_alpha = stats.norm.ppf(alpha)
    # safety stock = service level factor $z_\alpha$ * demand deviation $\sigma_d$
    safety_stock = math.ceil(z_alpha * sigma_d)

    return cycle_stock, safety_stock
