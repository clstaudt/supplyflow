"""Test fixtures."""

import pytest

from datetime import timedelta, datetime, date

from supplyflow.core import (
    DeterministicSupplyChain,
    LeadTimeSource,
    NormalDemand,
    NormalLeadTime,
    Product,
    ConstantDemand,
    StochasticDemandSource,
    StochasticSupplyChain,
    Warehouse,
    Enterprise,
)

from supplyflow.policy import (
    QPolicy,
    PQPolicy,
    RSPolicy,
)


@pytest.fixture
def demo_deterministic_supply_chain():
    supply_chain = DeterministicSupplyChain(
        product=Product(
            name="Spam",
            unit_cost=42,
            holding_cost=1,
        ),
        warehouse=Warehouse(
            _id=1,
            name="Demo Warehouse",
        ),
        demand_source=ConstantDemand(42),
        forecast_source=ConstantDemand(43),
        level=10,
    )
    supply_chain.set_policy(
        PQPolicy(quantity=10, reorder_point=2),
    )
    return supply_chain


@pytest.fixture
def demo_stochastic_supply_chain():
    supply_chain = StochasticSupplyChain(
        product=Product(
            name="Spam",
            unit_cost=42,
            holding_cost=1,
        ),
        warehouse=Warehouse(
            _id=1,
            name="Demo Warehouse",
        ),
        demand_source=NormalDemand(mu=10, sigma=4),
        forecast_source=NormalDemand(mu=9, sigma=3),
        lead_time_source=NormalLeadTime(mu=4, sigma=1),
        level=20,
    )
    supply_chain.set_policy(
        PQPolicy(quantity=10, reorder_point=2),
    )
    return supply_chain


@pytest.fixture
def demo_article_forecast():
    raise NotImplementedError(" #TODO: CS ")


@pytest.fixture
def demo_article_sales():
    raise NotImplementedError(" #TODO: CS ")


@pytest.fixture
def demo_enterprise():
    enterprise = Enterprise(
        name="ACME",
        start_time=datetime.now(),
    )
    n_products = 42
    n_warehouses = 2
    products = [
        Product(
            name="Spam",
            unit_cost=42,
            holding_cost=1,
        )
        for i in range(n_products)
    ]
    warehouses = [
        Warehouse(
            _id=i,
            name=f"Warehouse {i}",
        )
        for i in range(n_warehouses)
    ]
    for warehouse in warehouses:
        for product in products:
            enterprise.add(
                StochasticSupplyChain(
                    product=product,
                    warehouse=warehouse,
                    demand_source=NormalDemand(mu=10, sigma=4),
                    lead_time_source=NormalLeadTime(mu=4, sigma=1),
                    level=20,
                    freq=timedelta(days=1),
                ).set_policy(
                    RSPolicy(
                        review_period=7,
                        up_to_level=42,
                    )
                )
            )
    return enterprise
