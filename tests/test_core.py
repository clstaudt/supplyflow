# -*- coding: utf-8 -*-
from datetime import timedelta
import pytest

from supplyflow.core import (
    SupplyChain,
    DeterministicSupplyChain,
)
from supplyflow.policy import (
    construct_policy,
    PQPolicy,
    QPolicy,
    RPQPolicy,
    RSPolicy,
)


def test_construct_policy(demo_deterministic_supply_chain):

    supply_chain = demo_deterministic_supply_chain

    # QPolicy
    policy = construct_policy(
        quantity=42,
    )
    assert isinstance(policy, QPolicy)

    # PQPolicy
    policy = construct_policy(
        quantity=42,
        reorder_point=10,
    )
    assert isinstance(policy, PQPolicy)

    # RSPolicy
    policy = construct_policy(
        review_period=7,
        up_to_level=42,
    )
    assert isinstance(policy, RSPolicy)

    # RPQPolicy
    policy = construct_policy(
        review_period=7,
        reorder_point=10,
        quantity=42,
    )
    assert isinstance(policy, RPQPolicy)


def test_optimize_QPolicy_deterministic(demo_deterministic_supply_chain):
    policy = QPolicy.optimize(supply_chain=demo_deterministic_supply_chain)


def test_optimize_PQPolicy_deterministic(demo_deterministic_supply_chain):
    policy = PQPolicy.optimize(supply_chain=demo_deterministic_supply_chain)


def test_optimize_RSPolicy_deterministic(demo_deterministic_supply_chain):
    policy = RSPolicy.optimize(supply_chain=demo_deterministic_supply_chain)


def test_optimize_RPQolicy_deterministic(demo_deterministic_supply_chain):
    policy = RPQPolicy.optimize(supply_chain=demo_deterministic_supply_chain)


def test_optimize_QPolicy_stochastic(demo_stochastic_supply_chain):
    policy = QPolicy.optimize(supply_chain=demo_stochastic_supply_chain)


def test_optimize_PQPolicy_stochastic(demo_stochastic_supply_chain):
    policy = PQPolicy.optimize(supply_chain=demo_stochastic_supply_chain)


def test_optimize_RSPolicy_stochastic(demo_stochastic_supply_chain):
    policy = RSPolicy.optimize(supply_chain=demo_stochastic_supply_chain)


def test_optimize_RPQolicy_stochastic(demo_stochastic_supply_chain):
    policy = RPQPolicy.optimize(supply_chain=demo_stochastic_supply_chain)


def test_run_deterministic_supply_chain(demo_deterministic_supply_chain):
    """Test whether deterministic supply chain runs."""
    demo_deterministic_supply_chain.run(iterations=42)
    assert len(demo_deterministic_supply_chain.history) > 0


def test_run_stochastic_supply_chain(demo_stochastic_supply_chain):
    """Test whether stochastic supply chain runs."""
    demo_stochastic_supply_chain.run(iterations=42)
    assert len(demo_stochastic_supply_chain.history) > 0


def test_metric_fill_rate(demo_deterministic_supply_chain):
    demo_deterministic_supply_chain.run(iterations=42)
    fill_rate = demo_deterministic_supply_chain.fill_rate()
    assert 0 <= fill_rate <= 1


def test_metric_cycle_service_level(demo_deterministic_supply_chain):
    demo_deterministic_supply_chain.run(iterations=42)
    cycle_service_level = demo_deterministic_supply_chain.cycle_service_level()
    assert 0 <= cycle_service_level <= 1


def test_metric_period_service_level(demo_deterministic_supply_chain):
    demo_deterministic_supply_chain.run(iterations=42)
    period_service_level = demo_deterministic_supply_chain.period_service_level(
        period=timedelta(days=7)
    )
    assert 0 <= period_service_level <= 1


def test_expected_cycle_service_level(demo_stochastic_supply_chain):
    demo_stochastic_supply_chain.run(iterations=42)
    expected_service_level = demo_stochastic_supply_chain.expected_cycle_service_level()
    assert 0 <= expected_service_level <= 1


def test_enterprise_run(demo_enterprise):
    demo_enterprise.run(timespan=timedelta(days=365))


def test_enterprise_current_out_of_stock_rate(demo_enterprise):
    demo_enterprise.run(timespan=timedelta(days=365))
    out_of_stock_rate = demo_enterprise.current_out_of_stock_rate
    assert 0 <= out_of_stock_rate <= 1
