from supplyflow.formulary import optimal_order_up_to_levels, normal_distribution


def test_optimal_stock_levels():
    cycle_stock, safety_stock = optimal_order_up_to_levels(
        demand_distribution=normal_distribution(mu=100, sigma=25),
        cycle_service_level_target=0.95,
    )
    assert cycle_stock == 100
    assert safety_stock == 42
