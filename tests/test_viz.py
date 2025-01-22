from supplyflow.viz import supply_chain_plot, supply_chain_plot


def test_supply_chain_plot(demo_deterministic_supply_chain):
    demo_deterministic_supply_chain.run(iterations=42)
    supply_chain_plot(demo_deterministic_supply_chain)
