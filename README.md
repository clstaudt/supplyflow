# supplyflow

<div style="border: 1px solid orange; padding: 10px; margin-bottom: 10px;">
  <p style="margin: 0;">
   ⚠️  This is an incomplete and currently unmaintained implementation for educational purposes. It is not ready for production use, but should provide a good starting point for your own implementation, or ideally your contribution to continue development. 
  </p>
</div>

## Introduction

`supplyflow` is a Python library for the automated management and optimization of supply chains and inventories.

**an object model of supply chain concepts**

## Usage

```python
from supplyflow.model import Product, DeterministicSupplyChain

spam_supply_chain = DeterministicSupplyChain(
    product=Product(
        name="Spam",
        unit_cost=2,
    ),
    lead_time=5,
    transaction_fixed_cost=100,
)

eggs_supply_chain = DeterministicSupplyChain(
    product=Product(
        name="Eggs",
        unit_cost=1,
    ),
    lead_time=1,
    transaction_fixed_cost=50,
)

```



**implementations of inventory management policies**


```python
from supplyflow.policy import SQPolicy

my_policy = SQPolicy(
    reorder_point=5,
    quantity=42
)
```

**... and their optimization**


```python
best_policy = SQPolicy.optimize(
    supply_chain=spam_supply_chain
)
```

**supply chain performance metrics**

```python
 yearly_service_level = period_service_level(
    demand,
    stock,
    period="y",
)
```


## References

- [_Inventory Optimization: Models and Simulations_](https://www.researchgate.net/publication/343472734_Inventory_Optimization_Models_and_Simulations)