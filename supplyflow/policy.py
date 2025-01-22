from multiprocessing.sharedctypes import Value
from typing import Optional
import numpy

from scipy import stats

from supplyflow.formulary import optimal_order_up_to_levels

from .core import (
    NormalDemand,
    NormalLeadTime,
    StochasticSupplyChain,
    SupplyChain,
    DeterministicSupplyChain,
)


class InventoryPolicy:
    """Abstract base class for inventory policies."""

    def __init__(self):
        self.supply_chain = None

    def __init__(self):
        self.active = True

    def activate(self):
        """"""
        self.active = True

    def deactivate(self):
        """"""
        self.active = False

    def set_supply_chain(self, supply_chain: "SupplyChain"):
        self.supply_chain = supply_chain

    @classmethod
    def optimize(
        cls,
        supply_chain: "SupplyChain",
    ) -> "InventoryPolicy":
        """Returns instance with optimized parameters"""
        raise NotImplementedError("abstract method")

    def __repr__(self):
        return f"{self.__class__.__name__}: {vars(self)}"

    def __str__(self):
        return f"{self.__class__.__name__}: {vars(self)}"


class DeterministicPolicy(InventoryPolicy):
    pass


class StochasticPolicy(InventoryPolicy):
    pass


class QPolicy(DeterministicPolicy):
    """
    A minimalist continuous review policy with a fixed order quantity $Q$. Orders when inventory runs out.
    """

    def __init__(
        self,
        quantity: int,
    ):
        InventoryPolicy.__init__(self)
        self.quantity = quantity

    def __call__(self, t, level) -> int:
        if self.active:
            if level <= 0:
                return self.quantity
            else:
                return 0
        else:
            return None

    @property
    def cycle_stock(self):
        """Average stock we have through an order cycle to fulfill the (expected) demand"""
        return self.quantity / 2

    @property
    def holding_cost(self):
        return self.cycle_stock * self.supply_chain.product.unit_holding_cost

    @classmethod
    def optimize(
        cls,
        supply_chain: "SupplyChain",
    ) -> "QPolicy":
        """Optimize the policy parameters for the given supply chain."""

        q = supply_chain.EOQ()
        return cls(
            quantity=q,
        )


class PQPolicy(DeterministicPolicy):
    """
    $(p,q)$ A continuous review policy with a reorder point $p$ and a fixed order quantity $q$.
        + Pros: low amount of safety stock needed; optimized order quantity.
        - Cons: continuous review needed; multiple items cannot be grouped in one
        order with one supplier.
    """

    def __init__(
        self,
        quantity: int,
        reorder_point: int,
    ):
        InventoryPolicy.__init__(self)
        self.reorder_point = reorder_point
        self.quantity = quantity

    def __call__(self, t, level) -> int:
        if self.active:
            if level <= self.reorder_point:
                return self.quantity
            else:
                return 0
        else:
            return None

    @property
    def cycle_stock(self):
        """Average stock we have through an order cycle to fulfill the (expected) demand"""
        # TODO: review
        return (self.quantity / 2) + self.reorder_point

    @property
    def holding_cost(self):
        return self.cycle_stock * self.supply_chain.product.unit_holding_cost

    @classmethod
    def optimize(
        cls,
        supply_chain: DeterministicSupplyChain,
    ) -> "PQPolicy":
        """ """
        q = supply_chain.EOQ()
        p = supply_chain.demand_over_lead_time()
        return cls(
            quantity=q,
            reorder_point=p,
        )


class RSPolicy(DeterministicPolicy):
    """
    (r,s) A periodic review policy following a review period $r$ and an orderup-tolevel $s$.
        + Pros: multiple items can be grouped in one order with one supplier.
        - Cons: more safety stock needed; order quantity varies and can't be optimized.
    """

    def __init__(
        self,
        review_period: int,
        up_to_level: int = 0,
    ):
        InventoryPolicy.__init__(self)
        self.review_period = review_period
        self.up_to_level = up_to_level

    def __call__(self, t, level) -> int:
        if self.active:
            if (t % self.review_period) == 0:
                return self.up_to_level - level
            else:
                return 0
        else:
            return None

    @property
    def cycle_stock(self):
        """Average stock we have through an order cycle to fulfill the (expected) demand"""
        # Vandeput - p.37
        d_r = self.demand  # d_r: demand over review period
        C_s = d_r / 2
        return C_s

    @property
    def in_transit_stock(self):
        """Average stock we have through an order cycle to fulfill the (expected) demand"""
        in_transit = self.up_to_level - self.cycle_stock
        return in_transit

    @property
    def holding_cost(self):  # of order quantity over review period?
        raise NotImplementedError()

    @classmethod
    def optimize(
        cls,
        supply_chain,
        review_period: Optional[int] = None,
    ) -> "RSPolicy":
        """Optimize the policy parameters for the given supply chain.
        
        Args:
            supply_chain (SupplyChain): The supply chain to optimize for.
            review_period (Optional[int]): The time interval between reviews of the inventory position. 
                If not provided, will attempt to optimize.

        Returns:
            RSPolicy: The optimized inventory policy.
        """
        # Vandeput p.95, In general, multiple values are possible for R_opt.
        # However, since cls takes only one single integer pair as an argument, one value needs to be selected for R_opt und S_opt respectively.
        # For R_opt the value closest to the self.freq seems like a reasonable choice.
        # FIXME: remove freq_min attribute since freq is actually freq_min
        # try:
        #     freq = int(supply_chain.freq.days)
        #     freq_min = int(supply_chain.freq_min.days)
        # except:
        #     freq = int(supply_chain.freq.seconds / 3600)
        #     freq_min = int(supply_chain.freq_min.seconds / 3600)

        # R_opt_list = [2 ** k * freq_min for k in range(2 * freq / freq_min)]
        # differences_list = [abs(r - freq) for r in R_opt_list]
        # index_min_difference = differences_list.index(max(differences_list))
        # R_opt = R_opt_list[index_min_difference]

        # determine safety stock
        if isinstance(supply_chain, StochasticSupplyChain):
            if supply_chain.service_level_target is None:
                raise ValueError("SupplyChain needs a service level target")

            if review_period is not None:
                R_opt = review_period
            else:
                raise NotImplementedError("pass a review period - review period optimization not yet implemented")

            cycle_stock, safety_stock = optimal_order_up_to_levels(
                demand_distribution=supply_chain.expected_demand_distribution(
                    horizon=R_opt
                ),
                cycle_service_level_target=supply_chain.service_level_target,
            )

            # TODO: include lead time distribution

            S_opt = cycle_stock + safety_stock

        elif isinstance(supply_chain, DeterministicSupplyChain):
            # if demand and lead time are deterministic, we don't need safety stock
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"unknown SupplyChain type: {type(supply_chain)}")

        return cls(
            review_period=R_opt,
            up_to_level=S_opt,
        )


class RPQPolicy(DeterministicPolicy):
    """
    (r, p, q) A periodic review policy following a review period $r$, with a reorder point $p$
    and a fixed order quantity $q$.
        + Pros: can group multiple items in one order with one supplier; optimized or-
        der quantity.
        - Cons: even more safety stock needed; optimization is difficult.
    """

    def __init__(
        self,
        review_period: int,
        quantity: int,
        reorder_point: int = 0,
    ):
        InventoryPolicy.__init__(self)
        self.review_period = review_period
        self.quantity = quantity
        self.reorder_point = reorder_point

    def __call__(self, t, level) -> Optional[int]:
        if self.active:
            if (t % self.review_period) == 0:
                if level <= self.reorder_point:
                    return self.quantity
                else:
                    return 0
            else:
                return 0
        else:
            return None

    @property
    def cycle_stock(self):
        """Average stock we have through an order cycle to fulfill the (expected) demand"""
        raise NotImplementedError()

    @property
    def holding_cost(self):
        raise NotImplementedError()

    @classmethod
    def optimize(
        cls,
        supply_chain: "SupplyChain",
    ) -> "RPQPolicy":
        """ """
        raise NotImplementedError()


class RPSPolicy(DeterministicPolicy):
    """
    (r, p, s) A periodic review policy following a review period $r$, with a reorder point $p$
    and an order-up-to-level $s$.
        - Cons: difficult to optimize mathematically
    """

    def __init__(
        self,
        review_period: int,
        up_to_level: int,
        reorder_point: int = 0,
    ):
        InventoryPolicy.__init__(self)
        self.review_period = review_period
        self.up_to_level = up_to_level
        self.reorder_point = reorder_point

    def __call__(self, t, level) -> Optional[int]:
        if self.active:
            if (t % self.review_period) == 0:
                if level <= self.reorder_point:
                    return self.up_to_level - level
                else:
                    return 0
            else:
                return 0
        else:
            return None

    @property
    def cycle_stock(self):
        """Average stock we have through an order cycle to fulfill the (expected) demand"""
        raise NotImplementedError()

    @property
    def holding_cost(self):
        raise NotImplementedError()

    @classmethod
    def optimize(
        cls,
        supply_chain: "SupplyChain",
    ) -> "RPQPolicy":
        """ """
        raise NotImplementedError()


class ManualPolicy(InventoryPolicy):
    """This policy applies manual orders to the supply chain."""

    def __init__(
        self,
        orders,
    ):
        InventoryPolicy.__init__(self)
        self.orders = orders  # sequence of orders
        self.iterator = iter(self.orders)

    def __call__(self, t, level) -> int:
        if self.active:
            order = next(self.iterator)
            return order
        else:
            return None


def construct_policy(
    review_period: int = None,
    reorder_point: int = None,
    quantity: int = None,
    up_to_level: int = None,
):
    """Construct an inventory policy based on the provided parameters.

    Args:
        review_period (int, optional): The time interval between reviews of the inventory position.
        reorder_point (int, optional): The inventory level that triggers a replenishment order.
        quantity (int, optional): The fixed quantity to order when the reorder point is reached.
        up_to_level (int, optional): The target inventory level to order up to at each review period.

    Returns:
        InventoryPolicy: The constructed inventory policy.

    Raises:
        ValueError: If no valid policy can be constructed from the given parameters.
    """
    if (
        (review_period is None)
        and (reorder_point is None)
        and (quantity is not None)
        and (up_to_level is None)
    ):
        return QPolicy(
            quantity=quantity,
        )
    elif (
        (review_period is None)
        and (reorder_point is not None)
        and (quantity is not None)
        and (up_to_level is None)
    ):
        return PQPolicy(
            reorder_point=reorder_point,
            quantity=quantity,
        )
    elif (
        (review_period is not None)
        and (reorder_point is None)
        and (quantity is None)
        and (up_to_level is not None)
    ):
        return RSPolicy(
            review_period=review_period,
            up_to_level=up_to_level,
        )
    elif (
        (review_period is not None)
        and (reorder_point is not None)
        and (quantity is not None)
        and (up_to_level is None)
    ):
        return RPQPolicy(
            review_period=review_period,
            reorder_point=reorder_point,
            quantity=quantity,
        )
    else:
        raise ValueError("No policy for given configuration found")
