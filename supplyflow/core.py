# -*- coding: utf-8 -*-

"""Data model: products, supply chains, ..."""

from typing import Dict, Callable
from abc import ABC, abstractmethod

import sys
import os
import math
from dataclasses import dataclass
from typing import Optional
import random

import pandas
import numpy as np
from scipy import stats
from pandera.typing import DataFrame, Series
from datetime import datetime, timedelta
import tqdm
from loguru import logger


@dataclass(unsafe_hash=True)
class Product:
    name: str
    _id: int = None
    category: str = None
    unit_cost: float = 1  # [€]
    holding_cost_rate: float = 0.15  # fraction of unit cost [1 / unit·year], S.13
    holding_cost: float = None
    sale_price: float = None

    @property
    def unit_holding_cost(self):
        """Yearly unit holding cost."""
        if self.holding_cost:
            return self.holding_cost
        else:
            return self.holding_cost_rate * self.unit_cost


@dataclass(unsafe_hash=True)
class Warehouse:
    name: str
    _id: int = None
    capacity: int = None


@dataclass(unsafe_hash=True)
class Supplier:
    name: str
    _id: int = None


class DemandSource(ABC):
    @abstractmethod
    def __call__(self, t) -> int:
        pass
        
    @abstractmethod 
    def integrate(self, t_i, t_j) -> int:
        pass


class DeterministicDemandSource(DemandSource):
    """Demand source with deterministic (non-random) demand."""
    pass


class StochasticDemandSource(DemandSource):
    """Demand source with stochastic (random) demand."""
    pass


class ConstantDemand(DeterministicDemandSource):
    """Simple demand function: Demand is constant per iteration."""

    def __init__(self, demand):
        self.demand = demand

    def __call__(self, t) -> int:
        return self.demand

    def integrate(self, t_i, t_j) -> int:
        """Sum of demand units over time interval [t_i, t_j]."""
        assert t_j > t_i
        s = (t_j - t_i) * self.demand
        assert s >= 0
        return s


class DemandData(DeterministicDemandSource):
    """Feed values from time series of demand."""

    def __init__(self, values):
        self.values = values

    def __call__(self, t) -> int:
        try:
            return self.values[t]
        except:
            raise IndexError(f"no demand data for index {t}")

    def integrate(self, t_i, t_j) -> int:
        assert t_j > t_i
        return self.values[t_i : t_j + 1].sum()


class NormalDemand(StochasticDemandSource):
    """Normally distributed stochastic demand"""

    def __init__(self, mu, sigma, discrete=True, random_state=None):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm(mu, sigma)
        self.discrete = discrete
        self.random_state = random_state

    def __call__(self, t) -> int:
        r = self.dist.rvs(1, random_state=self.random_state)[0]
        if r < 0:
            r = 0
        if self.discrete:
            return round(r)
        else:
            return r
    
    def integrate(self, t_i, t_j) -> int:
        """Estimate the total demand over the interval [t_i, t_j]."""
        assert t_j > t_i
        num_steps = t_j - t_i
        demand = self.dist.rvs(num_steps, random_state=self.random_state)
        demand[demand < 0] = 0
        if self.discrete:
            demand = np.round(demand)
        return int(demand.sum())


class DemandEstimator:
    """Abstract base class for methods to estimate stochastic demand."""

    def get_distribution(
        self,
        horizon,
    ) -> stats.norm:
        """Estimate the normal distribution of demand for a certain horizon."""
        raise NotImplementedError("TODO")

    def set_supply_chain(
        self,
        supply_chain: "StochasticSupplyChain",
    ):
        self.supply_chain = supply_chain


class LeadTimeSource:
    """Abstract base class for lead time sources.
    
    Lead time source represents the lead time pattern or function.
    Lead time is the time between placing an order and receiving it.
    """
    pass


class DeterministicLeadTimeSource(LeadTimeSource):
    """Abstract base class for deterministic lead time sources."""
    pass


class StochasticLeadTimeSource(LeadTimeSource):
    """Abstract base class for stochastic lead time sources."""
    pass


class ConstantLeadTime(DeterministicLeadTimeSource):
    """Lead time with a constant value."""

    def __init__(
        self,
        time: timedelta,
    ):
        self.time = time

    def __call__(self, t) -> timedelta:
        return self.time


class NormalLeadTime(StochasticLeadTimeSource):
    """Normally distributed lead time."""

    def __init__(self, mu, sigma, discrete=True):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm(mu, sigma)
        self.discrete = discrete

    def __call__(self, t) -> int:
        """Return a random lead time value from the normal distribution.
        
        Args:
            t (int): Time step (not used, for compatibility).
            
        Returns:
            int: Random lead time value.
        """
        r = self.dist.rvs(1)[0]  
        if self.discrete:
            return round(r)
        else:
            return r


class SupplyChain:
    """
    Abstract base class representing a supply chain.
    
    A supply chain manages the flow of a product from a supplier to a warehouse, 
    tracking inventory levels and costs.

    Args:
        product (Product): The product managed by this supply chain.
        warehouse (Warehouse): The warehouse where the product is stored.
        supplier (Supplier): The supplier of the product.
        level (int): The initial stock level in units.
        freq (timedelta): The review period, i.e. how often inventory is checked.
        start_time (datetime): The start time of the simulation.
        transaction_fixed_cost (float): The fixed cost per order.
        transaction_cost_function (Callable[[int], float]): A function that computes the variable cost per order given the order quantity.
        forecast_horizon (int): The number of iterations into the future for which demand forecasts are available.
    """

    def __init__(
        self,
        product: Product,
        warehouse: Warehouse,
        supplier: Supplier = None,
        level: int = 0,
        freq: timedelta = timedelta(days=1),
        start_time: datetime = datetime.today(),
        transaction_fixed_cost: float = 0.0,
        transaction_cost_function: Callable[[int], float] = None,
        forecast_horizon: Optional[int] = None,
    ):
        self.product = product
        self.warehouse = warehouse
        self.supplier = supplier
        self.level = level
        self.freq = freq
        self.start_time = start_time
        self.transaction_cost_function = transaction_cost_function
        self.transaction_fixed_cost = transaction_fixed_cost
        self.forecast_horizon = forecast_horizon
        # init internal data structures
        self.reset()

    def reset(self):
        # initialize internal state data structures
        self.history = []  # data rows
        self._order_quantity = 0  # current order quantity
        self._lead_time = 0  # current lead time
        self._t = 0  # iterations
        self._current_time = datetime.now()
        self._incoming_orders = []
        self._total_demand = 0
        self._total_demand_iteration = 0
        self._demand_fullfilled_iteration = 0
        self._demand_fullfilled = 0  # demand fullfilled from inventory
        self._fill_rate = 0  # fill rate current iteration
        self._transaction_cost = 0  # sum of transaction costs
        self._holding_cost = 0  # sum of holding costs
        self._out_of_stock_events = 0  # number of iterations with out of stock
        self._forecast = None
        self._order_cycle = 0

    def __str__(self):
        return f"SupplyChain({self.product.name}, {self.warehouse.name})"

    @property
    def history_data(self):
        """History of the supply chain."""
        data = pandas.DataFrame(self.history)
        data["overall_avg_service_level"] = data["avg_service_level"].mean()
        # set a real time index
        time_index = pandas.date_range(
            start=self.start_time, periods=data.shape[0], freq=self.freq
        )
        data = data.set_index(time_index)
        return data

    # POLICY

    def set_policy(self, policy: "InventoryPolicy"):
        logger.info(f"policy set: {policy}")
        self.policy = policy
        self.policy.set_supply_chain(self)
        return self

    # MECHANISM

    def _on_iteration(self):
        """Record data per iteration"""
        # account for holding cost
        try:
            self._holding_cost += self.level * self.product.unit_holding_cost
        except:
            pass

        # write history
        row = {
            "time": self._current_time,
            "order": self._order_quantity,
            "demand": self._demand,
            "forecast": self._forecast,
            "level": self.level,
            "iteration": self._t,
            "fill_rate": self._fill_rate,
            "avg_service_level": (1 - (self._out_of_stock_events / (self._t + 1))),
            "order_cycle": self._order_cycle,
            "out_of_stock": self._out_of_stock,
        }
        logger.debug(row)
        self.history.append(row)

    def _order(self):
        """Send supply order."""
        self._lead_time = self.lead_time_source(self._t)
        self._incoming_orders.append(
            (
                self._order_quantity,
                arrival_time := self._t + self._lead_time,
            )  # ? iteration + lead_time?
        )
        # cost model
        self._transaction_cost += self.transaction_cost(self._order_quantity)
        self.policy.deactivate()  # wait for arriving order

    def _accept_delivery(self):
        """Accept incoming deliveries."""
        while (not len(self._incoming_orders) == 0) and (
            (first_arrival_time := self._incoming_orders[0][1]) <= self._t
        ):
            (order_quantity, _) = self._incoming_orders.pop()
            self.level += order_quantity
            self.policy.activate()

    def _serve_demand(self):
        """Serve demand from inventory."""
        self._demand = self.demand_source(self._t)
        self._total_demand += self._demand
        self._total_demand_iteration += self._demand
        if self._demand <= self.level:
            self._out_of_stock = False
            self._demand_fullfilled += self._demand
            self._demand_fullfilled_iteration += self._demand
            self.level -= self._demand
        else:
            self._out_of_stock = True
            self._demand_fullfilled += self.level
            self._demand_fullfilled_iteration += self.level
            self.level = 0

    def _current_fill_rate(self):
        """Fraction of demand that can be served directly from the stock in the current iteration.

        Returns:
            float: fill rate
        """
        demand_fullfilled_iteration = self._demand_fullfilled_iteration
        total_demand_iteration = self._total_demand_iteration
        if total_demand_iteration == 0:
            return 1.0  # fill rate is 1 if no demand
        current_fill_rate = demand_fullfilled_iteration / total_demand_iteration

        self._fill_rate = current_fill_rate

    def _events_count(self):
        if self._demand > self.level:
            self._out_of_stock_events += 1

    def run(self, iterations):
        """Run the supply chain for a number of iterations."""
        if self.policy is None:
            raise ValueError("Supply chain cannot run without InventoryPolicy")
        for i in tqdm.tqdm(range(iterations)):
            if self.forecast_source:
                self._forecast = self.forecast_source(self._t)
            self._order_quantity = self.policy(self._t, self.level)
            if self._order_quantity:
                self._order()
                self._order_cycle += 1  # new order cycle starts
            # accept orders
            self._accept_delivery()
            # serve demand
            self._serve_demand()

            self._current_fill_rate()

            self._events_count()
            #
            self._on_iteration()

            self._total_demand_iteration = 0
            self._demand_fullfilled_iteration = 0
            self._fill_rate = 0
            self._t += 1
            self._current_time += self.freq

    # COST

    def transaction_cost(self, order_quantity):
        if self.transaction_cost_function:
            return self.transaction_cost_function(order_quantity)
        else:
            return self.transaction_fixed_cost

    @property
    def total_transaction_cost(self):
        """Sum of transaction cost accumulated over the lifetime of the supply chain."""
        return self._transaction_cost

    @property
    def total_holding_cost(self):
        """Sum of holding cost accumulated over the lifetime of the supply chain."""
        return self._holding_cost

    @property
    def total_cost(self):
        """Sum of all cost accumulated over the lifetime of the supply chain."""
        total_cost = self.total_transaction_cost + self.total_holding_cost
        return total_cost

    # DEMAND
    @property
    def total_demand(self):
        """Sum of all demand accumulated over the lifetime of the supply chain."""
        return self._total_demand

    @property
    def fulfilled_demand(self):
        """Sum of all demand fulfilled from stocks accumulated over the lifetime of the supply chain."""
        return self._demand_fullfilled

    @property
    def sales(self):
        """Revenue over the lifetime of the supply chain."""
        return self.fulfilled_demand * self.product.sale_price

    @property
    def lost_sales(self):
        """Revenue lost due to unfulfilled demand over the lifetime of the supply chain."""
        unfulfilled_units = self.total_demand - self.fulfilled_demand
        return unfulfilled_units * self.product.sale_price

    # OPTIMIZATION

    def expected_total_demand(self, timespan) -> int:
        """Expected total yearly demand."""
        raise NotImplementedError("TODO")
        # TODO: calculate by extrapolating the forecast
        total = None
        return total

    def EOQ(self):
        """Economical Order Quantity - cf. Vandeput p. 18"""
        k = self.transaction_fixed_cost
        D = self.expected_total_demand(timespan=timedelta(days=365))
        h = self.product.unit_holding_cost  # per year
        quantity = math.ceil(math.sqrt(2 * k * D / h))
        return quantity

    def optimal_review_period(self, rounding_strategy="simple"):
        """Optimal review period matching the Economical Order Quantity"""
        review_period = self.EOQ() / self.expected_total_demand(
            timespan=timedelta(days=365)
        )
        if rounding_strategy == "simple":
            review_period = math.round(review_period)
        elif rounding_strategy == "power-of-2":
            raise NotImplementedError("TODO")
        return review_period

    # METRICS

    def fill_rate(self):
        """The fraction of the demand (over the long term) that is supplied directly from the on-hand inventory."""
        return self._demand_fullfilled / self._total_demand

    def cycle_service_level(self):
        """The empirical probability that no out-of-stock situation
        occurs between order cycles."""
        cycle_service_level = (
            ~self.history_data.filter(["order_cycle", "out_of_stock"])
            .groupby("order_cycle")["out_of_stock"]
            .any()
        ).mean()
        return cycle_service_level

    def period_service_level(self, period: timedelta) -> float:
        """The empirical probability that no out-of-stock situation
        occurs in the given period."""
        period_service_level = (
            ~self.history_data.resample(period)["out_of_stock"].any()
        ).mean()
        return period_service_level

    def cycle_out_of_stock_risk(self):
        """The empirical probability that an out-of-stock situation
        occurs between order cycles."""
        return 1 - self.cycle_service_level()

    def period_out_of_stock_risk(self, period):
        """The empirical probability that an out-of-stock situation
        occurs between order cycles."""
        return 1 - self.period_service_level(period)

    def lost_sales(self):
        raise NotImplementedError("TODO")

    # STOCK LEVEL PREDICTION

    def stock_level_lookahead(self, iterations):
        """Predict the stock level according to the forecast
        (but not applying the inventory policy).
        """
        raise NotImplementedError("TODO")
        if self.forecast_source is None:
            raise ValueError("Forecast required to predict future stock levels")
        for offset in range(iterations):
            d = self.forecast_source[self._t + offset]


class DeterministicSupplyChain(SupplyChain):
    """In a deterministic supply chain, demand and lead time are not random variables."""

    def __init__(
        self,
        product: Product,
        warehouse: Warehouse,
        demand_source: DeterministicDemandSource,
        supplier: Supplier = None,
        forecast_source: DeterministicDemandSource = None,
        lead_time_source: DeterministicLeadTimeSource = ConstantLeadTime(1),
        level: int = 0,  # stock level [units]
        freq: timedelta = timedelta(days=1),
        start_time: datetime = datetime.today(),
        transaction_fixed_cost: float = 0.0,  #
        transaction_cost_function: Callable[[int], float] = None,
        forecast_horizon: Optional[
            int
        ] = None,  # how many iterations into the future is a forecast available?
    ):
        SupplyChain.__init__(
            self,
            product=product,
            warehouse=warehouse,
            supplier=supplier,
            level=level,
            freq=freq,
            start_time=start_time,
            transaction_fixed_cost=transaction_fixed_cost,
            transaction_cost_function=transaction_cost_function,
            forecast_horizon=forecast_horizon,
        )
        self.demand_source = demand_source
        self.forecast_source = forecast_source
        self.lead_time_source = lead_time_source

    def demand_over_lead_time(self):
        """ """
        raise NotImplementedError("TODO")


class StochasticSupplyChain(SupplyChain):
    """In a stochastic supply chain, demand and lead time are random variables."""

    def __init__(
        self,
        product: Product,
        warehouse: Warehouse,
        demand_source: StochasticDemandSource,
        lead_time_source: StochasticLeadTimeSource,
        supplier: Supplier = None,
        forecast_source: StochasticDemandSource = None,
        level: int = 0,  # stock level [units]
        freq: timedelta = timedelta(days=1),  # review period
        start_time: datetime = datetime.today(),
        transaction_fixed_cost: float = 0.0,  #
        transaction_cost_function: Callable[[int], float] = None,
        service_level_target: Optional[float] = None,  # required cycle service level
        demand_estimator: Optional[DemandEstimator] = None,
        forecast_horizon: Optional[
            int
        ] = None,  # how many iterations into the future is a forecast available?
    ):
        SupplyChain.__init__(
            self,
            product=product,
            warehouse=warehouse,
            supplier=supplier,
            level=level,
            freq=freq,
            start_time=start_time,
            transaction_fixed_cost=transaction_fixed_cost,
            transaction_cost_function=transaction_cost_function,
            forecast_horizon=forecast_horizon,
        )
        self.demand_source = demand_source
        self.forecast_source = forecast_source
        self.lead_time_source = lead_time_source
        self.service_level_target = service_level_target
        self.demand_estimator = demand_estimator
        if self.demand_estimator is not None:
            self.demand_estimator.set_supply_chain(self)

    def expected_demand_distribution(
        self,
        horizon,
        from_time: datetime = None,
    ) -> stats.norm:
        """Return the expected demand distribution for a given horizon."""
        if self.demand_estimator is not None:
            if from_time is not None:
                start_date = from_time.date()
            start_date = self._current_time.date()
            distribution = self.demand_estimator.get_distribution(horizon=horizon)
            return distribution
        else:
            raise ValueError("A DemandEstimator is required")

    def expected_stock_level(
        self,
        horizon: int,
        confidence: float = 0.95,
    ):
        """Return the expected stock level for a given horizon."""
        demand_dist = self.expected_demand_distribution(horizon)
        expected_level = max(self.level - demand_dist.mean(), 0)
        return expected_level

    def demand_over(self, horizon, from_time=None):
        """Estimated demand over a given horizon."""
        if not from_time:
            from_time = self._t
        demand = self.forecast_source.integrate(from_time, from_time + horizon)
        return demand

    def demand_over_lead_time(self):
        """Estimated demand over lead time."""
        # FIXME: reimplement
        # l = self.lead_time

        # demand_forecast = forecast_df.loc[:, "daily_prediction"] - self.mu_e
        # average_demand = demand_forecast["daily_prediction"].mean()
        # demand_over_lead_time = average_demand * l

        # TODO: calculate from demand over lead time distribution
        raise NotImplementedError("TODO")
        return demand_over_lead_time

    def avg_in_transit_inventory(self):
        """Average in transit inventory, S. 95"""
        in_transit = self.demand_over_lead_time
        return in_transit

    def optimal_review_period(self):
        """Optimal review period basen on the base review period freq_min and review period (freq) selected before optimization"""
        # TODO: suggestion: this should be implemented as part of the .optimize methods of policies, not as a method of SupplyChain
        # FIXME: remove freq_min attribute since freq is actually freq_min
        if self.policy == RSPolicy:
            try:
                freq = int(self.supply_chain.freq.days)
                freq_min = int(self.supply_chain.freq_min.days)
            except:
                freq = int(self.supply_chain.freq.seconds / 3600)
                freq_min = int(self.supply_chain.freq_min.seconds / 3600)

            R_opt_list = [2 ** k * freq_min for k in range(freq)]
            differences_list = [abs(r - freq) for r in R_opt_list]
            index_min_difference = differences_list.index(max(differences_list))
            R_opt = R_opt_list[
                index_min_difference
            ]  # from several possible optimal review periods in the list R_opt_list the one closest to the initially selected review period is returned
            return R_opt
        else:
            return None

    def optimal_up_to_level(self, forecast_data):
        """
        Optimal level to be "ordered up to" for the currently set review period self.freq (not the optimal review period computed in the function above),
        based on the demand forecast and the leadtime with the corresponding variances
        """
        # Vandeput - S.95
        # TODO: this should be implemented as part of the .optimize methods of policies, not as a method of SupplyChain
        # FIXME: review implementation
        if self.policy == RSPolicy:

            forecast_df = pandas.DataFrame(forecast_data)

            demand_forecast = forecast_df.loc[:, "daily_prediction"] - self.mu_e  
            demand_over_review = demand_forecast["daily_prediction"].sum()
            alpha = self.service_level_target
            z = stats.norm.ppf(alpha)
            x_std = np.sqrt(
                (self.mu_lt + self.freq) * self.sigma_e ** 2
                + self.sigma_lt ** 2 * demand_forecast["daily_prediction"].mean() ** 2
            )
            safety_stocks = np.round(x_std * z).astype(int)

            demand_over_lead_time = self.demand_over_lead_time(forecast_data)

            S_opt = demand_over_lead_time + demand_over_review + safety_stocks

            return S_opt

    def expected_cycle_service_level(self):
        """Expected cycle service level based on
        - inventory at the beginning of the cycle
        - expected demand and demand variance
        """
        raise NotImplementedError("FIXME")
        # Vandeput - S.54



    def expected_period_service_level(self, period: int):
        """Expected period service level based on
        - inventory at the beginning of the period
        - expected demand and demand variance
        """
        raise NotImplementedError("FIXME")



@dataclass
class Basket:
    """A basket of articles ordered by a client."""

    order_time: datetime
    content: DataFrame


class Enterprise:
    """An enterprise consists of many supply chains."""

    def __init__(
        self,
        name: str,
        start_time: datetime = datetime.now(),
    ):
        self.name = name
        self.start_time = start_time
        self.supply_chains = []

    def add(self, supply_chain: SupplyChain):
        # enforce equal start times
        supply_chain.start_time = self.start_time
        self.supply_chains.append(supply_chain)

    def get_supply_chain(product: Product, warehouse: Warehouse) -> SupplyChain:
        """Get the supply chain for a product / warehouse combination"""
        raise NotImplementedError("TODO")

    @property
    def products(self):
        return set((supplychain.product for supplychain in self.supply_chains))

    @property
    def warehouses(self):
        return set((supplychain.warehouse for supplychain in self.supply_chains))

    @property
    def suppliers(self):
        return set((supplychain.supplier for supplychain in self.supply_chains))

    # RUNNING

    def run(self, timespan: timedelta):
        """Run all supply chains for a given timespan."""
        for supply_chain in self.supply_chains:
            iterations = timespan // supply_chain.freq
            supply_chain.run(iterations)

    # METRICS

    @property
    def fill_rate(self):
        """Average fill rate over all supply chains"""
        fill_rates = pandas.Series(
            [supply_chain.fill_rate() for supply_chain in self.supply_chains]
        )
        return fill_rates.mean()

    @property
    def current_out_of_stock_rate(self):
        """Fraction of supply chains having a stock-out currently."""
        stock_levels = pandas.Series(
            [supply_chain.level for supply_chain in self.supply_chains]
        )
        out_of_stock_rate = (stock_levels < 0).sum() / len(self.supply_chains)
        return out_of_stock_rate

    def client_service_level(
        self,
        orders,
    ):
        """
        The probability that a client receives their full order (often containing multiple
        different products) from stock.

        Args:
            orders ([type]): [description]
            stocks ([type]): [description]
        """
        raise NotImplementedError("TODO")
