# -*- coding: utf-8 -*-

"""Unit tests for the estimation module."""

from scipy.stats import norm
import pandas
import numpy

from supplyflow import estimation


def test_fit_normal_distribution():
    data = norm.rvs(10.0, 2.5, size=500)
    (mu, sigma) = estimation.fit_normal_distribution(data)
    print((mu, sigma))


def test_forecast_error_distribution(demo_article_forecast):
    forecast_min = demo_article_forecast["f"].min()
    forecast_max = demo_article_forecast["f"].max()

    real_demand_dummy = pandas.DataFrame(
        numpy.random.randint(
            0.5 * forecast_min, 0.5 * forecast_max, size=(len(demo_article_forecast), 2)
        ),
        columns=["d", "demand"],
    )
    real_demand_dummy.loc[real_demand_dummy["demand"] < 0, "demand"] = 0
    real_demand_dummy["d"] = demo_article_forecast["d"]

    error_dist = estimation.forecast_error_distribution(
        article_forecast=demo_article_forecast,
        article_actual=real_demand_dummy,
        forecast_sales_col="f",
        actual_sales_col="demand",
    )
