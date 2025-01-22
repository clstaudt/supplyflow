# -*- coding: utf-8 -*-

from scipy.stats import norm
from pathlib import Path
from pandera.typing import DataFrame

import numpy
import pandas


def fit_normal_distribution(data):
    """[summary]

    Args:
        demand ([type]): [description]
    """
    mu, sigma = norm.fit(data)
    return (mu, sigma)


def forecast_error_distribution(
    article_forecast: DataFrame,
    article_actual: DataFrame,
    forecast_sales_col: str,
    actual_sales_col: str,
):
    """Fit normal distribution to article forecast errors."""
    errors = (
        article_forecast.loc[:, forecast_sales_col]
        - article_actual.loc[:, actual_sales_col]
    )
    errors = errors.dropna()
    # TODO: return distribution
    mu, sigma = norm.fit(errors)
    return (mu, sigma)


def leadtime_distribution(path_reference_data_leadtime):
    leadtime_data = pandas.read_csv(path_reference_data_leadtime)
    return fit_normal_distribution(leadtime_data)
