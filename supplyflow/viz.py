# -*- coding: utf-8 -*-

"""Data visualization tools."""

from datetime import timedelta

import numpy
import pandas
import altair
from pandera.typing import DataFrame


from .core import SupplyChain
from .policy import InventoryPolicy

# Vega color schemes: https://vega.github.io/vega/docs/schemes/
default_color_scheme = "dark2"


def dark_theme():
    return {
        "config": {
            "view": {
                "continuousHeight": 300,
                "continuousWidth": 400,
            },  # from the default theme
            "range": {"category": {"scheme": "category20"}},
        }
    }


def enable_theme(theme_name="dark"):
    """Enable one of the available custom Altair theme."""
    if theme_name == "dark":
        altair.themes.register("dark", dark_theme)
        altair.themes.enable("dark")
        altair.renderers.set_embed_options(theme="dark")
    elif theme_name == "default_dark":
        altair.renderers.set_embed_options(theme="dark")
    else:
        raise ValueError("unknown theme: {theme_name}")


def supply_chain_plot(
    supply_chain: SupplyChain,
    metrics: bool = True,
):

    plot_title = altair.TitleParams(
        f"{supply_chain}",
        subtitle=[
            f"fill rate: {supply_chain.fill_rate():.3f}"
            + " / "
            + f"weekly service level: {supply_chain.period_service_level(period=timedelta(days=7)):.3f}"
            + " / "
            + f"cycle service level: {supply_chain.cycle_service_level():.3f}",
            f"cost: {supply_chain.total_cost:.2f}",
        ],
    )

    plot_data = supply_chain.history_data
    plot_data["order"] = plot_data["order"].fillna(0)
    plot_data = pandas.melt(supply_chain.history_data, id_vars=["time"])

    domain = [
        "level",
        "demand",
        "order",
        "fill_rate",
        "avg_service_level",
        "overall_avg_service_level",
    ]

    color_range = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    main_layers = []

    demand_plot = (
        altair.Chart(plot_data[plot_data["variable"] == "demand"], title=plot_title)
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            # color=altair.Color('variable:N', title = "")
            color=altair.Color(
                "variable:N",
                scale=altair.Scale(domain=domain, range=color_range),
                title="",
            ),
        )
        .properties(
            width=900,
            height=200,
        )
    )
    main_layers.append(demand_plot)

    if supply_chain.forecast_source:
        forecast_plot = (
            altair.Chart(
                plot_data,
                title=plot_title,
            )
            .mark_line()
            .encode(
                x="time:T",
                y="forecast:Q",
                color=altair.Color(
                    "variable:N",
                    scale=altair.Scale(domain=domain, range=color_range),
                ),
            )
            .properties(
                width=900,
                height=200,
            )
        )
        main_layers.append(forecast_plot)

    level_plot = (
        altair.Chart(plot_data[plot_data["variable"] == "level"])
        .mark_area(opacity=0.6)
        .encode(
            x="time:T",
            y="value:Q",
            # color=altair.Color('variable:N', title = "")
            color=altair.Color(
                "variable:N", scale=altair.Scale(domain=domain, range=color_range)
            ),
        )
        .properties(
            width=900,
            height=200,
        )
    )
    main_layers.append(level_plot)

    order_plot = (
        altair.Chart(plot_data[plot_data["variable"] == "order"])
        .mark_bar()
        .encode(
            x="time:T",
            y="value:Q",
            # color=altair.Color('variable:N', title = "")
            color=altair.Color(
                "variable:N", scale=altair.Scale(domain=domain, range=color_range)
            ),
        )
        .properties(
            width=900,
            height=200,
        )
    )
    main_layers.append(order_plot)

    main_layer = altair.layer(*main_layers)

    # metrics

    fill_rate_plot = (
        altair.Chart(plot_data[plot_data["variable"] == "fill_rate"])
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            # color=altair.Color('variable:N', title = "")
            color=altair.Color(
                "variable:N", scale=altair.Scale(domain=domain, range=color_range)
            ),
        )
        .properties(
            width=900,
            height=200,
        )
    )

    service_level_plot = (
        altair.Chart(plot_data[plot_data["variable"] == "avg_service_level"])
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            # color=altair.Color('variable:N', title = "")
            color=altair.Color(
                "variable:N", scale=altair.Scale(domain=domain, range=color_range)
            ),
        )
        .properties(
            width=900,
            height=200,
        )
    )

    overall_service_level_plot = (
        altair.Chart(plot_data[plot_data["variable"] == "overall_avg_service_level"])
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            #    color=altair.Color('variable:N', title = "")
            color=altair.Color(
                "variable:N", scale=altair.Scale(domain=domain, range=color_range)
            ),
        )
        .properties(
            width=900,
            height=200,
        )
    )
    metrics_layer = altair.layer(
        fill_rate_plot,
        service_level_plot,
        overall_service_level_plot,
    )

    # layer_plot = layer1 & layer2.encode(y = "time:T")
    combined_plot = altair.vconcat(main_layer, metrics_layer)

    return combined_plot


def plot_stock_level_forecast(
    stock_level_forecast: DataFrame,
):
    plot = (
        altair.Chart(stock_level_forecast.reset_index())
        .mark_area(
            opacity=0.6,
        )
        .encode(x="index:T", y="level:Q")
        .properties(
            width=900,
            height=200,
        )
    )
    return plot


def plot_normal_distribution(distribution):
    mu = distribution.mean()
    sigma = distribution.std()  
    x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = distribution.pdf(x)
    
    plot = (
        altair.Chart(
            pandas.DataFrame(
                { 
                    "x": x,
                    "y": y
                }
            )
        ).mark_line()
        .encode(
            x="x",
            y="y",
        )
    )
    return plot
