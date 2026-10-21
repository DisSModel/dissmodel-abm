"""
Explorer for the models ported from TerraME's ``logo`` package.

Usage
-----
    pip install -e ".[viz]"
    streamlit run examples/streamlit/logo_explorer.py

Pick a model in the sidebar, set its parameters and the seed, and run. The
map is a ``RasterMap`` over the model's ``CellSpace``; the chart is a
``Chart`` that reads the attributes declared with ``@track_plot``.
"""
from __future__ import annotations

import streamlit as st

from dissmodel.core import Environment
from dissmodel.visualization import Chart, RasterMap
from dissmodel_abm.models import Ants, LifeCycle, PredatorPrey, Schelling, SingleAgent
from dissmodel_abm.models import ants, life_cycle, predator_prey, schelling

st.set_page_config(page_title="dissmodel-abm: logo models", layout="wide")
st.title("TerraME logo models in dissmodel-abm")

name = st.sidebar.selectbox(
    "Model", ["Schelling", "PredatorPrey", "LifeCycle", "Ants", "SingleAgent"]
)
seed = int(st.sidebar.number_input("Seed", min_value=0, value=0, step=1))

if name == "Schelling":
    cls, band, colors, labels, final = Schelling, "state", schelling.COLORS, schelling.LABELS, 500
    params = dict(
        dim=st.sidebar.slider("dim", 10, 100, 25),
        free_space=st.sidebar.slider("free_space", 0.05, 0.5, 0.05, step=0.05),
        preference=st.sidebar.slider("preference", 1, 8, 3),
    )
elif name == "PredatorPrey":
    cls, band, colors, labels, final = (
        PredatorPrey, "who", predator_prey.WHO_COLORS, predator_prey.WHO_LABELS, 500,
    )
    params = dict(
        dim=st.sidebar.slider("dim", 10, 80, 30),
        rabbits=st.sidebar.slider("rabbits", 1, 200, 20),
        wolves=st.sidebar.slider("wolves", 1, 200, 20),
    )
elif name == "LifeCycle":
    cls, band, colors, labels, final = LifeCycle, "state", life_cycle.COLORS, life_cycle.LABELS, 100
    params = dict(
        dim=st.sidebar.slider("dim", 5, 50, 10),
        quantity=st.sidebar.slider("quantity", 1, 50, 10),
    )
elif name == "Ants":
    cls, band, colors, labels, final = Ants, "view", ants.COLORS, ants.LABELS, 450
    params = dict(
        dimension=st.sidebar.slider("dimension", 20, 100, 50),
        society_size=st.sidebar.slider("society_size", 10, 500, 10),
        rate_evaporation=st.sidebar.slider("rate_evaporation", 0.05, 0.95, 0.2, step=0.05),
    )
else:
    cls, band, colors, labels, final = SingleAgent, "n_agents", {0: "#1a9850", 1: "#000000"}, {0: "empty", 1: "agent"}, 100
    params = dict(
        dim=st.sidebar.slider("dim", 5, 50, 15),
        quantity=st.sidebar.slider("quantity", 1, 20, 1),
    )

steps = st.sidebar.slider("Steps", 1, 2 * final, min(final, 200))
st.caption(f"Port of `{name}` from TerraME `logo` 0.5.6. Default steps in the original: {final}.")

col_map, col_chart = st.columns(2)
map_area = col_map.empty()
chart_area = col_chart.empty()

if st.sidebar.button("Run"):
    env = Environment(start_time=1, end_time=steps)
    model = cls(seed=seed, **params)
    RasterMap(
        backend=model.cs.backend,
        band=band,
        title=name,
        color_map=colors,
        labels=labels,
        plot_area=map_area,
    )
    if getattr(cls, "_plot_info", None):
        Chart(plot_area=chart_area, show_grid=True, title=name)
    env.run()
    st.sidebar.success(f"Done: {steps} steps, seed {seed}.")
