"""
Predator-Prey (Wolf-Sheep) — Streamlit
=========================================
Classic agent-based predator-prey model, ported from TerraME's
``logo`` package (https://www.terrame.org/package/logo/models/#predatorprey).

Usage
-----
    streamlit run examples/streamlit/abm_predator_prey.py
"""
from __future__ import annotations

from matplotlib.colors import ListedColormap
import geopandas as gpd
import numpy as np
import streamlit as st

from dissmodel.core import Environment
from dissmodel.visualization import Map, Chart
from dissmodel_abm.models import PredatorPreyModel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predator-Prey", layout="centered")
st.title("Predator-Prey (dissmodel-abm)")
st.caption(
    "TerraME logo: PredatorPrey — "
    "https://www.terrame.org/package/logo/models/#predatorprey"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

n_sheep    = st.sidebar.slider("Initial sheep", min_value=5, max_value=200, value=60)
n_wolves   = st.sidebar.slider("Initial wolves", min_value=1, max_value=50, value=8)
dim        = st.sidebar.slider("Space dimension", min_value=10, max_value=200, value=50)
steps      = st.sidebar.slider("Simulation steps", min_value=1, max_value=200, value=30)

st.sidebar.subheader("Dynamics")
step_size   = st.sidebar.slider("Step size", 0.1, 10.0, 2.0, 0.1)
eat_radius  = st.sidebar.slider("Eat radius", 0.5, 10.0, 3.0, 0.5)
energy_loss = st.sidebar.slider("Energy loss / step", 0.0, 5.0, 0.5, 0.1)
energy_gain = st.sidebar.slider("Energy gain (eating)", 0.0, 20.0, 6.0, 0.5)
graze_gain  = st.sidebar.slider("Graze gain (sheep)", 0.0, 5.0, 0.7, 0.1)
reproduce_threshold = st.sidebar.slider("Reproduce threshold", 5.0, 50.0, 15.0, 1.0)

seed = st.sidebar.number_input("Random seed", min_value=0, value=7, step=1)
run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
bounds = (0, 0, dim, dim)
rng = np.random.default_rng(seed)
n_total = n_sheep + n_wolves
xs = rng.uniform(bounds[0], bounds[2], n_total)
ys = rng.uniform(bounds[1], bounds[3], n_total)

gdf = gpd.GeoDataFrame({
    "geometry": gpd.points_from_xy(xs, ys),
    "kind": ["sheep"] * n_sheep + ["wolf"] * n_wolves,
    "energy": [12.0] * n_total,
})

env = Environment(start_time=0, end_time=steps)

model = PredatorPreyModel(
    gdf=gdf,
    step_size=step_size,
    bounds=bounds,
    eat_radius=eat_radius,
    energy_loss=energy_loss,
    energy_gain=energy_gain,
    reproduce_threshold=reproduce_threshold,
    graze_gain=graze_gain,
)

# PredatorPreyModel.sheep_count / wolves_count are registered via
# @track_plot, so Chart() below picks them up automatically — same
# convention as dissmodel-sysdyn's SIR model.

# ---------------------------------------------------------------------------
# Visualization: map (sheep=blue, wolf=red) + population chart
# ---------------------------------------------------------------------------
cmap = ListedColormap(["tab:blue", "tab:red"])  # sheep, wolf (alphabetical)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Spatial distribution")
    map_area = st.empty()
with col2:
    st.subheader("Population over time")
    chart_area = st.empty()

Map(
    gdf=gdf,
    plot_params={"column": "kind", "cmap": cmap, "markersize": 25},
    plot_area=map_area,
)

Chart(
    plot_area=chart_area,
    show_legend=True,
    show_grid=True,
    title="Population",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    env.reset()
    env.run()
