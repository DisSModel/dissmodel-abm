"""
Random Walk — Streamlit
========================
Agents perform an independent random walk within a bounding box.

Usage
-----
    streamlit run examples/streamlit/abm_random_walk.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import streamlit as st

from dissmodel.core import Environment
from dissmodel.visualization import Map
from dissmodel_abm.models import RandomWalkModel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Random Walk", layout="centered")
st.title("Random Walk (dissmodel-abm)")
st.caption(
    "TerraME logo: SingleAgent / generalized to N agents — "
    "https://www.terrame.org/package/logo/models/#singleagent"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

n_agents  = st.sidebar.slider("Number of agents", min_value=1, max_value=200, value=20)
step_size = st.sidebar.slider("Step size", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
dim       = st.sidebar.slider("Space dimension", min_value=10, max_value=200, value=100)
steps     = st.sidebar.slider("Simulation steps", min_value=1, max_value=200, value=30)
seed      = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
bounds = (0, 0, dim, dim)
rng = np.random.default_rng(seed)
xs = rng.uniform(bounds[0], bounds[2], n_agents)
ys = rng.uniform(bounds[1], bounds[3], n_agents)

gdf = gpd.GeoDataFrame({"geometry": gpd.points_from_xy(xs, ys)})

env = Environment(start_time=0, end_time=steps)

model = RandomWalkModel(gdf=gdf, step_size=step_size, bounds=bounds)

plot_area = st.empty()
Map(
    gdf=gdf,
    plot_params={"markersize": 30, "color": "tab:blue"},
    plot_area=plot_area,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    env.reset()
    env.run()
