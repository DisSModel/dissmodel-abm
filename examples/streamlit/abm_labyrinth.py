"""
Labyrinth — Streamlit
=======================
Maze-escape agent-based model, ported from TerraME's ``logo`` package
(https://www.terrame.org/package/logo/models/#labyrinth).

Usage
-----
    streamlit run examples/streamlit/abm_labyrinth.py
"""
from __future__ import annotations

from matplotlib.colors import ListedColormap
import streamlit as st

from dissmodel.core import Environment
from dissmodel.visualization import Map, Chart
from dissmodel_abm.models import LabyrinthModel, build_labyrinth, PATTERNS, STATE_CODES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Labyrinth", layout="centered")
st.title("Labyrinth (dissmodel-abm)")
st.caption(
    "TerraME logo: Labyrinth — "
    "https://www.terrame.org/package/logo/models/#labyrinth"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

pattern    = st.sidebar.selectbox("Maze pattern", options=list(PATTERNS), index=0)
n_walkers  = st.sidebar.slider("Number of walkers", min_value=1, max_value=30, value=5)
steps      = st.sidebar.slider("Simulation steps", min_value=1, max_value=500, value=100)
seed       = st.sidebar.number_input("Random seed", min_value=0, value=0, step=1)

run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
gdf = build_labyrinth(pattern)

env = Environment(start_time=0, end_time=steps)

model = LabyrinthModel(gdf=gdf, n_walkers=n_walkers, seed=seed)

# LabyrinthModel.active / .found are registered via @track_plot, so
# Chart() below picks them up automatically — same convention as
# dissmodel-sysdyn's SIR model.

# ---------------------------------------------------------------------------
# Visualization: map (empty=white, exit=red, found=green, wall=black,
# walker=orange) + active/found chart
# ---------------------------------------------------------------------------
# Colors ordered by STATE_CODES (empty, exit, found, wall, walker).
# Plotting the numeric "state_code" column with fixed vmin/vmax keeps
# colors stable across steps, even when a state (e.g. "exit") is no
# longer present in the data once it is found.
cmap = ListedColormap(["white", "tab:red", "tab:green", "black", "tab:orange"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Maze")
    map_area = st.empty()
with col2:
    st.subheader("Walkers over time")
    chart_area = st.empty()

Map(
    gdf=gdf,
    plot_params={
        "column": "state_code",
        "cmap": cmap,
        "vmin": 0,
        "vmax": len(STATE_CODES) - 1,
        "markersize": 25,
    },
    plot_area=map_area,
)

Chart(
    plot_area=chart_area,
    show_legend=True,
    show_grid=True,
    title="Walkers",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    env.reset()
    env.run()
