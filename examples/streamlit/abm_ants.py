"""
Ants — Streamlit
===================
Ant colony foraging agent-based model, ported from TerraME's ``logo``
package (https://www.terrame.org/package/logo/models/#ants).

Usage
-----
    streamlit run examples/streamlit/abm_ants.py
"""
from __future__ import annotations

from matplotlib.colors import ListedColormap
import streamlit as st

from dissmodel.core import Environment
from dissmodel.visualization import Map, Chart
from dissmodel_abm.models import AntsModel, build_colony, DISPLAY_CODES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ants", layout="centered")
st.title("Ants (dissmodel-abm)")
st.caption(
    "TerraME logo: Ants — "
    "https://www.terrame.org/package/logo/models/#ants"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

dim         = st.sidebar.slider("Grid dimension", min_value=10, max_value=60, value=25)
n_food      = st.sidebar.slider("Food cells", min_value=5, max_value=200, value=40)
n_ants      = st.sidebar.slider("Number of ants", min_value=1, max_value=200, value=25)
steps       = st.sidebar.slider("Simulation steps", min_value=1, max_value=500, value=150)

st.sidebar.subheader("Pheromone dynamics")
evaporation_rate = st.sidebar.slider("Evaporation rate", 0.0, 0.9, 0.2, 0.05)
deposit_amount   = st.sidebar.slider("Deposit amount", 0.5, 20.0, 5.0, 0.5)
strong_threshold = st.sidebar.slider("Strong-trail threshold", 0.1, 10.0, 1.0, 0.1)

seed = st.sidebar.number_input("Random seed", min_value=0, value=0, step=1)
run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
gdf = build_colony(dimension=dim, n_food_cells=n_food, seed=seed)

env = Environment(start_time=0, end_time=steps)

model = AntsModel(
    gdf=gdf,
    n_ants=n_ants,
    evaporation_rate=evaporation_rate,
    deposit_amount=deposit_amount,
    strong_threshold=strong_threshold,
    seed=seed,
)

# AntsModel.collected / .foraging are registered via @track_plot, so
# Chart() below picks them up automatically — same convention as
# dissmodel-sysdyn's SIR model.

# ---------------------------------------------------------------------------
# Visualization: map (empty=tan, weak trail=dark green, trail=green,
# food=blue, nest=red, ant=gold) + collected/foraging chart
# ---------------------------------------------------------------------------
# Colors ordered by DISPLAY_CODES (empty, trail_weak, trail, food, nest,
# ant). Plotting the numeric "display_code" column with fixed vmin/vmax
# keeps colors stable across steps, even as trails and food appear and
# disappear.
cmap = ListedColormap(["tan", "darkgreen", "limegreen", "dodgerblue", "crimson", "gold"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Colony")
    map_area = st.empty()
with col2:
    st.subheader("Foraging over time")
    chart_area = st.empty()

Map(
    gdf=gdf,
    plot_params={
        "column": "display_code",
        "cmap": cmap,
        "vmin": 0,
        "vmax": len(DISPLAY_CODES) - 1,
        "markersize": 12,
    },
    plot_area=map_area,
)

Chart(
    plot_area=chart_area,
    show_legend=True,
    show_grid=True,
    title="Ants",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    env.reset()
    env.run()
