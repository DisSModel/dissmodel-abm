"""
Peripherisation Model — Streamlit
====================================
The Peripherisation Model (Barros and Alves Jr., 2003), ported from the
TerraME/Lua implementation.  Settlement is density-based: high and middle
income settle on the cell closest to the existing core; low income is
restricted to low-density peripheral cells and banned from the dense core.

Usage
-----
    streamlit run examples/streamlit/abm_peripherisation.py
"""
from __future__ import annotations

from matplotlib.colors import ListedColormap
import streamlit as st

from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel.visualization import Map, Chart
from dissmodel_abm.models import PeripherisationModel

st.set_page_config(page_title="Peripherisation Model", layout="centered")
st.title("Peripherisation Model (dissmodel-abm)")
st.caption("Barros and Alves Jr. (2003) — Lua/TerraME variant")
st.markdown(
    "**Red** (high) and **yellow** (middle income) settle on the empty cell "
    "closest to the existing settlement core. **Blue** (low income) is "
    "restricted to low-density peripheral cells and expelled from the dense "
    "core — producing the concentric-ring pattern observed in Latin American cities."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

dim             = st.sidebar.slider("Grid dimension", 11, 61, 31, step=2)
n_agents        = st.sidebar.slider("Number of agents", 10, 2000, 900)
agents_per_step = st.sidebar.slider("Agents processed per tick", 1, 50, 5)

st.sidebar.subheader("Density thresholds")
low_dens    = st.sidebar.slider("Low-density threshold (blue max)", 0.1, 0.9, 0.4, step=0.05)
centro_dens = st.sidebar.slider("Centro threshold (blue ban above)", 0.1, 0.9, 0.9, step=0.05)

st.sidebar.subheader("Proportion of agents per group")
pct_red    = st.sidebar.slider("Red (high income) %", 1, 50, 10)
pct_yellow = st.sidebar.slider("Yellow (middle income) %", 1, 80, 40)
pct_blue   = max(0, 100 - pct_red - pct_yellow)
st.sidebar.caption(f"Blue (low income): {pct_blue}%")

end_time = st.sidebar.slider("Max simulation ticks", 100, 5000, 1500, step=100)
seed     = st.sidebar.number_input("Random seed", min_value=0, value=3, step=1)

run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
gdf = vector_grid(dimension=(dim, dim), resolution=1)
env = Environment(start_time=0, end_time=end_time)
proportions = (pct_red / 100, pct_yellow / 100, pct_blue / 100)

model = PeripherisationModel(
    gdf=gdf,
    proportions=proportions,
    n_agents=n_agents,
    agents_per_step=agents_per_step,
    low_density_threshold=low_dens,
    centro_density_threshold=centro_dens,
    seed=seed,
)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
cmap = ListedColormap(["white", "tab:red", "gold", "tab:blue"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Spatial distribution")
    map_area = st.empty()
with col2:
    st.subheader("Population over time")
    chart_area = st.empty()

Map(
    gdf=gdf,
    plot_params={
        "column": "group",
        "cmap": cmap,
        "vmin": -1,
        "vmax": 2,
        "ec": "lightgray",
        "linewidth": 0.2,
    },
    plot_area=map_area,
)

Chart(
    plot_area=chart_area,
    show_legend=True,
    show_grid=True,
    title="Population by group",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    env.reset()
    env.run()
    if model.is_done():
        st.success(f"All {n_agents} agents settled.")
    else:
        st.info(f"{model.pending} agents still pending — increase max ticks.")
