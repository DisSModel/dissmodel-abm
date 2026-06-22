"""
Peripherisation Model — Streamlit
====================================
The Peripherisation Model (Barros and Alves Jr., 2003), from Joana
Barros' PhD thesis "Urban Growth in Latin American Cities" (UCL, 2004).

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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Peripherisation Model", layout="centered")
st.title("Peripherisation Model (dissmodel-abm)")
st.caption(
    "Barros and Alves Jr. (2003) — Joana Barros' PhD thesis, "
    "\"Urban Growth in Latin American Cities\" (UCL, 2004)"
)
st.markdown(
    "Three economic groups settle on a grid, all preferring proximity to "
    "**red** (high-income) cells. **Red** can settle anywhere, **yellow** "
    "(middle-income) anywhere except on red, and **blue** (low-income) only "
    "on empty cells — producing the core-periphery pattern typical of Latin "
    "American urban segregation."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

dim       = st.sidebar.slider("Grid dimension", min_value=11, max_value=61, value=31, step=2)
n_agents  = st.sidebar.slider("Number of agents", min_value=10, max_value=2000, value=900)
steps     = st.sidebar.slider("Walk steps per settlement attempt", min_value=1, max_value=10, value=2)
agents_per_step = st.sidebar.slider("Agents processed per tick", min_value=1, max_value=50, value=10)

st.sidebar.subheader("Proportion of agents per group")
pct_red    = st.sidebar.slider("Red (high income) %", 1, 50, 10)
pct_yellow = st.sidebar.slider("Yellow (middle income) %", 1, 80, 40)
pct_blue   = max(0, 100 - pct_red - pct_yellow)
st.sidebar.caption(f"Blue (low income): {pct_blue}%")

end_time = st.sidebar.slider("Max simulation ticks", min_value=100, max_value=5000, value=1500, step=100)
seed = st.sidebar.number_input("Random seed", min_value=0, value=3, step=1)

run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
gdf = vector_grid(dimension=(dim, dim), resolution=1)

env = Environment(start_time=0, end_time=end_time)

proportions = (pct_red / 100, pct_yellow / 100, pct_blue / 100)

model = PeripherisationModel(
    gdf=gdf,
    steps=steps,
    proportions=proportions,
    n_agents=n_agents,
    agents_per_step=agents_per_step,
    seed=seed,
)

# PeripherisationModel.red / .yellow / .blue / .pending are registered
# via @track_plot, so Chart() below picks them up automatically — same
# convention as dissmodel-sysdyn's SIR model.

# ---------------------------------------------------------------------------
# Visualization: map (white=empty, red, yellow, blue) + population chart
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
        st.info(f"{model.pending} agents still pending — increase max ticks to let them settle.")
