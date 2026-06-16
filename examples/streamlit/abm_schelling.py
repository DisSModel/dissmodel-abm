"""
Schelling Segregation Model — Streamlit
==========================================
Ported from TerraME's ``logo`` package
(https://www.terrame.org/package/logo/models/#schelling).

Usage
-----
    streamlit run examples/streamlit/abm_schelling.py
"""
from __future__ import annotations

from matplotlib.colors import ListedColormap
import streamlit as st

from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel.visualization import Map, Chart
from dissmodel_abm.models import SchellingModel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Schelling Segregation", layout="centered")
st.title("Schelling Segregation Model (dissmodel-abm)")
st.caption(
    "TerraME logo: Schelling — "
    "https://www.terrame.org/package/logo/models/#schelling"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

dim         = st.sidebar.slider("Grid dimension", min_value=5, max_value=80, value=25)
free_space  = st.sidebar.slider("Free space (%)", min_value=5, max_value=80, value=25) / 100
preference  = st.sidebar.slider("Preference (min. same-type neighbors)", min_value=0, max_value=8, value=3)
steps       = st.sidebar.slider("Simulation steps", min_value=1, max_value=200, value=30)
seed        = st.sidebar.number_input("Random seed", min_value=0, value=0, step=1)

run = st.button("Run Simulation")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
gdf = vector_grid(dimension=(dim, dim), resolution=1)

env = Environment(start_time=0, end_time=steps)

model = SchellingModel(
    gdf=gdf,
    free_space=free_space,
    preference=preference,
    seed=seed,
)

# SchellingModel.satisfaction is registered via @track_plot, so Chart()
# below picks it up automatically — same convention as
# dissmodel-sysdyn's SIR model.

# ---------------------------------------------------------------------------
# Visualization: map (-1=empty white, 0=red, 1=blue) + satisfaction chart
# ---------------------------------------------------------------------------
cmap = ListedColormap(["white", "tab:red", "tab:blue"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Spatial distribution")
    map_area = st.empty()
with col2:
    st.subheader("Satisfaction over time")
    chart_area = st.empty()

Map(
    gdf=gdf,
    plot_params={
        "column": "agent_type",
        "cmap": cmap,
        "vmin": -1,
        "vmax": 1,
        "ec": "gray",
    },
    plot_area=map_area,
)

Chart(
    plot_area=chart_area,
    show_legend=False,
    show_grid=True,
    title="Fraction satisfied",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    env.reset()
    env.run()
