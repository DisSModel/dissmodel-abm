"""
Schelling segregation model — CLI example
===========================================
Ported from TerraME's ``logo`` package
(https://www.terrame.org/package/logo/models/#schelling).

Two agent types are randomly distributed over a grid with a fraction of
empty cells. Each step, unhappy agents (fewer than ``preference``
same-type Queen-neighbors) move to a random empty cell. The system
typically converges to a segregated configuration.

Usage
-----
    python examples/cli/abm_schelling.py
"""
from __future__ import annotations

from matplotlib.colors import ListedColormap

from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel.visualization import Map
from dissmodel_abm.models import SchellingModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DIM = 25            # grid is DIM x DIM cells (TerraME default)
FREE_SPACE = 0.25    # fraction of empty cells (TerraME default: 25%)
PREFERENCE = 3       # min. same-type neighbors to be "happy" (TerraME default)
END_TIME = 30

gdf = vector_grid(dimension=(DIM, DIM), resolution=1)

env = Environment(start_time=0, end_time=END_TIME)

model = SchellingModel(gdf=gdf, free_space=FREE_SPACE, preference=PREFERENCE, seed=0)

# ---------------------------------------------------------------------------
# Visualization
# -1 = empty (white), 0 = "red" agents, 1 = "blue" agents
# ---------------------------------------------------------------------------
cmap = ListedColormap(["white", "tab:red", "tab:blue"])
Map(
    gdf=gdf,
    plot_params={
        "column": "agent_type",
        "cmap": cmap,
        "vmin": -1,
        "vmax": 1,
        "ec": "gray",
    },
    pause=False,        # set True for an interactive matplotlib window
    save_frames=True,   # writes one PNG per step to ./map_frames/
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"t=0   satisfied = {model.fraction_satisfied():.3f}")
env.run()
print(f"t=end satisfied = {model.fraction_satisfied():.3f}")
