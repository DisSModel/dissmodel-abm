"""
Predator-Prey (Wolf-Sheep) — CLI example
=========================================
Classic agent-based predator-prey model using dissmodel-abm.

Sheep graze (slowly regain energy via ``graze_gain``), wolves hunt sheep
within ``eat_radius``. Both species lose energy each step and die when
energy reaches zero; agents reproduce once their energy crosses
``reproduce_threshold``.

Usage
-----
    python examples/cli/abm_predator_prey.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np

from dissmodel.core import Environment
from dissmodel_abm.models import PredatorPreyModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
N_SHEEP = 60
N_WOLVES = 8
BOUNDS = (0, 0, 50, 50)

rng = np.random.default_rng(7)
n_total = N_SHEEP + N_WOLVES
xs = rng.uniform(BOUNDS[0], BOUNDS[2], n_total)
ys = rng.uniform(BOUNDS[1], BOUNDS[3], n_total)

gdf = gpd.GeoDataFrame({
    "geometry": gpd.points_from_xy(xs, ys),
    "kind": ["sheep"] * N_SHEEP + ["wolf"] * N_WOLVES,
    "energy": [12.0] * n_total,
})

env = Environment(start_time=0, end_time=20)

model = PredatorPreyModel(
    gdf=gdf,
    step_size=2.0,
    bounds=BOUNDS,
    eat_radius=3.0,
    energy_loss=0.5,
    energy_gain=6.0,
    reproduce_threshold=15.0,
    graze_gain=0.7,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"t=0    sheep={N_SHEEP:3d}  wolves={N_WOLVES:3d}  total={n_total:3d}")
env.run()
