"""
Random Walk — CLI example
==========================
Agents perform an independent random walk within a bounding box.

Usage
-----
    python examples/cli/abm_random_walk.py
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np

from dissmodel.core import Environment
from dissmodel_abm.models import RandomWalkModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
N_AGENTS = 20
BOUNDS = (0, 0, 100, 100)

rng = np.random.default_rng(42)
xs = rng.uniform(BOUNDS[0], BOUNDS[2], N_AGENTS)
ys = rng.uniform(BOUNDS[1], BOUNDS[3], N_AGENTS)

gdf = gpd.GeoDataFrame({"geometry": gpd.points_from_xy(xs, ys)})

env = Environment(start_time=0, end_time=20)

model = RandomWalkModel(gdf=gdf, step_size=2.0, bounds=BOUNDS)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
env.run()

print(model.gdf.head())
