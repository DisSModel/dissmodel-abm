"""
Peripherisation Model — CLI example
=====================================
The Peripherisation Model (Barros and Alves Jr., 2003), from Joana
Barros' PhD thesis on urban growth in Latin American cities.

Three economic groups (red=high income, yellow=middle, blue=low) settle
on a grid, all preferring proximity to red cells but differing in their
power to displace others: red can settle anywhere, yellow can settle
anywhere except on red, blue only on empty cells. This produces the
core-periphery pattern characteristic of Latin American urban
segregation: red clusters near the seed, yellow forms a ring around it,
and blue is pushed to the outer periphery.

Usage
-----
    python examples/cli/abm_peripherisation.py
"""
from __future__ import annotations

from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel_abm.models import PeripherisationModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DIM = 31              # grid is DIM x DIM cells
N_AGENTS = 900         # total agents to place (thesis tests several sizes)
STEPS = 2              # biased random-walk length per settlement attempt
PROPORTIONS = (0.10, 0.40, 0.50)  # red, yellow, blue (thesis default)
AGENTS_PER_STEP = 10   # how many pending agents try to settle each tick
END_TIME = 1000

gdf = vector_grid(dimension=(DIM, DIM), resolution=1)

env = Environment(start_time=0, end_time=END_TIME)

model = PeripherisationModel(
    gdf=gdf,
    steps=STEPS,
    proportions=PROPORTIONS,
    n_agents=N_AGENTS,
    agents_per_step=AGENTS_PER_STEP,
    seed=3,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"t=0    pending={N_AGENTS:4d}  red=1  yellow=0  blue=0")
env.run()
print(
    f"t=end  pending={model.pending:4d}  "
    f"red={model.red:4d}  yellow={model.yellow:4d}  blue={model.blue:4d}  "
    f"done={model.is_done()}"
)
