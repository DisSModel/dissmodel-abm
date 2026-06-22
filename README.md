# dissmodel-abm

**Agent-Based Modeling (ABM) extension for [dissmodel](https://github.com/DisSModel/dissmodel)**,
inspired by TerraME's [`Agent`](https://www.terrame.org/base/types/agent/) /
`Society` API.

This is a **standalone, additive package** — it does not modify dissmodel's
core. It depends on `dissmodel` as a regular package dependency, exactly like
[`dissmodel-ca`](https://github.com/DisSModel/dissmodel-ca).

## `Society` / `Agent`: a protective layer over the substrate

The whole point of this package is that **model code never touches
`self.gdf` directly**. Today the substrate is a vector `GeoDataFrame`
(raster support is planned next), but a model written against
`self.society` doesn't know or care about that — it only sees agents as
objects:

```python
for agent in self.society:
    if agent.energy <= 0:
        agent.die()
    elif agent.energy >= 15:
        agent.reproduce(energy=5.0)
```

instead of:

```python
self.gdf = self.gdf[self.gdf["energy"] > 0].reset_index(drop=True)
mask = self.gdf["energy"] >= 15
children = self.gdf.loc[mask].copy()
children["energy"] = 5.0
self.gdf = gpd.GeoDataFrame(pd.concat([self.gdf, children]), crs=self.gdf.crs)
```

`Society` owns no data of its own — it reads and writes through the host
model's `gdf` attribute, so `model.gdf` and `model.society` are always
the same data. `Map`, `Chart`, `ModelExecutor`, and any code that still
expects `model.gdf` keep working unmodified, whether or not a given
model uses `self.society` internally.

## Concept mapping (TerraME → dissmodel-abm)

| TerraME (Agent / Society)        | dissmodel-abm                                        |
| ---------------------------------- | ------------------------------------------------------ |
| `execute(self)`                   | `execute()` (Model lifecycle)                          |
| `init(self)`                      | `setup()` (Model lifecycle)                             |
| `Society` (collection of Agents)  | `self.society` — object-oriented view over `self.gdf`  |
| `Agent`                            | `self.society[idx]` — a proxy over one row             |
| `placement` / `getCell()`         | `agent.geometry`                                        |
| `enter(cell)`                      | `agent.enter(x, y)` — give a location-less agent a position |
| `leave()`                           | `agent.leave()` — remove an agent's position without removing it from its society |
| `move(cell)` / `walk()`            | `agent.move_to(x, y)` / `agent.walk(step_size, bounds)` |
| `die()`                             | `agent.die()`                                           |
| `reproduce()`                       | `agent.reproduce(**overrides)`                          |
| `emptyNeighbor` / neighborhood     | `agent.neighbors(radius)` (points) or `agent.grid_neighbors()` (cells) |
| `Society:add` / `Society:remove`   | `society.add(**attrs)` / `society.remove(agent_or_idx)` |
| `Society:sample`                    | `society.sample(n)`                                     |
| `forEachAgent`                      | `for agent in society: ...`                             |

Point-agent models (`RandomWalkModel`, `PredatorPreyModel`) back
`self.society` with Point geometry + state columns. One-agent-per-cell
models (`SchellingModel`) back it with a polygon grid
(`dissmodel.geo.vector.vector_grid`), matching the design of TerraME's
[`logo`](https://github.com/TerraME/logo) package — *"Implements spatial
agent-based models with at most one agent per cell"*.

A small set of legacy batch/vectorized methods (`walk`, `die_if`,
`reproduce_if`, `neighbors_within`) is still available directly on
`AgentModel` for backward compatibility and for cases where vectorized
performance matters more than per-agent readability — see the API
reference below. New models should prefer `self.society`.

### Agents without a location

Following TerraME — where an `Agent` may exist with no `placement` at
all until `Agent:enter()` is called — an agent here can exist with
`geometry = None`:

```python
orphan = self.society.add(energy=4.0)   # no position yet
orphan.has_location                     # False

orphan.enter(10, 10)                    # give it a position
orphan.has_location                     # True

orphan.leave()                          # remove the position, keep the agent
orphan.has_location                     # False
```

Calling a spatial method (`walk`, `move_to` excepted — `move_to`/`enter`
are exactly how a location-less agent gets a position — `neighbors`,
`distance_to`) on an agent with no location raises a clear
`RuntimeError` rather than failing on `None` deep inside geopandas.
`agent.reproduce()` of a location-less agent yields a location-less
child, unless `geometry` is given in the overrides.

This is useful for agents that aren't always spatially situated — e.g.
a population still "off-map" before being placed, or an agent that
temporarily leaves the simulated space.

## Installation

```bash
pip install -e .
```

Requires `dissmodel>=0.6.0` (installs `geopandas`, `shapely`, `numpy`,
`pandas`, `libpysal` as transitive dependencies).

## Quick start

```python
import geopandas as gpd
import numpy as np
from dissmodel.core import Environment
from dissmodel_abm.models import RandomWalkModel

n = 20
bounds = (0, 0, 100, 100)
rng = np.random.default_rng(42)
gdf = gpd.GeoDataFrame({
    "geometry": gpd.points_from_xy(
        rng.uniform(bounds[0], bounds[2], n),
        rng.uniform(bounds[1], bounds[3], n),
    )
})

env = Environment(start_time=0, end_time=20)
model = RandomWalkModel(gdf=gdf, step_size=2.0, bounds=bounds)
env.run()
```

## Writing your own model

```python
from dissmodel_abm.core import AgentModel

class MyModel(AgentModel):
    def setup(self, **params):
        ...  # one-time initialization

    def execute(self):
        for agent in self.society:
            agent.walk(step_size=1.0, bounds=(0, 0, 100, 100))
            if agent.energy <= 0:
                agent.die()
            elif agent.energy >= 15:
                agent.reproduce(energy=5.0)
```

`Society` also provides batch helpers when you don't need a Python
`for` loop over every agent: `society.select(predicate)`,
`society.remove_if(predicate)`, `society.count(predicate)`,
`society.sample(n)`, and `society.walk_all(step_size, bounds)` (the
vectorized equivalent of looping `agent.walk(...)` over everyone).

To expose a live chart in Streamlit (or any `dissmodel.visualization.Chart`),
decorate the class with `@track_plot` and update the matching attribute
(same name, lowercased) inside `execute()` — no separate tracker model
needed:

```python
from dissmodel.visualization import track_plot

@track_plot("Population", "tab:blue")
class MyModel(AgentModel):
    population: int = 0

    def execute(self):
        ...
        self.population = len(self.society)
```

## Repository structure

```
dissmodel-abm/
├── src/dissmodel_abm/
│   ├── core/
│   │   ├── agent_model.py   # AgentModel(SpatialModel) — no core changes
│   │   └── society.py       # Society / Agent — the protective layer
│   └── models/
│       ├── random_walk.py        # minimal example (point agents)
│       ├── predator_prey.py      # wolf-sheep, written against self.society
│       ├── schelling.py          # segregation model, one agent per cell
│       └── peripherisation.py    # Barros & Alves Jr. (2003) urban-growth model
├── examples/
│   ├── cli/
│   │   ├── abm_random_walk.py
│   │   ├── abm_predator_prey.py
│   │   ├── abm_schelling.py
│   │   └── abm_peripherisation.py
│   └── streamlit/
│       ├── abm_random_walk.py
│       ├── abm_predator_prey.py
│       ├── abm_schelling.py
│       └── abm_peripherisation.py
├── docs/
│   └── agent.md
├── tests/
│   └── test_agent_model.py
└── pyproject.toml
```

## Included models

| Model               | Substrate    | Live-plotted attributes | Source                                                        | Description                                            |
| ------------------- | ------------ | ------------------------ | --------------------------------------------------------------| ------------------------------------------------------- |
| `RandomWalkModel`    | Vector (points) | —                       | —                                                            | Independent random walk within a bounding box.          |
| `PredatorPreyModel`  | Vector (points) | `sheep`, `wolves`       | —                                                            | Wolf-sheep dynamics: movement, predation, death, reproduction, optional grazing. |
| `SchellingModel`     | Vector (grid cells) | `satisfaction`      | [TerraME `logo`: Schelling](https://www.terrame.org/package/logo/models/#schelling) | Segregation model: agents move to empty cells when unhappy with same-type neighbor count. |
| `PeripherisationModel` | Vector (grid cells) | `red`, `yellow`, `blue`, `pending` | [Barros & Alves Jr. (2003)](http://www.dpi.inpe.br/gilberto/cursos/st-society/barros-phd-thesis.pdf) | Urban-growth segregation model: three income groups settle on a grid, all preferring proximity to high-income cells, producing a core-periphery pattern. |

### Schelling model (ported from TerraME `logo`)

```python
from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel_abm.models import SchellingModel

gdf = vector_grid(dimension=(25, 25), resolution=1)
env = Environment(start_time=0, end_time=30)

model = SchellingModel(gdf=gdf, free_space=0.25, preference=3, seed=0)
env.run()

print(model.fraction_satisfied())  # -> 1.0 once converged
```

Internally, every agent is reached through `self.society` —
`agent.agent_type`, `agent.grid_neighbors()` — never raw `gdf` rows.
Parameters match TerraME's defaults (`dim=25`, `freeSpace=25%`,
`preference=3`). `agent_type`: `-1` = empty, `0`/`1` = the two agent
groups. `model.fraction_satisfied()` returns the fraction of agents with
at least `preference` same-type Queen-neighbors.

### Peripherisation Model (Barros & Alves Jr., 2003)

```python
from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel_abm.models import PeripherisationModel

gdf = vector_grid(dimension=(31, 31), resolution=1)
env = Environment(start_time=0, end_time=1500)

model = PeripherisationModel(
    gdf=gdf, steps=2, n_agents=900,
    proportions=(0.10, 0.40, 0.50),  # red, yellow, blue
    seed=3,
)
env.run()

print(model.red, model.yellow, model.blue, model.is_done())
```

Ported from Joana Barros' PhD thesis *"Urban Growth in Latin American
Cities: Exploring urban dynamics through agent-based simulation"* (UCL,
2004) — originally implemented in StarLogo, later in FORTRAN, RePast
(Java), and a TerraME/Lua dialect. This is a from-scratch port of the
*original* StarLogo rule set (Barros & Alves Jr., 2003), not a
translation of the Lua dialect, which implements a different
region/density-based rule set on top of the same general idea.

Population is split into three economic groups — `red` (high income,
minority), `yellow` (middle income), `blue` (low income, majority) — all
sharing the same locational preference (proximity to red/high-income
cells, representing infrastructure), but differing in their economic
power to displace others: `red` can settle anywhere (evicting whoever is
there), `yellow` can settle anywhere except on `red`, and `blue` can
only settle on empty cells. Each pending agent performs a biased random
walk of length `steps` (the model's central parameter) toward the
centroid of existing red cells before attempting to settle; evicted
agents re-enter the pool and try again. This produces the
**core-periphery** pattern central to the thesis — red clustered near
the seed, yellow forming a ring around it, and blue pushed to the
periphery — the inverse of the classic Burgess concentric-ring model,
matching the income gradient observed in Latin American cities (see
thesis chapter 6 and Barros & Alves Jr., 2003, section 4).

`model.is_done()` reports whether every agent has settled;
`model.occupied_fraction()` returns the fraction of the grid currently
occupied.

## Documentation

[`docs/agent.md`](docs/agent.md) — a full `Agent` reference page,
mirroring the structure of
[TerraME's own `Agent` docs](https://www.terrame.org/base/types/agent/),
comparing every function side-by-side (`die`, `enter`, `leave`, `move`,
`walk`, `reproduce`, ...) with worked examples from `PredatorPreyModel`
and `SchellingModel`, plus a list of TerraME functions with no
dissmodel-abm equivalent yet (social networks, state machines).

## Roadmap

- **Raster substrate**: a `Society` backed by a NumPy/raster array
  instead of a `GeoDataFrame`, exposing the exact same `Agent` /
  `Society` interface described above, so existing model code (written
  against `self.society`) keeps working unchanged regardless of
  substrate. Vector support is intentionally being hardened first.
- **Social networks**: TerraME's `addSocialNetwork` / `message` /
  `on_message`, likely as a thin layer over a graph library (e.g.
  `networkx`) keyed by `agent.id`.
- **State machines**: TerraME's `State` / `Jump` / `Flow`, for agents
  whose behavior depends on a discrete internal state.

## Running the examples

```bash
python examples/cli/abm_random_walk.py
python examples/cli/abm_predator_prey.py
python examples/cli/abm_schelling.py        # also writes PNG frames to ./map_frames/
python examples/cli/abm_peripherisation.py
```

## Streamlit explorers

Each model has an interactive Streamlit app under `examples/streamlit/`,
following the same `Map` + `Chart` convention used by
[`dissmodel-ca`](https://github.com/DisSModel/dissmodel-ca) and
[`dissmodel-sysdyn`](https://github.com/DisSModel/dissmodel-sysdyn):
sidebar sliders control parameters, a live `Map` shows the spatial
state, and (where relevant) a live `Chart` shows tracked variables over
time. `Chart` requires no per-app wiring — it reads whatever the model
exposes via `@track_plot` (`PredatorPreyModel.sheep` / `.wolves`,
`SchellingModel.satisfaction`, `PeripherisationModel.red` / `.yellow` /
`.blue` / `.pending`), the exact convention used by `dissmodel-sysdyn`'s
`SIR` model.

```bash
pip install -e ".[viz]"

streamlit run examples/streamlit/abm_random_walk.py
streamlit run examples/streamlit/abm_predator_prey.py   # Map + population Chart
streamlit run examples/streamlit/abm_schelling.py        # Map + satisfaction Chart
streamlit run examples/streamlit/abm_peripherisation.py  # Map + population-by-group Chart
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## `Society` / `Agent` API reference

**`self.society`** (a `Society`, available on any `AgentModel`):

- `len(society)` — number of agents.
- `for agent in society: ...` — iterate over all agents.
- `society[idx]` / `idx in society` — get an agent by id / check existence.
- `society.add(geometry=None, **attrs)` — create a new agent; returns it.
- `society.remove(agent_or_idx)` — remove one agent.
- `society.remove_if(predicate)` — remove every agent matching `predicate(agent)`; returns the count removed.
- `society.select(predicate)` — return the list of agents matching `predicate(agent)`.
- `society.sample(n)` — `n` agents chosen uniformly at random.
- `society.count(predicate=None)` — count agents, optionally matching `predicate`.
- `society.walk_all(step_size, bounds=None)` — vectorized random walk for every agent (use when you don't need a per-agent loop).
- `society.neighbors_within(idx, radius)` — agents within `radius` of agent `idx` (point agents).
- `society.grid_neighbors_of(idx)` — topological neighbors of cell `idx` (requires `create_neighborhood(...)`).

**`Agent`** (what you get from `society[idx]` or by iterating `society`):

- `agent.id` — the agent's index in the underlying data.
- `agent.<column>` / `agent.<column> = value` — read/write any attribute; reflects immediately in `model.gdf`.
- `agent.geometry` — the agent's position; any `shapely` geometry (Point, Polygon, ...), or `None` if it has no location.
- `agent.has_location` — whether this agent currently has a position.
- `agent.enter(x, y)` / `agent.enter(geometry)` — give a location-less agent a position for the first time; accepts coordinates or any geometry directly.
- `agent.leave()` — remove this agent's position, without removing it from its society.
- `agent.move_to(x, y)` / `agent.move_to(geometry)` — move to an absolute position; accepts coordinates (a `Point` shortcut) or any geometry directly (the equivalent of TerraME's `Agent:move(cell)`).
- `agent.walk(step_size, bounds=None)` — move by a random offset around the geometry's centroid; always produces a `Point` (requires a location).
- `agent.distance_to(other_agent)` — Euclidean distance (requires both agents to have a location).
- `agent.neighbors(radius)` — nearby agents (point agents), as `Agent` objects (requires a location).
- `agent.grid_neighbors()` — topological cell neighbors, as `Agent` objects.
- `agent.die()` — remove this agent from its society.
- `agent.reproduce(**overrides)` — create a child at the same position, copying attributes and applying `overrides`.

Do not hold onto `Agent` objects across a step that may remove or add
agents elsewhere — re-obtain them via `society[idx]` or by iterating
`society` again within the current step.

## Legacy `AgentModel` methods (batch/vectorized)

Kept for backward compatibility; equivalent functionality is available
through `self.society` above.

- `walk(step_size, bounds=None, mask=None)` — random displacement in x/y for every agent.
- `move_to(idx, x, y)` — move a single agent to an absolute position.
- `die_if(condition)` — remove agents for which `condition(row)` is True (resets the GeoDataFrame index).
- `reproduce_if(condition, child_fn=None)` — duplicate agents for which `condition(row)` is True.
- `neighbors_within(idx, radius)` — indices of agents within `radius` of agent `idx`.
- `all_neighbors_within(radius)` — `{idx: [neighbor_idx, ...]}` for every agent.

For **one-agent-per-cell** models (e.g. `SchellingModel`), use the
inherited `SpatialModel.create_neighborhood(strategy=Queen, use_index=True)`
and `neighs_id(idx)` (or `agent.grid_neighbors()`) — these operate on the
grid topology rather than on point distances.

All other `SpatialModel` / `Model` functionality (`self.env`,
`create_neighborhood`, `pre_execute` / `post_execute`, plot tracking, the
`ModelExecutor` / `ExperimentRecord` pipeline, `dissmodel.io`,
`dissmodel.visualization`) is inherited unchanged and works exactly as in
`dissmodel-ca`.

## License

MIT
