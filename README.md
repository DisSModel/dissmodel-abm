# dissmodel-abm

**TerraME-style agent-based modeling for [DisSModel](https://github.com/DisSModel/dissmodel).**

Agents are Python objects, grouped in societies and placed in a cellular
space whose attributes are the bands of a DisSModel `RasterBackend`. The
concepts and names follow TerraME (`Agent`, `Society`, `CellularSpace`,
`createPlacement`, `walkToEmpty`, `reproduce`, `Group`, `Trajectory`), with
three adaptations to Python and to the DisSModel ecosystem:

- **One clock.** A model is a regular DisSModel `Model`, run by
  `dissmodel.core.Environment`. There is no `Timer`/`Event`: the model's
  `execute()` says what happens in a step.
- **One random generator.** `AgentModel(seed=...)` creates a NumPy
  generator that every society, cell space and agent uses. The same seed
  gives the same run, byte for byte.
- **Cells are raster bands.** Agents read and write cells one at a time
  (`cell.cover = FOOD`); cell rules can update the same bands with NumPy
  in one operation, or run as a `dissmodel-ca` cellular automaton on the
  same backend; `RasterMap` draws them; a `RasterBackend` read from a
  GeoTIFF or built by DisSCube can be the space. The cost of a step follows
  the number of agents, not the size of the grid.

Status: version 0.2.0, alpha. The raster space is the only one for now; a
vector space (polygons from a `GeoDataFrame`) is planned behind the same
interface. Version 0.2.0 replaces the 0.1.0 API (agents as rows of a
`GeoDataFrame`) completely.

## Installation

```bash
pip install -e .            # requires Python 3.11+ and dissmodel 0.6.6+
pip install -e ".[viz]"     # matplotlib and streamlit, for the examples
```

## A model in a few lines

```python
from dissmodel.core import Environment
from dissmodel_abm import Agent, AgentModel, CellSpace, Society


class Walker(Agent):
    age = 0                           # default state of every new agent

    def execute(self):                # once per step
        self.walk_to_empty()
        self.age += 1
        if self.rng.random() < 0.01:  # the model's generator, never numpy.random
            self.die()


class Walkers(AgentModel):
    def setup(self, dim=20, quantity=50):
        self.cs = CellSpace.grid(dim)
        self.cs.create_neighborhood()                     # Moore, as in TerraME
        self.society = Society(Walker, quantity=quantity)
        self.place(self.society, cellspace=self.cs)       # random, one per cell

    def execute(self):
        self.society.execute()


env = Environment(start_time=1, end_time=100)   # 100 steps, as finalTime = 100
model = Walkers(seed=42)
env.run()
print(len(model.society))
```

`model.cs.backend.arrays["n_agents"]` holds the number of agents in each
cell after every step, ready for `RasterMap(backend=model.cs.backend, band="n_agents")`.

## Concepts

| TerraME | dissmodel-abm |
| --- | --- |
| `Model{init = function(model) ... end}` | `class M(AgentModel)` with `setup()` and `execute()` |
| `Timer{Event{action = soc}}` | `self.society.execute()` inside `execute()`; `Environment` is the clock |
| `random = true`, `Random():number()` | `AgentModel(seed=...)`, `self.rng`, `agent.rng` |
| `Agent{energy = 40, execute = ...}` | `class Rabbit(Agent): energy = 40; def execute(self): ...` |
| `init = function(agent)` | `def init(self)` |
| `Society{instance = ag, quantity = n}` | `Society(Rabbit, quantity=n)` |
| `CellularSpace{xdim = n}` | `CellSpace.grid(n)` or `CellSpace(raster_backend)` |
| `Cell{state = function(cell) ... end}` | `cs.add_agent_band("state", fn)` (or `add_derived_band`) |
| `cs:createNeighborhood{wrap = true}` | `cs.create_neighborhood(wrap=True)` |
| `Environment{cs, soc}:createPlacement{}` | `self.place(soc, cellspace=cs)` |
| `agent:getCell()` | `agent.cell` |
| `cell:getAgent()`, `cell:isEmpty()` | `cell.agent`, `cell.is_empty()` |
| `forEachNeighbor(cell, f)` | `for neigh in cell.neighborhood: ...` |
| `cell:getNeighborhood():sample()` | `cell.neighborhood.sample()` |
| `agent:move(cell)`, `enter`, `leave` | `agent.move(cell)`, `enter`, `leave` |
| `agent:walk()`, `walkIfEmpty()`, `walkToEmpty()` | `agent.walk()`, `walk_if_empty()`, `walk_to_empty()` |
| `agent:emptyNeighbor()` | `agent.empty_neighbor()` |
| `agent:die()`, `agent:reproduce()` | `agent.die()`, `agent.reproduce()` |
| `Group{target = soc, select = f}`, `group:filter()`, `group:rebuild()` | `Group(soc, select=f)`, `group.filter()`, `group.rebuild()` |
| `Trajectory{target = cs, select = f}` | `Trajectory(cs, select=f)` |
| `soc:myFunction()` (on every agent) | `soc.call("my_function")` |
| `soc:createSocialNetwork{quantity = 6}` | `soc.create_social_network(quantity=6)` |
| `forEachConnection(agent, f)` | `for other in agent.connections(): ...` |
| `agent:message{receiver = b, delay = 1}` | `agent.message(b, delay=1)` |
| `on_message = function(self, msg)` | `def on_message(self, msg)` |

[`docs/agent.md`](docs/agent.md) describes each piece, the semantics kept
from TerraME and the places where this package differs.

## Models ported from TerraME `logo`

Six of the twelve models of [`logo`](https://github.com/TerraME/logo)
(version 0.5.6), written to follow the Lua code rule by rule, with the
original defaults:

| Model | What it exercises | Original `finalTime` |
| --- | --- | --- |
| `SingleAgent` | `walkToEmpty` | 100 |
| `Schelling` | `Group`, `Trajectory`, wrapped neighborhood, one move per step | 500 |
| `PredatorPrey` | two societies, `reproduce`, `die`, a cell rule (pasture regrowth) | 500 |
| `LifeCycle` | births and deaths by age, agents sharing a cell | 100 |
| `Ants` | `void` placement, `enter`, bands written by agents, evaporation | 450 |
| `Disease` | social network redrawn every step, messages with delay; no space | 30 |

```bash
python examples/cli/run_logo.py schelling --seed 3
streamlit run examples/streamlit/logo_explorer.py
```

`examples/cli/fire_brigade.py` shows agents and a cellular automaton
together: firefighters against the forest fire of
[`dissmodel-ca`](https://github.com/DisSModel/dissmodel-ca), on one shared
backend.

### How the ports were checked

TerraME and NumPy draw different random numbers, so a run cannot be
compared with TerraME value by value. Two kinds of tests check the ports:

- **Rules** (`tests/test_models.py`): invariants such as at most one agent
  per cell, one move per step in Schelling, pasture regrowth, and that the
  same seed gives the same bands. The mean population of `LifeCycle`, a
  branching process, is checked against its exact expectation.
- **TerraME itself** (`tests/test_goldens.py`, marker `golden`): the models
  of `logo` 0.5.6 were run in TerraME 2.0.1 for seeds 1 to 50
  ([`terrame-goldens`](https://github.com/LambdaGeo/terrame-goldens)), and
  20 runs of each port are compared with them at a few steps, with a
  Kolmogorov-Smirnov test and a test on the means
  (`tests/goldens/` holds the extract and how it was made).

| Model | Variable, step | TerraME (50 seeds) | Port (20 seeds) |
| --- | --- | --- | --- |
| `Schelling` | unhappy agents (%), t = 500 | 5.6 ± 0.8 | 5.6 ± 1.0 |
| `PredatorPrey` | rabbits, t = 20 | 855 ± 6 | 859 ± 7 |
| `PredatorPrey` | wolves, t = 20 | 15.8 ± 2.0 | 15.2 ± 1.9 |
| `LifeCycle` | agents, t = 40 | 101 ± 16 | 92 ± 12 (exact mean: 97.8) |
| `Ants` | food left, t = 40 | 90.6 ± 3.1 | 90.3 ± 4.0 |
| `Disease` | never infected, t = 30 | 2.7 ± 1.8 | 2.6 ± 1.2 |

What came out of this comparison:

- `Schelling` first stopped with almost no unhappy agents (0.4 % at
  t = 500) where TerraME leaves 5.6 %. The cause was `Group.filter()`, which
  rebuilt the group from the whole society; TerraME's `filter` only checks
  the current members again, so an agent that becomes unhappy after leaving
  the group is never moved. `Group` and `Trajectory` now follow TerraME
  (`filter` and `rebuild`, see `docs/agent.md`).
- The differences that the single reference charts of `logo` suggested in
  `Ants` (food left at t = 40: 97 in the chart) and `Disease` (12 to 17
  agents never infected) were chance: over 50 seeds TerraME gives the same
  as the port.
- The `LifeCycle` chart in `logo` (2016) was made with an earlier birth
  rule that placed children only in empty neighbors, and it levels off near
  100 agents. The current code lets a child share its parent's cell, so the
  population is not capped. The port follows the current code, and TerraME
  agrees.

The ports also keep two details of the original that look like slips:
`Schelling` defaults to 5 % free space (the help text says 25), and in
`Ants` the functions `findFood` and `findNest` always return `false`. Both
are documented in the model modules.

## Repository structure

```
src/dissmodel_abm/
├── core/
│   ├── cellspace.py   # CellSpace, Cell, Neighborhood (raster)
│   ├── agent.py       # Agent
│   ├── society.py     # Society, Group, Trajectory, social network strategies
│   ├── network.py     # SocialNetwork, Message
│   └── model.py       # AgentModel: seed, placement, derived bands
└── models/            # ports of TerraME logo
examples/
├── cli/run_logo.py
├── cli/fire_brigade.py   # agents + a dissmodel-ca automaton on one backend
└── streamlit/logo_explorer.py
tests/
├── test_core.py
├── test_network.py
├── test_cellspace_scale.py
└── test_models.py
```

## Roadmap

- Vector space: `CellSpace` over polygons of a `GeoDataFrame`, with the
  neighborhoods of `dissmodel.geo.vector`.
- Several named placements per agent (TerraME's `placement` argument).
- The remaining `logo` models (`Sugarscape`, `Heatbugs`, `SpatialPD`,
  `Labyrinth`, `GrowingSociety`, `Overpopulation`).
- Runs of the original models in `terrame-docker`, over many seeds, to
  compare distributions instead of single charts.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use dissmodel-abm, please cite it with the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this
repository").

## License

MIT; see [`LICENSE`](LICENSE).
