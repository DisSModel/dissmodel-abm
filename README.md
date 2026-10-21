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
  in one operation; `RasterMap` draws them; a `RasterBackend` read from a
  GeoTIFF or built by DisSCube can be the space.

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
| `Cell{state = function(cell) ... end}` | `cs.add_derived_band("state", fn)` |
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
| `Group{target = soc, select = f}`, `group:filter()` | `Group(soc, select=f)`, `group.filter()` |
| `Trajectory{target = cs, select = f}` | `Trajectory(cs, select=f)` |
| `soc:myFunction()` (on every agent) | `soc.call("my_function")` |

[`docs/agent.md`](docs/agent.md) describes each piece, the semantics kept
from TerraME and the places where this package differs.

## Models ported from TerraME `logo`

Five of the eleven models of [`logo`](https://github.com/TerraME/logo)
(version 0.5.6), written to follow the Lua code rule by rule, with the
original defaults:

| Model | What it exercises | Original `finalTime` |
| --- | --- | --- |
| `SingleAgent` | `walkToEmpty` | 100 |
| `Schelling` | `Group`, `Trajectory`, wrapped neighborhood, one move per step | 500 |
| `PredatorPrey` | two societies, `reproduce`, `die`, a cell rule (pasture regrowth) | 500 |
| `LifeCycle` | births and deaths by age, agents sharing a cell | 100 |
| `Ants` | `void` placement, `enter`, bands written by agents, evaporation | 450 |

```bash
python examples/cli/run_logo.py schelling --seed 3
streamlit run examples/streamlit/logo_explorer.py
```

### How the ports were checked

TerraME and NumPy draw different random numbers, so a run cannot be
compared with TerraME value by value. The tests check the rules
(invariants such as at most one agent per cell, one move per step in
Schelling, pasture regrowth) and that the same seed gives the same bands.
Where `logo` keeps a reference chart from its own tests (`log/*-chart-1.bmp`),
the port was run over several seeds and compared with it:

| Model | Reference chart (`logo`) | Port (median over seeds) |
| --- | --- | --- |
| `PredatorPrey`, t = 20 | about 858 rabbits, 17 wolves | 857 rabbits, 16 wolves (10 seeds) |
| `Schelling`, t = 0 | 18.4 % unhappy | 18.1 % (10 seeds) |
| `LifeCycle`, t = 40 | about 91 agents | 94.5 (20 seeds) |
| `Ants`, first food taken | t = 8 | t = 8 in 14 of 50 seeds, never earlier |
| `Ants`, food left at t = 40 | 97 | 90 (50 seeds; 97 or more in 2) |

Two findings came out of this comparison:

- The `LifeCycle` chart in `logo` (2016) was made with an earlier birth
  rule that placed children only in empty neighbors, and it levels off near
  100 agents. The current code lets a child share its parent's cell, so the
  population is not capped. The port follows the current code.
- In `Ants`, the port takes more food by step 40 than the single reference
  run. Whether this is chance or a difference (the order in which TerraME
  2.0 visited neighbors is the main suspect) needs several runs of the
  original, for example in the `terrame-docker` image.

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
│   ├── society.py     # Society, Group, Trajectory
│   └── model.py       # AgentModel: seed, placement, derived bands
└── models/            # ports of TerraME logo
examples/
├── cli/run_logo.py
└── streamlit/logo_explorer.py
tests/
├── test_core.py
└── test_models.py
```

## Roadmap

- Vector space: `CellSpace` over polygons of a `GeoDataFrame`, with the
  neighborhoods of `dissmodel.geo.vector`.
- Several named placements per agent (TerraME's `placement` argument).
- Social networks and messages (`SocialNetwork`, `message`, `on_message`)
  and the remaining `logo` models (`Disease`, `Sugarscape`, `Heatbugs`,
  `SpatialPD`, `Labyrinth`, `GrowingSociety`, `Overpopulation`).
- Runs of the original models in `terrame-docker`, over many seeds, to
  compare distributions instead of single charts.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
