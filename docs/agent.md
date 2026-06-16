# Agent

*dissmodel-abm — compared against [TerraME's `Agent`](https://www.terrame.org/base/types/agent/)*

An object representing one individual in a [`Society`](#society). In
**dissmodel-abm**, an `Agent` is a thin proxy over one row of the
underlying `GeoDataFrame`: reading or writing an attribute
(`agent.energy`, `agent.energy = 5`) reads or writes that row directly,
with no copy and no separate state to keep in sync. Agents are obtained
from a [`Society`](#society) — never constructed directly.

This page mirrors the structure of TerraME's `Agent` reference page,
function by function, so the comparison is easy to follow. Functions
TerraME has and dissmodel-abm does not are listed at the end, under
[Not implemented](#not-implemented).

---

## Arguments

TerraME's `Agent{...}` constructor takes a table of attributes and
functions (`execute`, `id`, `init`, `on_message`, ...). dissmodel-abm
has no equivalent constructor — agents are rows added to a `Society`
via [`society.add()`](#society-add), and their behavior lives in the
*model*, not in the agent itself.

| | TerraME | dissmodel-abm |
|---|---|---|
| Where behavior lives | In the `Agent` table itself (`execute`, `on_message`, ...) | In the `Model.execute()` method of the model that owns the `Society` |
| Creating an agent | `Agent{size = 10, execute = function(self) ... end}` | `society.add(size=10)` — behavior is defined once, in the model |
| Unique identifier | `id` (string, assigned by the `Society`) | `agent.id` (the GeoDataFrame index, assigned by the `Society`) |

This is the central structural difference between the two: TerraME
agents are **autonomous objects** that carry their own behavior.
dissmodel-abm agents are **data with a uniform interface** — behavior
lives once in the model's `execute()`, applied identically to every
agent in the loop. This trades TerraME's per-agent heterogeneous
behavior for simplicity and for staying close to a vectorizable
substrate (a GeoDataFrame).

---

## Attributes

| TerraME | dissmodel-abm | Notes |
|---|---|---|
| `id` | `agent.id` | Both: unique, assigned by the Society/collection |
| `parent` | *(not exposed)* | The agent doesn't know which `Society`/model it belongs to; only the model knows its `society` |
| `placement` | `agent.geometry` | TerraME's placement is a `Trajectory`; here it's directly the row's geometry |
| `cells` | *(not applicable)* | dissmodel-abm agents have at most one position, never several |
| `socialnetworks` | *(not implemented)* | See [Not implemented](#not-implemented) |
| `state_` | *(not applicable)* | No state-machine layer (see [Not implemented](#not-implemented)) |
| `cObj_` | *(not applicable)* | Implementation detail specific to TerraME's C++ core |

---

## Usage

**TerraME:**

```lua
singleFooAgent = Agent{
    size = 10,
    name = "foo",
    execute = function(self)
        self.size = self.size + 1
        self:walk()
    end,
    on_hello = function(self, m)
        self:message{receiver = m.sender, content = "hi"}
    end
}
```

**dissmodel-abm:**

```python
from dissmodel_abm.core import AgentModel

class FooModel(AgentModel):
    def setup(self):
        pass  # self.society already holds the agents passed in via gdf=

    def execute(self):
        for agent in self.society:
            agent.size = agent.size + 1
            agent.walk()
```

The TerraME snippet defines one agent's behavior inline. The
dissmodel-abm snippet defines the *model's* behavior once, applied to
every agent in `self.society` each step — the loop body is what plays
the role of TerraME's `execute = function(self) ... end`.

---

## Functions

| | TerraME | dissmodel-abm |
|---|---|---|
| [add](#add) | Add a Trajectory or State to the Agent | *(not implemented)* |
| [addSocialNetwork](#addsocialnetwork) | Add a SocialNetwork to the Agent | *(not implemented)* |
| [die](#die) | Kill the agent and remove it from its Society | [`agent.die()`](#die_1) |
| [emptyNeighbor](#emptyneighbor) | Return an empty neighbor Cell | *(not implemented as a method — see [Schelling](#worked-example-schelling))* |
| [enter](#enter) | Put the Agent into a Cell | [`agent.enter(x, y)`](#enter_1) |
| [execute](#execute) | Entry point for executing the Agent | `Model.execute()` (model-level, not agent-level) |
| [getCell](#getcell) | Return the Cell where the Agent is located | [`agent.geometry`](#getcell_1) |
| [getCells](#getcells) | Return all Cells pointed by the Agent | *(not applicable — single position only)* |
| [getLatency](#getlatency) | Time of the last State transition | *(not applicable — no State machine)* |
| [getSocialNetwork](#getsocialnetwork) | Return a named SocialNetwork | *(not implemented)* |
| [getStateName](#getstatename) | Current State name | *(not applicable — no State machine)* |
| [getTrajectoryStatus](#gettrajectorystatus) | Status of the Agent's Trajectories | *(not applicable)* |
| [init](#init) | Initialize the Agent on entering a Society | `Model.setup()` (model-level) |
| [leave](#leave) | Remove the Agent from its current Cell | [`agent.leave()`](#leave_1) |
| [message](#message) | Send a message to another Agent | *(not implemented)* |
| [move](#move) | Move the Agent to a new Cell | [`agent.move_to(x, y)`](#move_1) |
| [notify](#notify) | Notify the Agent's Observers | `dissmodel.visualization.Chart` / `Map` (model-level, via `@track_plot`) |
| [on\_message](#on_message) | Handle a received message | *(not implemented)* |
| [reproduce](#reproduce) | Create an Agent with the same behavior in the same Cell | [`agent.reproduce(**overrides)`](#reproduce_1) |
| [sample](#sample) | Random Agent from a SocialNetwork | [`society.sample(n)`](#society-sample) *(samples the whole Society, not a SocialNetwork)* |
| [setTrajectoryStatus](#settrajectorystatus) | Activate/deactivate the Agent's Trajectories | *(not applicable)* |
| [walk](#walk) | Random walk to a neighbor Cell | [`agent.walk(step_size, bounds)`](#walk_1) |
| [walkIfEmpty](#walkifempty) | Move to a random neighbor Cell only if empty | *(not implemented)* |
| [walkToEmpty](#walktoempty) | Walk to one of the available empty neighbor Cells | *(not implemented as a method — see [Schelling](#worked-example-schelling))* |

---

## die

Kill the agent and remove it from the Society it belongs to.

**TerraME:**

```lua
agent = Agent{
    execute = function(self)
        if self.energy <= 0 then
            agent:die()
        end
    end
}
```

**dissmodel-abm:**

```python
def execute(self):
    for agent in self.society:
        if agent.energy <= 0:
            agent.die()
```

Or, for the common "remove everyone matching a condition" case, the
batch form avoids the explicit loop:

```python
self.society.remove_if(lambda agent: agent.energy <= 0)
```

**Comparison:** equivalent semantics. TerraME's `die()` also cleans up
any placements (`enter`/`leave`/`move`); dissmodel-abm's `die()` simply
removes the row, which already takes the (single) geometry with it.

---

## enter

Put the Agent into a Cell — TerraME assumes an Agent can be in one and
only one Cell at a time, and `enter()` is required before `move()` or
`walk()` can be used.

**TerraME:**

```lua
soc = Society{instance = Agent{}, quantity = 30}
cs = CellularSpace{xdim = 10}
env = Environment{soc, cs}
env:createPlacement{strategy = "void"}

agent = soc:sample()
agent:enter(cs:sample())
```

**dissmodel-abm:**

```python
orphan = self.society.add(energy=4.0)   # created with no location
orphan.has_location                     # False
orphan.enter(10, 10)                    # give it a position
orphan.has_location                     # True
```

**Comparison:** equivalent semantics, ported faithfully. In TerraME,
`createPlacement{strategy = "void"}` is what makes the initial
"location-less" state explicit; in dissmodel-abm, `society.add(...)`
without a `geometry` argument does the same thing. See
[Agents without a location](#agents-without-a-location).

---

## leave

Remove the Agent from its current Cell, without removing the Agent from
its Society.

**TerraME:**

```lua
ag1 = Agent{}
cs = CellularSpace{xdim = 3}
myEnv = Environment{cs, ag1}
myEnv:createPlacement()
ag1:leave()
```

**dissmodel-abm:**

```python
agent.leave()
agent.has_location   # False — still in the society, just unplaced
```

**Comparison:** equivalent. TerraME raises an error if the Agent has no
Cell to leave; dissmodel-abm simply sets `geometry = None` (calling
`leave()` twice is harmless, it stays `None`).

---

## move

Move the Agent to a new Cell.

**TerraME:**

```lua
ag = soc:sample()
cell = cs:sample()
ag:move(cell)
```

**dissmodel-abm:**

```python
agent.move_to(x, y)        # coordinate shortcut — creates a Point
agent.move_to(some_polygon) # or: pass any shapely geometry directly
```

**Comparison:** equivalent. TerraME's `Cell` can be any geometry the
`CellularSpace` defines (typically a polygon, but not necessarily);
`move_to` accepts either two coordinates (a convenience shortcut that
builds a `Point`) or any `shapely` geometry directly — a `Polygon`, a
`LineString`, or whatever the model needs — exactly mirroring
`Agent:move(cell)`, which takes the whole `Cell` object rather than
coordinates. `enter(x, y)` accepts the same arguments.

---

## walk

Random walk.

**TerraME:**

```lua
singleFooAgent = Agent{}
cs = CellularSpace{xdim = 10}
cs:createNeighborhood()
e = Environment{cs, singleFooAgent}
e:createPlacement()

singleFooAgent:walk()
```

**dissmodel-abm:**

```python
agent.walk(step_size=1.0, bounds=(0, 0, 100, 100))
```

For walking every agent in a step (the common case), use the vectorized
form instead of a Python loop:

```python
self.society.walk_all(step_size=1.0, bounds=(0, 0, 100, 100))
```

**Comparison:** different mechanics, same intent. TerraME's `walk()`
moves to a random **neighbor Cell** in a discrete `Neighborhood` (one
hop on the grid graph). dissmodel-abm's `walk()` moves by a continuous
random offset around the agent's centroid, clipped to `bounds`, and
**always becomes a `Point`** — even if the agent previously had a
Polygon geometry. This makes `walk()` a point-agent operation by
design: use it for point-agent models (`RandomWalkModel`,
`PredatorPreyModel`); for grid/cell models,
swap attributes between cells instead (see
[Schelling](#worked-example-schelling)).

Raises `RuntimeError` if the agent has no location — see
[Agents without a location](#agents-without-a-location).

---

## reproduce

Create an Agent with the same behavior, in the same Cell as the
original.

**TerraME:**

```lua
agent = Agent{}
soc = Society{instance = agent, quantity = 100}

soc.agents[1]:reproduce()
print(#soc)   -- 101
```

**dissmodel-abm:**

```python
parent = self.society[0]
child = parent.reproduce(energy=5.0)   # copies all attrs, then overrides energy
len(self.society)                      # one more than before
```

**Comparison:** equivalent semantics. Both create a new individual at
the same position, added to the same collection, returning the new
individual. dissmodel-abm's `**overrides` plays the role of TerraME's
optional attribute table argument to `reproduce()`. If the parent has no
location (see [below](#agents-without-a-location)), the child is also
created without one, unless `geometry` is given in `overrides`.

---

## execute / init

**TerraME** defines these per-Agent: `execute` describes one Agent's
behavior each step; `init` runs once when the Agent enters a Society.

**dissmodel-abm** defines these per-Model, applied uniformly to every
agent in `self.society`:

```python
class MyModel(AgentModel):
    def setup(self):          # ~ TerraME Agent:init()
        ...

    def execute(self):        # ~ TerraME Agent:execute(), but for the
        for agent in self.society:   #   whole Society at once
            ...
```

**Comparison:** same lifecycle moments, different granularity. TerraME
can give each Agent in a Society different `execute` functions (a
Society's agents need not be identical); dissmodel-abm's `execute()` is
one function applied identically to every agent via the loop —
heterogeneous per-agent behavior would have to be expressed as
branching inside that loop (`if agent.kind == "wolf": ... else: ...`,
as in [`PredatorPreyModel`](#worked-example-predator-prey)).

---

## notify

**TerraME:**

```lua
agent = Agent{value = 1}
Chart{target = agent}
agent:notify(1)
```

**dissmodel-abm:**

```python
@track_plot("Population", "tab:blue")
class MyModel(AgentModel):
    population: int = 0

    def execute(self):
        ...
        self.population = len(self.society)
```

Any `dissmodel.visualization.Chart` connected to the same `Environment`
plots `population` automatically.

**Comparison:** different shape, same intent. TerraME's `notify()` is
called per-Agent and a `Chart` can target an individual Agent directly.
dissmodel-abm's tracking is model-level: a model exposes an attribute
via `@track_plot`, and that's what gets plotted — there is no
"observe this one agent's value over time" equivalent, since per-agent
Observers aren't implemented.

---

## Geometry is not limited to points

A question that comes up naturally: in TerraME, a `Cell` can be any
geometry the `CellularSpace` defines — usually a polygon, but the
framework doesn't enforce that. Is `agent.geometry` here limited to
points, since `move_to` takes coordinates?

No — `agent.geometry` is whatever `shapely` geometry is stored in that
row, full stop. `SchellingModel` already proves this in production:
every agent's `geometry` is a `Polygon` (one grid cell), and `agent.die()`,
`agent.reproduce()`, `agent.agent_type` work identically regardless of
geometry type, because none of those operations ever assume a shape.

```python
from shapely.geometry import Polygon

cell = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
agent.move_to(cell)            # move_to also accepts a geometry directly
agent.geometry.geom_type        # 'Polygon'
```

The one method that *is* point-specific by design is `walk()`: it
always produces a `Point` (a random offset from the geometry's
centroid), because "random walk in continuous space" only makes sense
for point agents. For polygon/cell agents, the TerraME-equivalent of
moving is swapping an attribute between cells (see
[Schelling](#worked-example-schelling)) rather than calling `walk()`.

---

## Agents without a location

TerraME's Agent can exist with no `placement` until `Agent:enter()` is
called. dissmodel-abm follows the same model: an agent's `geometry` may
be `None`.

```python
orphan = self.society.add(energy=4.0)   # no geometry given
orphan.has_location                     # False
orphan.geometry                         # None

orphan.enter(10, 10)
orphan.has_location                     # True
```

Calling a spatial method (`walk`, `neighbors`, `distance_to`) on an
agent with no location raises a clear `RuntimeError` rather than
failing deep inside geopandas:

```python
orphan.leave()
orphan.walk()
# RuntimeError: Agent 7 has no location; cannot walk. Call agent.enter(x, y) first.
```

---

## Worked example: Predator-Prey

A side-by-side reading of [`PredatorPreyModel`](../src/dissmodel_abm/models/predator_prey.py)
against the TerraME functions it uses:

```python
def execute(self) -> None:
    society = self.society

    # 1. walk() — every agent takes a random step
    for agent in society:
        agent.walk(step_size=self.step_size, bounds=self.bounds)

    # 2. plain attribute access — no TerraME equivalent function needed
    for agent in society:
        if self.graze_gain and agent.kind == "sheep":
            agent.energy += self.graze_gain
        agent.energy -= self.energy_loss

    # 3. neighbors() — TerraME emptyNeighbor()-style spatial query
    self._predation_step(society)

    # 4. remove_if() — batch die()
    society.remove_if(lambda agent: agent.energy <= 0)

    # 5. reproduce() — same signature and intent as TerraME's
    for agent in society.select(lambda a: a.energy >= self.reproduce_threshold):
        agent.reproduce(energy=self.reproduce_threshold / 2.0)
```

```python
def _predation_step(self, society) -> None:
    wolves = society.select(lambda a: a.kind == "wolf")
    eaten: set = set()
    for wolf in wolves:
        sheep_nearby = [
            prey for prey in wolf.neighbors(self.eat_radius)   # spatial query
            if prey.id not in eaten and prey.kind == "sheep"
        ]
        if sheep_nearby:
            prey = sheep_nearby[0]
            eaten.add(prey.id)
            wolf.energy += self.energy_gain
    for prey_id in eaten:
        society.remove(prey_id)                                 # ~ die()
```

Every line above maps to a named TerraME `Agent` function in the table
above (`walk`, `neighbors`≈`emptyNeighbor`, `die`, `reproduce`) — none of
it touches the underlying GeoDataFrame directly.

---

## Worked example: Schelling

[`SchellingModel`](../src/dissmodel_abm/models/schelling.py) is the
one-agent-per-cell case TerraME's
[`logo`](https://github.com/TerraME/logo) package targets. It doesn't
have a `walkToEmpty()`/`emptyNeighbor()` method to call — that logic is
written out explicitly using `grid_neighbors()` and a manual scan for an
empty cell, since dissmodel-abm doesn't (yet) provide those two
TerraME functions as reusable `Agent` methods:

```python
def execute(self) -> None:
    society = self.society
    type_map = {agent.id: agent.agent_type for agent in society}
    empty_idx = [idx for idx, t in type_map.items() if t == EMPTY]

    for idx in order:
        t = type_map[idx]
        neighs = self.neighs_id(idx)          # ~ Cell:getNeighborhood()
        same = sum(1 for nb in neighs if type_map.get(nb, EMPTY) == t)
        if same < self.preference and empty_idx:
            target = empty_idx.pop(rng.integers(len(empty_idx)))
            type_map[idx], type_map[target] = EMPTY, t   # ~ walkToEmpty()
```

If `emptyNeighbor()` / `walkToEmpty()` get added as real `Agent` methods
later, this loop body is exactly what they would wrap.

---

## Not implemented

These TerraME `Agent` functions have no dissmodel-abm equivalent today:

| Function | What it does in TerraME | Why it's missing here |
|---|---|---|
| `addSocialNetwork` | Attach a SocialNetwork to the Agent | No social-network layer yet (see roadmap in the project README) |
| `getSocialNetwork` | Retrieve a named SocialNetwork | Same |
| `message` / `on_message` | Send/receive messages between Agents | Same |
| `sample` *(SocialNetwork)* | Random Agent from a SocialNetwork | `society.sample(n)` exists but samples the whole Society, not a relational subset |
| `add` *(State/Trajectory)* | Attach a State machine or Trajectory | No State-machine layer (`State`/`Jump`/`Flow`) |
| `getStateName` / `getLatency` | Introspect the current State | Same |
| `getTrajectoryStatus` / `setTrajectoryStatus` | Enable/disable Trajectories | Same |
| `emptyNeighbor` / `walkIfEmpty` / `walkToEmpty` | One-agent-per-cell movement helpers | Logic exists (see [Schelling](#worked-example-schelling)) but isn't exposed as reusable `Agent` methods yet |
| `getCells` | Multiple Cells per Agent | dissmodel-abm agents have at most one position |

---

*See also: [`Society`](https://github.com/DisSModel/dissmodel-abm) reference
(dissmodel-abm) · TerraME [`Society`](https://www.terrame.org/base/types/society/),
[`Cell`](https://www.terrame.org/base/types/cell/),
[`SocialNetwork`](https://www.terrame.org/base/types/socialNetwork/).*
