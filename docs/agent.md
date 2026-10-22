# Agents, societies and space in dissmodel-abm

This page describes the pieces of `dissmodel-abm` 0.2, the TerraME
semantics they keep, and where they differ. The TerraME behavior described
here was read from the source of the `base` package
([TerraME/terrame](https://github.com/TerraME/terrame), `packages/base/lua`).

## The model

`AgentModel` is a DisSModel `Model`. Write `setup()` (TerraME's `init`) to
build the space, the societies and the placement, and `execute()` to say
what happens in each step:

```python
class PredatorPrey(AgentModel):
    def setup(self, dim=30):
        self.cs = CellSpace.grid(dim, state=PASTURE, count=0)
        self.cs.create_neighborhood()
        self.rabbits = Society(Rabbit, quantity=20)
        self.wolves = Society(Wolf, quantity=20)
        self.place(self.rabbits, self.wolves, cellspace=self.cs)

    def execute(self):
        self.regrow()              # a cell rule, in NumPy
        self.rabbits.execute()
        self.wolves.execute()
```

**Time.** TerraME runs events at `t = 1, ..., finalTime`.
`Environment(start_time=0, end_time=n)` runs `n + 1` steps, so use
`Environment(start_time=1, end_time=final_time)` to match the number of
steps of a TerraME model.

**Randomness.** `AgentModel(seed=...)` creates `self.rng`, a
`numpy.random.Generator`. Societies and cell spaces created inside
`setup()` take it automatically, and agents reach it as `agent.rng`. Draw
every random number from it (`self.rng.random()`, `self.rng.integers(n)`),
never from `numpy.random` or `random`, and two runs with the same seed and
code give the same bands. With `seed=None`, the seed actually used is kept
in `model.seed_used`.

**Bands computed from cells.** A TerraME cell attribute defined as a
function (`Cell{state = function(cell) ... end}`) becomes an agent band or a
derived band (see below). The model recomputes both after `setup()` and
after each step (`post_execute`). A subclass that overrides `post_execute`
must call `super().post_execute()`.

## CellSpace and Cell

`CellSpace.grid(xdim, ydim=None, **bands)` creates a regular space; extra
keyword arguments create bands filled with a value. `CellSpace(backend)`
uses an existing `RasterBackend`, such as one read from a GeoTIFF or built
by DisSCube; cells outside `backend.nodata_mask` do not exist (`get` returns
`None`, iteration skips them, and they are never neighbors).

A `Cell` reads and writes the bands at its position: `cell.cover`,
`cell.chemical += 1`. Writing a band that does not exist raises
`AttributeError`; create it first with `cs.add_band(name, value)`. TerraME
lets any attribute be created on the fly, but a misspelled name there
silently creates a new attribute.

Coordinates follow TerraME: `x` is the column, `y` the row, both from 0,
and `cs.get(x, y)` returns the cell. The band `n_agents` is kept by every
cell space with the number of agents in each cell.

**Agent bands and derived bands.** Bands that show the agents, for drawing
or for cell rules, are declared once:

```python
# PredatorPrey: who is in each cell
cs.add_agent_band("who", lambda cell: WOLF if cell.agent.name == "wolf" else RABBIT,
                  empty=EMPTY)
# Ants: the land cover, with the ants drawn over it
cs.add_agent_band("view", lambda cell: ANT, background="cover")
```

`add_agent_band` calls the function only for the cells that have agents and
fills the rest with `empty` (or copies the band `background`) in NumPy, so
it costs the same on a 50 × 50 grid and on a 3000 × 3000 one.
`add_derived_band(name, fn)` calls `fn` for every cell, in Python: use it
only for small spaces (it takes about 8 s per step at 1000 × 1000).

**Neighborhood.** `cs.create_neighborhood(strategy="moore", wrap=False,
include_self=False)`, with TerraME's defaults; `strategy="vonneumann"` gives
the four cells that share an edge. `cell.neighborhood` lists the neighbors
in TerraME's order (rows from `dy = -1` to `1`, columns from `dx = -1` to
`1`), so a loop that stops at the first match behaves like
`forEachNeighbor` with `return false`.

## Cell rules and cellular automata

Rules that apply to every cell, such as pasture regrowth or evaporation,
are best written as in `dissmodel-ca`: with NumPy over the whole band, in
the model's `execute()` (as `PredatorPrey.regrow` and `Ants.evaporate`), or
as a `dissmodel.geo.RasterCellularAutomaton` that shares the backend:

```python
backend = RasterBackend(shape=(100, 100))
fire = FireModel(backend=backend)          # dissmodel-ca, unchanged
brigade = Brigade(backend=backend)         # an AgentModel with CellSpace(backend)
```

Both models read and write the same arrays; in each step the environment
runs them in the order they were created, and each one sees what the
previous one wrote. `RasterCellularAutomaton` replaces the arrays of the
bands it returns at every step; cells look the array up on each access, so
they always see the current one. `examples/cli/fire_brigade.py` runs
firefighters against the fire of `dissmodel-ca` this way.

## Cost on large grids

The cost of a step follows the number of agents and the cells around them,
not the size of the grid: `Cell` objects are created only when an agent or
the model reaches them, and placement (`model.place`), `cs.sample()`,
`cs.sample_empty()`, `cs.count_empty()` and agent bands work on NumPy
indices. With 1000 agents walking on a 1000 × 1000 grid, setup takes
0.04 s and a step about 18 ms (Python 3.11).

Two things still visit every cell in Python and are meant for small spaces:
iterating the space (`for cell in cs`, `Trajectory`, `cs.empty_cells()`) and
`add_derived_band`. `cs.n_cells_created` tells how many cells a model has
created, which is a quick way to find such a loop.

## Agent

An agent type is a class. Class attributes are the default state of every
new agent; `init()` runs once when the agent joins a society; `execute()`
runs once per step.

| Method | TerraME | Behavior |
| --- | --- | --- |
| `agent.cell` | `getCell()` | The cell, or `None`. |
| `enter(cell)` | `enter` | Puts the agent in the cell. If it already has one, warns and moves it, as TerraME warns. |
| `leave()` | `leave` | Takes the agent out of its cell; it stays in its society. |
| `move(cell)` | `move` | `leave()` then `enter(cell)`. |
| `walk()` | `walk` | Moves to a random neighbor, empty or not. |
| `walk_if_empty()` | `walkIfEmpty` | Draws a random neighbor and moves only if it is empty. |
| `walk_to_empty()` | `walkToEmpty` | Moves to a random empty neighbor, if any. |
| `empty_neighbor()` | `emptyNeighbor` | A random empty neighbor, or `None`. |
| `neighbor_agents()` | `forEachNeighborAgent` | Agents in the neighboring cells. |
| `die()` | `die` | Leaves the cell and the society; the rest of the step skips it. |
| `reproduce(**attrs)` | `reproduce` | New agent of the same type, from the type's defaults, in the parent's cell. |

As in TerraME, `move` and `enter` do not check whether the cell is empty,
and `reproduce` puts the child in the parent's cell even if that leaves two
agents there. Models with one agent per cell test `cell.is_empty()` and
move the child, as in `child.move(empty_cell)`.

`reproduce` creates the child from the type, not as a copy of the parent:
class defaults, then `attrs`, then `init()`. This is what TerraME does
(`self.parent:add(data)`), and it matters: in `PredatorPrey` a wolf's child
starts with 40 energy, not with the parent's.

## Society

`Society(AgentType, quantity=n, **attrs)` creates `n` agents with `attrs`
and runs `init()` on each. Agents get ids 1, 2, 3, ... in creation order.

`society.execute(order="insertion")` runs the agents alive at the start of
the step, in creation order (TerraME's default), skipping agents that die
during the step; agents born during the step act from the next one.
`order="random"` draws a new order at every step from the model's generator.

TerraME differs in one corner case: its loop runs as many iterations as
there were agents at the start, so when agents die during a step, the same
number of newborns at the end of the list may still act in the step they
were born.

Other methods: `add`, `remove`, `sample`, `select`, `count`, `call` (call a
method on every agent, like TerraME's summary functions), `agents`,
`to_geodataframe` (one point per agent at its cell center, using the
backend's `transform` and `crs`).

## Placement

`model.place(*societies, cellspace=cs, strategy="random", max=1)`
follows TerraME's `createPlacement`:

- `"random"` (default): shuffles the agents and the cells and fills the
  cells in order, `max` agents each. Fails if there are more agents than
  `len(cs) * max`.
- `"uniform"`: cells in row order, one agent each, wrapping around.
- `"spread"`: each agent picks a random cell among those with fewer than
  `max` agents.
- `"void"`: agents stay without a cell until they `enter` one.

Only one placement per agent is supported for now; TerraME allows several
named placements.

## Group and Trajectory

`Group(society, select=fn, sort=key, random=False)` and
`Trajectory(cs, select=fn, sort=key, random=False)` are filtered subsets of
agents and of cells. As in TerraME, a subset is built from its target when it
is created and does not follow changes by itself:

- `filter()` applies `select` again to the **current members only**. An
  agent that left the group does not come back, and agents of the society
  outside the group are not looked at (TerraME `Group:filter`). A group of
  unhappy agents filtered every step therefore empties, even if some agents
  become unhappy again after leaving it; this is how `Schelling` stops.
- `rebuild()` builds the subset again from the target, as if it were created
  again (TerraME `rebuild`).
- `sort()` sorts by the `sort` key, `randomize()` shuffles, `sample()` draws
  one member.

A `Group` can also take another `Group` as target. `filter()` without
`select`, `sort()` without `sort`, and `sort` together with `random` raise
`ValueError`, as in TerraME. `filter()` also drops agents that died.

## Social networks and messages

`society.create_social_network(...)` gives every agent a social network,
following TerraME's `createSocialNetwork`. The strategy is inferred from
the arguments when omitted (`quantity`, `probability` or `filter`), or
given explicitly:

| Strategy | Connections of each agent |
| --- | --- |
| `"quantity"` | `quantity` other agents drawn at random |
| `"probability"` | each other agent with `probability` |
| `"function"` | the agents `b` for which `filter(a, b)` is true |
| `"cell"` | the other agents in the same cell (`include_self` adds itself) |
| `"neighbor"` | the agents in the neighboring cells |
| `"erdos"` | `quantity` random undirected links in the whole society |
| `"barabasi"` | preferential attachment: `start` seed agents, `quantity` links per later agent |
| `"watts"` | ring of `quantity` neighbors, each link rewired with `probability` |
| `"void"` | none; fill with `agent.get_social_network().add(other)` |

With `inmemory=False` (first five strategies), the network is a function
called at every `agent.get_social_network()`: new random contacts, or the
agents in the current cell, every time. `symmetric=True` (for `"quantity"`
and `"probability"` in memory) adds the reverse of each link. Several
networks can coexist under different `name`s.

A `SocialNetwork` keeps `(agent, weight)` pairs in insertion order, keyed by
the agent itself, so it may join agents of different societies (TerraME
keys by `id`, which repeats across societies). `agent.connections(name)`
lists the connected agents, like `forEachConnection`.

**Messages.** `agent.message(receiver, delay=0, subject=None, **content)`:

- With `delay=0`, the receiver's handler runs now: `on_<subject>(msg)` when
  a subject is given, `on_message(msg)` otherwise; `message` returns what
  the handler returns.
- With `delay > 0`, the message waits in the sender's society.
  `society.execute()` first calls `society.synchronize()`, which subtracts
  one step (the model's `step`) from each waiting message and delivers
  those that reach zero, then runs the agents, as TerraME does for
  `Event{action = society}`. A message sent with `delay=1` in step `t` is
  received at the start of step `t + 1`, and the receiver can act on it in
  that same step.
- `msg.sender`, `msg.receiver`, `msg.subject` and `msg.content` describe the
  message; extra keyword arguments can also be read as attributes
  (`msg.value`).
- A message to an agent that has died is dropped (TerraME warns and
  ignores it); sending one with `delay=0` warns.

## Not yet available

- A space of polygons (vector), with the neighborhoods of
  `dissmodel.geo.vector`.
- Several named placements, and neighborhoods other than Moore and von
  Neumann (TerraME's `mxn`, `function`, `coord`).
- State machines (`State`, `Jump`, `Flow`).
