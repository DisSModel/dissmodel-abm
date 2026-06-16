"""
dissmodel_abm/models/labyrinth.py
==================================
Labyrinth, ported from TerraME's ``logo`` package
(https://www.terrame.org/package/logo/models/#labyrinth).

Original Lua source:
https://www.terrame.org/package/logo/lua/Labyrinth.lua

    Labyrinth = Model{
        quantity = 1,
        finalTime = 1000,
        labyrinth = Choice(patterns),
        random = true,
        init = function(model)
            model.cs = getLabyrinth(model.labyrinth)
            model.cs:createNeighborhood()
            ...
            model.agent = Agent{
                execute = function(agent)
                    local empty = {}
                    local exit
                    forEachNeighbor(agent:getCell(), function(neigh)
                        if neigh.state == "exit" then
                            exit = neigh
                        elseif neigh.state == "empty" then
                            table.insert(empty, neigh)
                        end
                    end)
                    if exit then
                        exit.state = "found"
                        agent:leave()
                        agent.execute = function() end
                    else
                        agent:move(Random(empty):sample())
                    end
                end
            }
            ...
        end
    }

Rule
----
A cellular grid represents a maze of ``"wall"``, ``"exit"`` and
``"empty"`` cells. A handful of walker agents start on random empty
cells; each step, every walker looks at its (Queen) neighborhood:

- if a neighbor cell is the ``"exit"``, the walker marks it
  ``"found"`` (TerraME: ``exit.state = "found"``) and leaves the
  simulation (TerraME: ``agent:leave()``, then replacing
  ``agent.execute`` with a no-op);
- otherwise, it moves to a randomly chosen ``"empty"`` neighbor cell
  (TerraME: ``agent:move(Random(empty):sample())``).

A walker with no ``"exit"``/``"empty"`` neighbor at all (boxed in by
walls) simply stays put for the step.

ASCII patterns
---------------
TerraME loads named maze patterns from ``.labyrinth`` files via
``getLabyrinth(model.labyrinth)``; this port embeds a couple of
patterns as plain ASCII grids in :data:`PATTERNS` instead (``#`` wall,
``.`` empty, ``E`` exit) — add new mazes there following the same
convention, or pass a custom ``list[str]`` of equal-length rows to
:func:`build_labyrinth`.

dissmodel-abm mapping
----------------------
Unlike Schelling (one agent per cell, the cell itself is the agent),
Labyrinth has two kinds of agent sharing the same ``self.gdf`` /
``self.society``, distinguished by the ``kind`` column:

- ``kind="cell"``   — one per maze tile (Polygon), mutable ``state``
  (only transition: the exit cell becomes ``"found"`` once reached).
- ``kind="walker"`` — the moving agents (Point, placed at the
  centroid of their current cell), tracked via ``cell_id`` (the maze
  cell's index). Walkers ``die()`` once they reach the exit, the
  equivalent of TerraME's ``agent:leave()``.

Both kinds live in the same GeoDataFrame so a single
:class:`dissmodel.visualization.Map` can render the maze and the
walkers together (mixed Point/Polygon geometry, colored by ``state``).
The maze's cell-to-cell neighborhood (used to decide where each walker
can move) is built once in :meth:`LabyrinthModel.setup`, *before* any
walker row is added — :class:`~dissmodel.geo.vector.SpatialModel`
neighborhood strategies (e.g. ``Queen``) expect a homogeneous polygon
layer.

Live plotting
-------------
``active`` (walkers still searching) and ``found`` (exits reached) are
registered via ``@track_plot``, so any
:class:`dissmodel.visualization.Chart` connected to the same
``Environment`` plots them automatically — no extra wiring needed,
same convention used by ``dissmodel-sysdyn``'s ``SIR`` model.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from libpysal.weights import Queen

from dissmodel.geo.vector import vector_grid
from dissmodel.visualization import track_plot
from dissmodel_abm.core import AgentModel

#: Built-in ASCII maze patterns (``#`` wall, ``.`` empty, ``E`` exit).
#: All rows of a pattern must have equal length; row 0 is the top of
#: the rendered map.
PATTERNS: dict[str, list[str]] = {
    "room": [
        "#############",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "############E",
    ],
    "cross": [
        "#########",
        "#...#...#",
        "#...#...#",
        "#.......#",
        "#...#...#",
        "#...#...#",
        "########E",
    ],
}

#: Fixed numeric code for each ``state`` value, used by the ``state_code``
#: column (see :func:`build_labyrinth`). Plotting on this numeric column
#: with explicit ``vmin=0, vmax=len(STATE_CODES) - 1`` keeps colors stable
#: across steps, regardless of which states are actually present in a
#: given frame -- unlike a plain string column, where geopandas/matplotlib
#: rescale the colormap to whatever categories happen to be in the data.
STATE_CODES: dict[str, int] = {
    "empty": 0,
    "exit": 1,
    "found": 2,
    "wall": 3,
    "walker": 4,
}


def build_labyrinth(
    pattern: str | list[str] = "room",
    resolution: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Build the maze GeoDataFrame consumed by :class:`LabyrinthModel`.

    Equivalent to TerraME's ``getLabyrinth(name)`` — turns a named (or
    custom) ASCII pattern into a polygon grid with one ``"cell"`` agent
    per tile and a ``state`` column in ``{"wall", "exit", "empty"}``.

    Parameters
    ----------
    pattern : str or list of str, optional
        Either a key into :data:`PATTERNS` (default ``"room"``), or a
        custom list of equal-length strings (``#`` wall, ``.`` empty,
        ``E`` exit). Row 0 is the top of the rendered map.
    resolution : float, optional
        Cell size, by default 1.0.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon grid with ``state`` (string), ``state_code`` (numeric,
        see :data:`STATE_CODES`) and ``kind="cell"`` columns, ready to
        pass as ``gdf=`` to :class:`LabyrinthModel`.
    """
    rows = PATTERNS[pattern] if isinstance(pattern, str) else pattern
    n_rows = len(rows)
    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("All rows of a labyrinth pattern must have the same length.")

    grid = vector_grid(dimension=(n_cols, n_rows), resolution=resolution)

    state: dict[str, str] = {}
    for ascii_row, line in enumerate(rows):
        grid_row = n_rows - 1 - ascii_row  # ASCII row 0 = top of the map
        for col, ch in enumerate(line):
            if ch == "#":
                state[f"{grid_row}-{col}"] = "wall"
            elif ch == "E":
                state[f"{grid_row}-{col}"] = "exit"
            else:
                state[f"{grid_row}-{col}"] = "empty"

    grid["state"] = grid.index.map(state)
    grid["state_code"] = grid["state"].map(STATE_CODES)
    grid["kind"] = "cell"
    return grid


@track_plot("Active", "tab:orange")
@track_plot("Found", "tab:green")
class LabyrinthModel(AgentModel):
    """
    Labyrinth (maze escape) model.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Maze grid, from :func:`build_labyrinth` — one ``"cell"`` agent
        per tile, with a ``state`` column in
        ``{"wall", "exit", "empty"}``.
    n_walkers : int, optional
        Number of walker agents to place on random empty cells, by
        default 1 (TerraME's ``quantity``).
    seed : int, optional
        Random seed for the walkers' starting positions, by default
        None.

    Notes
    -----
    Each walker agent (``kind="walker"``) holds ``cell_id``, the index
    of the maze cell it currently occupies.

    Examples
    --------
    >>> from dissmodel.core import Environment
    >>> from dissmodel_abm.models.labyrinth import LabyrinthModel, build_labyrinth
    >>> gdf = build_labyrinth("room")
    >>> env = Environment(end_time=200)
    >>> model = LabyrinthModel(gdf=gdf, n_walkers=5, seed=0)
    >>> env.run()  # doctest: +SKIP
    """

    #: Number of walkers still searching for the exit (tracked for live plotting).
    active: int = 0

    #: Number of exits reached so far (tracked for live plotting).
    found: int = 0

    def setup(self, n_walkers: int = 1, seed: int | None = None) -> None:
        # Neighborhood must be built while self.gdf only holds the
        # (polygon) maze cells -- libpysal's Queen contiguity assumes a
        # homogeneous polygon layer.
        self.create_neighborhood(strategy=Queen, use_index=True)

        if "kind" not in self.gdf.columns:
            self.gdf["kind"] = "cell"

        empty_ids = self.gdf.index[self.gdf["state"] == "empty"].tolist()
        if not empty_ids:
            raise ValueError("Labyrinth pattern has no empty cells to place walkers on.")

        rng = np.random.default_rng(seed)
        start_ids = rng.choice(empty_ids, size=n_walkers, replace=True)

        for start_id in start_ids:
            centroid = self.gdf.at[start_id, "geometry"].centroid
            self.society.add(
                geometry=centroid,
                kind="walker",
                cell_id=start_id,
                state="walker",
                state_code=STATE_CODES["walker"],
            )

        self.active = n_walkers
        self.found = 0

    def execute(self) -> None:
        society = self.society
        walkers = society.select(lambda a: a.kind == "walker")
        if not walkers:
            return

        rng = np.random.default_rng()

        for walker in walkers:
            cell_id = walker.cell_id
            exit_id = None
            empty_ids = []

            for neighbor_id in self.neighs_id(cell_id):
                neighbor_state = self.gdf.at[neighbor_id, "state"]
                if neighbor_state == "exit":
                    exit_id = neighbor_id
                elif neighbor_state == "empty":
                    empty_ids.append(neighbor_id)

            if exit_id is not None:
                self.gdf.at[exit_id, "state"] = "found"
                self.gdf.at[exit_id, "state_code"] = STATE_CODES["found"]
                walker.die()
                self.found += 1
            elif empty_ids:
                new_cell_id = empty_ids[rng.integers(len(empty_ids))]
                walker.cell_id = new_cell_id
                walker.move_to(self.gdf.at[new_cell_id, "geometry"].centroid)
            # else: boxed in by walls -- stay put this step.

        self.active = society.count(lambda a: a.kind == "walker")


__all__ = ["LabyrinthModel", "build_labyrinth", "PATTERNS", "STATE_CODES"]
