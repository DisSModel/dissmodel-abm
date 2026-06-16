"""
dissmodel_abm/models/ants.py
==============================
Ant colony foraging model, ported from TerraME's ``logo`` package
(https://www.terrame.org/package/logo/models/#ants).

Original Lua source:
https://www.terrame.org/package/logo/lua/Ants.lua

Rule
----
A square cellular grid holds a single nest cell, a number of food
cells, and a pheromone (chemical) concentration that evaporates every
step. A colony of ant agents alternates between two states:

- ``"searching"``: if the ant's current cell still has food, it picks
  up one unit, switches to ``"bringing"``, and deposits a strong dose
  of pheromone. Otherwise it moves to the neighboring cell with the
  most pheromone (following the strongest trail it can smell); if no
  neighbor has any pheromone, it picks a random neighbor instead
  (an unbiased random walk, exactly like TerraME's
  ``agent:move(Random(empty):sample())`` in the no-trail case).
- ``"bringing"``: the ant heads straight back toward the nest, always
  stepping to whichever neighbor is geometrically closest to it, and
  depositing pheromone along the way. Reaching the nest counts one
  unit of food collected and switches the ant back to ``"searching"``.

Pheromone evaporates by a fixed fraction every step
(``evaporation_rate``), matching TerraME's ``rateEvaporation``.

This is a simplification of the original NetLogo-style "ants" model
referenced by TerraME's ``logo`` package: pheromone is deposited only
on the ant's current cell (not splashed onto neighbors), and food is
scattered as single-unit cells rather than clustered piles. Both are
deliberate simplifications to keep the port self-contained; see
``build_colony`` to customize the food layout.

dissmodel-abm mapping
----------------------
Like :mod:`dissmodel_abm.models.labyrinth`, the grid cells and the
moving agents share one ``self.gdf`` / ``self.society``, distinguished
by the ``kind`` column:

- ``kind="cell"`` — one per grid tile (Polygon). Mutable ``pheromone``
  (float) and ``food`` (int, units remaining) columns, plus a fixed
  ``is_nest`` flag.
- ``kind="ant"``  — the moving agents (Point, placed at the centroid
  of their current cell), tracked via ``cell_id`` and ``state``
  (``"searching"`` / ``"bringing"``).

Both kinds carry a ``display_code`` column (see :data:`DISPLAY_CODES`)
recomputed every step, so a single
:class:`dissmodel.visualization.Map` can render cells and ants
together with a stable color scale (numeric column + fixed
``vmin``/``vmax``, the same technique used by
:mod:`dissmodel_abm.models.labyrinth` and
:mod:`dissmodel_abm.models.schelling` to avoid matplotlib rescaling
the colormap when a category temporarily disappears).

Live plotting
-------------
``collected`` (cumulative food brought to the nest) and ``foraging``
(ants currently searching, as opposed to bringing food home) are
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

#: Fixed numeric code for each visual category, used by the
#: ``display_code`` column. Plotting on this numeric column with
#: explicit ``vmin=0, vmax=len(DISPLAY_CODES) - 1`` keeps colors stable
#: across steps, regardless of which categories are actually present
#: in a given frame.
DISPLAY_CODES: dict[str, int] = {
    "empty": 0,
    "trail_weak": 1,
    "trail": 2,
    "food": 3,
    "nest": 4,
    "ant": 5,
}


def build_colony(
    dimension: int = 50,
    resolution: float = 1.0,
    n_food_cells: int = 100,
    seed: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Build the grid GeoDataFrame consumed by :class:`AntsModel`.

    Creates a square grid with a single nest cell at its center and
    ``n_food_cells`` single-unit food cells scattered at random.

    Parameters
    ----------
    dimension : int, optional
        Grid width and height, in cells, by default 50 (TerraME's
        ``dimension``).
    resolution : float, optional
        Cell size, by default 1.0.
    n_food_cells : int, optional
        Number of cells seeded with one unit of food each, by default
        100 (TerraME's ``initialFood``).
    seed : int, optional
        Random seed for food placement, by default None.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon grid with ``pheromone``, ``food``, ``is_nest``,
        ``display_code`` and ``kind="cell"`` columns, ready to pass as
        ``gdf=`` to :class:`AntsModel`.
    """
    grid = vector_grid(dimension=(dimension, dimension), resolution=resolution)
    grid["kind"] = "cell"
    grid["pheromone"] = 0.0
    grid["food"] = 0
    grid["is_nest"] = False

    nest_id = f"{dimension // 2}-{dimension // 2}"
    grid.at[nest_id, "is_nest"] = True

    rng = np.random.default_rng(seed)
    candidates = [idx for idx in grid.index if idx != nest_id]
    n_food_cells = min(n_food_cells, len(candidates))
    food_ids = rng.choice(candidates, size=n_food_cells, replace=False)
    grid.loc[food_ids, "food"] = 1

    grid["display_code"] = DISPLAY_CODES["empty"]
    grid.loc[grid["food"] > 0, "display_code"] = DISPLAY_CODES["food"]
    grid.at[nest_id, "display_code"] = DISPLAY_CODES["nest"]

    return grid


@track_plot("Foraging", "tab:orange")
@track_plot("Collected", "tab:green")
class AntsModel(AgentModel):
    """
    Ant colony foraging model.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Colony grid, from :func:`build_colony` — one ``"cell"`` agent
        per tile, with ``pheromone``, ``food`` and ``is_nest`` columns.
    n_ants : int, optional
        Number of ant agents, starting at the nest, by default 10
        (TerraME's ``societySize``).
    evaporation_rate : float, optional
        Fraction of pheromone lost every step, in ``[0, 1]``, by
        default 0.2 (TerraME's ``rateEvaporation``).
    deposit_amount : float, optional
        Pheromone added to an ant's current cell when it picks up food
        or while it is bringing food home, by default 5.0.
    strong_threshold : float, optional
        Minimum pheromone concentration for a cell to be drawn as a
        "strong" trail rather than a "weak" (faded) one, by default
        1.0. Purely cosmetic — does not affect ant behavior.
    seed : int, optional
        Random seed for the ants' initial random-walk decisions
        (the colony layout itself is seeded via :func:`build_colony`),
        by default None.

    Notes
    -----
    Each ant agent (``kind="ant"``) holds ``cell_id`` (the maze cell it
    currently occupies) and ``state`` (``"searching"`` or
    ``"bringing"``).

    Examples
    --------
    >>> from dissmodel.core import Environment
    >>> from dissmodel_abm.models.ants import AntsModel, build_colony
    >>> gdf = build_colony(dimension=30, n_food_cells=40, seed=0)
    >>> env = Environment(end_time=300)
    >>> model = AntsModel(gdf=gdf, n_ants=20, seed=0)
    >>> env.run()  # doctest: +SKIP
    """

    #: Cumulative food units brought to the nest (tracked for live plotting).
    collected: int = 0

    #: Ants currently searching for food, as opposed to bringing it home
    #: (tracked for live plotting).
    foraging: int = 0

    def setup(
        self,
        n_ants: int = 10,
        evaporation_rate: float = 0.2,
        deposit_amount: float = 5.0,
        strong_threshold: float = 1.0,
        seed: int | None = None,
    ) -> None:
        # Neighborhood must be built while self.gdf only holds the
        # (polygon) grid cells -- libpysal's Queen contiguity assumes a
        # homogeneous polygon layer.
        self.create_neighborhood(strategy=Queen, use_index=True)

        for col, default in (
            ("kind", "cell"),
            ("pheromone", 0.0),
            ("food", 0),
            ("is_nest", False),
            ("display_code", DISPLAY_CODES["empty"]),
        ):
            if col not in self.gdf.columns:
                self.gdf[col] = default

        nest_ids = self.gdf.index[self.gdf["is_nest"]].tolist()
        if not nest_ids:
            raise ValueError(
                "Colony grid has no nest cell (set is_nest=True on exactly one cell)."
            )
        self.nest_id = nest_ids[0]

        self.evaporation_rate = evaporation_rate
        self.deposit_amount = deposit_amount
        self.strong_threshold = strong_threshold

        nest_point = self.gdf.at[self.nest_id, "geometry"].centroid
        for _ in range(n_ants):
            self.society.add(
                geometry=nest_point,
                kind="ant",
                state="searching",
                cell_id=self.nest_id,
                display_code=DISPLAY_CODES["ant"],
            )

        self.collected = 0
        self.foraging = n_ants

    def execute(self) -> None:
        society = self.society
        ants = society.select(lambda a: a.kind == "ant")
        rng = np.random.default_rng()

        for ant in ants:
            if ant.state == "searching":
                self._search_step(ant, rng)
            else:
                self._bring_step(ant, rng)

        # Evaporation, vectorized over every cell at once.
        cells_mask = self.gdf["kind"] == "cell"
        pheromone = self.gdf.loc[cells_mask, "pheromone"] * (1 - self.evaporation_rate)
        pheromone[pheromone < 0.01] = 0.0
        self.gdf.loc[cells_mask, "pheromone"] = pheromone

        self._update_display_codes(cells_mask)

        self.foraging = society.count(lambda a: a.kind == "ant" and a.state == "searching")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_step(self, ant, rng: np.random.Generator) -> None:
        cell_id = ant.cell_id
        if self.gdf.at[cell_id, "food"] > 0:
            self.gdf.at[cell_id, "food"] -= 1
            ant.state = "bringing"
            self._deposit(cell_id)
            return

        neighbor_ids = list(self.neighs_id(cell_id))
        if not neighbor_ids:
            return
        rng.shuffle(neighbor_ids)

        best = max(neighbor_ids, key=lambda nb: self.gdf.at[nb, "pheromone"])
        if self.gdf.at[best, "pheromone"] > 0:
            target = best
        else:
            target = neighbor_ids[rng.integers(len(neighbor_ids))]

        self._move_ant(ant, target)

    def _bring_step(self, ant, rng: np.random.Generator) -> None:
        cell_id = ant.cell_id
        if cell_id == self.nest_id:
            ant.state = "searching"
            self.collected += 1
            return

        neighbor_ids = list(self.neighs_id(cell_id))
        if not neighbor_ids:
            return
        rng.shuffle(neighbor_ids)

        nest_point = self.gdf.at[self.nest_id, "geometry"].centroid
        target = min(
            neighbor_ids,
            key=lambda nb: self.gdf.at[nb, "geometry"].centroid.distance(nest_point),
        )

        self._deposit(cell_id)
        self._move_ant(ant, target)

    def _deposit(self, cell_id) -> None:
        self.gdf.at[cell_id, "pheromone"] += self.deposit_amount

    def _move_ant(self, ant, target_cell_id) -> None:
        ant.cell_id = target_cell_id
        ant.move_to(self.gdf.at[target_cell_id, "geometry"].centroid)

    def _update_display_codes(self, cells_mask) -> None:
        pheromone = self.gdf.loc[cells_mask, "pheromone"]
        food = self.gdf.loc[cells_mask, "food"]
        is_nest = self.gdf.loc[cells_mask, "is_nest"]

        codes = np.full(len(pheromone), DISPLAY_CODES["empty"])
        codes[(pheromone > 0).to_numpy()] = DISPLAY_CODES["trail_weak"]
        codes[(pheromone >= self.strong_threshold).to_numpy()] = DISPLAY_CODES["trail"]
        codes[(food > 0).to_numpy()] = DISPLAY_CODES["food"]
        # `is_nest` is object-dtype at the column level (the "ant" rows
        # leave it as NaN, which upcasts the whole Series from bool),
        # so cast explicitly before using it as a boolean mask.
        codes[is_nest.astype(bool).to_numpy()] = DISPLAY_CODES["nest"]

        self.gdf.loc[cells_mask, "display_code"] = codes


__all__ = ["AntsModel", "build_colony", "DISPLAY_CODES"]
