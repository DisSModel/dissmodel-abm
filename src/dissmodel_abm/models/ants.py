"""
A colony of ants bringing food to their nest.

Port of ``Ants`` from TerraME's ``logo`` package (version 0.5.6,
``lua/Ants.lua``), implemented there by Bernardo Dornellas de Mendonça
Martins dos Reis after Fesseha Belay and Javier Morata.

Rules, as in the original
-------------------------
- A nest (the center cell and its eight neighbors) and four food piles (a
  cell, its neighbors and their neighbors) on a ``dimension`` × ``dimension``
  space with Moore neighborhood.
- Each ant starts at one of four cells two steps from the nest center, drawn
  at random.
- A **searching** ant first checks its neighbors for food (see below); then
  moves to the first empty neighbor marked with chemical, in neighborhood
  order; with no such neighbor, walks one step toward a random destination
  cell, avoiding food and nest cells and occupied cells.
- A **bringing** ant first checks its neighbors for the nest; then steps
  toward the nest center, marking its cell and spreading chemical to its
  neighbors when the next cell is neither food nor nest. It moves even into
  an occupied cell.
- After the ants, every cell with chemical loses ``rate_evaporation``; cells
  without chemical become empty and cells with less than 1 become
  ``LESSCHEM``.

A quirk kept from the original
------------------------------
``findFood`` and ``findNest`` end with ``return true`` *inside* the function
passed to ``forEachNeighbor``. ``forEachNeighbor`` stops only when that
function returns ``false``, so the loop goes on over all neighbors and both
functions always return ``false``. Consequences, reproduced here: an ant
next to several food cells takes all of them in one step (and the food
counter drops by that many), and an ant that finds food or the nest still
makes the move of the state it was in. :meth:`Ant.find_food` and
:meth:`Ant.find_nest` follow that behavior.

Defaults: ``dimension = 50``, ``society_size = 10``, ``initial_food = 100``,
``rate_diffusion = 1``, ``rate_evaporation = 0.2``, ``finalTime = 450``.
"""
from __future__ import annotations

import math

import numpy as np

from dissmodel.visualization import track_plot
from dissmodel_abm.core import Agent, AgentModel, Cell, CellSpace, Society

#: Codes of the ``cover`` band.
EMPTY, FOOD, NEST, CHEMICAL, LESSCHEM = 0, 1, 2, 3, 4
#: Extra code of the ``view`` band: a cell with at least one ant.
ANT = 5

COLORS = {
    EMPTY: "#8c510a",
    FOOD: "#2166ac",
    NEST: "#d73027",
    CHEMICAL: "#1a9850",
    LESSCHEM: "#00441b",
    ANT: "#ffff33",
}
LABELS = {
    EMPTY: "empty",
    FOOD: "food",
    NEST: "nest",
    CHEMICAL: "chemical",
    LESSCHEM: "lesschem",
    ANT: "ant",
}

_BLOCKED = (FOOD, NEST)


def _toward(cell: Cell, dest: Cell) -> tuple[int, int]:
    """One step from ``cell`` toward ``dest`` in x and in y (``getNextCoordinateTowardDestiny``)."""
    x = cell.x + (1 if cell.x < dest.x else -1 if cell.x > dest.x else 0)
    y = cell.y + (1 if cell.y < dest.y else -1 if cell.y > dest.y else 0)
    return x, y


class Ant(Agent):
    state: str = "searching"
    dest: Cell | None = None

    def init(self) -> None:
        self.state = "searching"
        self.dest = None

    def execute(self) -> None:
        if self.state == "searching":
            self.search_food()
        elif self.state == "bringing":
            self.bring_food()

    # -- searching -----------------------------------------------------------

    def find_food(self) -> bool:
        """Take every food cell around and switch to bringing; always ``False`` (see module notes)."""
        model = self.model
        cell = self.cell
        for neigh in cell.neighborhood:
            if neigh.cover != FOOD:
                continue
            neigh.cover = CHEMICAL
            for neigh2 in self.cell.neighborhood:
                if neigh2.cover not in _BLOCKED:
                    neigh2.cover = CHEMICAL
                    neigh2.chemical += model.rate_diffusion
                    for neigh3 in neigh2.neighborhood:
                        if neigh3.cover not in _BLOCKED:
                            neigh3.cover = LESSCHEM
                            neigh3.chemical += model.rate_diffusion / 2
            self.state = "bringing"
            model.remaining_food -= 1
        return False

    def search_food(self) -> None:
        if self.find_food():
            return

        any_chem = False
        for neigh in self.cell.neighborhood:
            if neigh.cover in (CHEMICAL, LESSCHEM) and neigh.is_empty():
                self.move(neigh)
                any_chem = True
                break

        if not any_chem:
            if self.dest is None:
                self.dest = self.model.cs.sample()
            self.goto_cell()

    def goto_cell(self) -> None:
        cs = self.model.cs
        dest = self.dest
        new_cell = cs.get(*_toward(self.cell, dest))

        if new_cell.cover not in _BLOCKED and new_cell.is_empty():
            self.move(new_cell)
        else:
            new_cell = self.cell.neighborhood.sample()
            if new_cell.cover not in _BLOCKED and new_cell.is_empty():
                self.move(new_cell)

        if new_cell is dest:
            self.dest = None

    # -- bringing --------------------------------------------------------------

    def find_nest(self) -> bool:
        """Switch to searching when next to the nest; always ``False`` (see module notes)."""
        for neigh in self.cell.neighborhood:
            if neigh.cover == NEST:
                self.state = "searching"
        return False

    def bring_food(self) -> None:
        if self.find_nest():
            return

        model = self.model
        cell = self.cell
        new_cell = model.cs.get(*_toward(cell, model.nest_cell))

        if new_cell.cover not in _BLOCKED:
            cell.cover = CHEMICAL
            cell.chemical += model.rate_diffusion
            for neigh in cell.neighborhood:
                if neigh.cover not in (FOOD, NEST, CHEMICAL):
                    neigh.chemical += model.rate_diffusion / 2
                    if 0 < neigh.chemical <= 1:
                        neigh.cover = LESSCHEM
                    elif neigh.chemical > 1:
                        neigh.cover = CHEMICAL

        self.move(new_cell)

    # -- start --------------------------------------------------------------------

    def random_start(self) -> None:
        start = self.model.start_cells[int(self.rng.integers(4))]
        self.enter(start)


@track_plot("Food", "tab:blue")
class Ants(AgentModel):
    """
    Parameters (``setup``)
    ----------------------
    dimension : int
        Width and height of the space. Default 50.
    society_size : int
        Number of ants. Default 10.
    initial_food : int
        Initial value of the food counter. Default 100.
    rate_diffusion : float
        Chemical deposited by an ant. Default 1.
    rate_evaporation : float
        Chemical lost per step by each cell. Default 0.2.

    Bands: ``cover`` (:data:`EMPTY`, :data:`FOOD`, :data:`NEST`,
    :data:`CHEMICAL`, :data:`LESSCHEM`), ``chemical``, ``view`` (``cover``
    with :data:`ANT` where there are ants) and ``n_agents``.
    """

    food: float = 0

    def setup(
        self,
        dimension: int = 50,
        society_size: int = 10,
        initial_food: int = 100,
        rate_diffusion: float = 1,
        rate_evaporation: float = 0.2,
    ) -> None:
        self.dimension = dimension
        self.rate_diffusion = rate_diffusion
        self.rate_evaporation = rate_evaporation

        self.cs = CellSpace.grid(dimension, cover=np.int8(EMPTY), chemical=0.0)
        self.cs.create_neighborhood()

        center = dimension // 2
        self.nest_cell = self.cs.get(center, center)
        self.nest_cell.cover = NEST
        for neigh in self.nest_cell.neighborhood:
            neigh.cover = NEST

        cx, cy = self.nest_cell.x, self.nest_cell.y
        self.start_cells = [
            self.cs.get(cx + 2, cy),
            self.cs.get(cx - 2, cy),
            self.cs.get(cx, cy - 2),
            self.cs.get(cx, cy + 2),
        ]

        d = dimension
        for x, y in ((d / 8, d / 8), (d / 7, d * 3 / 4), (d * 3 / 4, d / 3), (d * 3 / 4, d * 4 / 5)):
            self._draw_food(x, y)

        self.society = Society(Ant, quantity=society_size)
        self.place(self.society, cellspace=self.cs, strategy="void")
        self.society.call("random_start")

        self.cs.add_derived_band("view", _view, dtype=np.int8)
        # Counter changed by the ants (the original ``cs.food``); ``food``
        # copies it once per step so that the chart gets one point per step.
        self.remaining_food = initial_food
        self.food = initial_food

    def _draw_food(self, x: float, y: float) -> None:
        cell = self.cs.get(math.floor(x), math.floor(y))
        cell.cover = FOOD
        for neigh in cell.neighborhood:
            neigh.cover = FOOD
            for neigh2 in neigh.neighborhood:
                neigh2.cover = FOOD

    def evaporate(self) -> None:
        """Cell rule, vectorized (the ``execute`` of the original cells)."""
        cover = self.cs.backend.arrays["cover"]
        chem = self.cs.backend.arrays["chemical"]
        positive = chem > 0
        chem[positive] -= self.rate_evaporation
        marked = (cover == CHEMICAL) | (cover == LESSCHEM)
        to_empty = (chem <= 0) & marked
        to_less = ~to_empty & (chem < 1) & (cover == CHEMICAL)
        cover[to_empty] = EMPTY
        cover[to_less] = LESSCHEM

    def execute(self) -> None:
        self.society.execute()
        self.evaporate()
        self.food = self.remaining_food


def _view(cell: Cell) -> int:
    return ANT if cell.agents else cell.cover


__all__ = [
    "Ants",
    "Ant",
    "EMPTY",
    "FOOD",
    "NEST",
    "CHEMICAL",
    "LESSCHEM",
    "ANT",
    "COLORS",
    "LABELS",
]
