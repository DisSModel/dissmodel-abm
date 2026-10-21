"""
Predator-prey dynamics: rabbits, wolves and pasture.

Port of ``PredatorPrey`` from TerraME's ``logo`` package (version 0.5.6,
``lua/PredatorPrey.lua``).

Rules, as in the original
-------------------------
Cells (run first in each step): a ``soil`` cell counts steps and becomes
``pasture`` again on the fourth.

Rabbits (start with energy 40): look at a random neighbor and move there if
it is empty; lose 1 energy; eat the pasture of their cell (+20 energy, the
cell becomes soil); look at another random neighbor and, with energy of 30
or more and that neighbor empty, give birth there and split their energy
with the child; die when energy falls below 0.

Wolves (start with energy 40): look at a random neighbor; if it is empty,
give birth there (energy 50 or more, halving their energy; the child starts
with 40) or move there; if it holds a rabbit, eat it (+20% of the rabbit's
energy); lose 4 energy; die when energy falls below 0.

Order in a step: cells, rabbits, wolves. Defaults: ``dim = 30``, 20 rabbits,
20 wolves, ``finalTime = 500``.
"""
from __future__ import annotations

import numpy as np

from dissmodel.visualization import track_plot
from dissmodel_abm.core import Agent, AgentModel, CellSpace, Society

#: Codes of the ``state`` band.
SOIL, PASTURE = 0, 1
#: Codes of the ``who`` band.
EMPTY, WOLF, RABBIT = 0, 1, 2

STATE_COLORS = {SOIL: "#8c510a", PASTURE: "#1a9850"}
STATE_LABELS = {SOIL: "soil", PASTURE: "pasture"}
WHO_COLORS = {EMPTY: "#ffffff", WOLF: "#d73027", RABBIT: "#4575b4"}
WHO_LABELS = {EMPTY: "empty", WOLF: "wolf", RABBIT: "rabbit"}


class Rabbit(Agent):
    energy: float = 40
    name = "rabbit"

    def eat(self) -> None:
        cell = self.cell
        if cell.state == PASTURE:
            cell.state = SOIL
            self.energy += 20

    def execute(self) -> None:
        cell = self.cell.neighborhood.sample()
        if cell.is_empty():
            self.move(cell)

        self.energy -= 1
        self.eat()

        cell = self.cell.neighborhood.sample()
        if self.energy >= 30 and cell.is_empty():
            child = self.reproduce()
            child.move(cell)
            self.energy /= 2
            child.energy = self.energy

        if self.energy < 0:
            self.die()


class Wolf(Agent):
    energy: float = 40
    name = "wolf"

    def execute(self) -> None:
        cell = self.cell.neighborhood.sample()

        if cell.is_empty():
            if self.energy >= 50:
                child = self.reproduce()
                child.move(cell)
                self.energy /= 2
            else:
                self.move(cell)
        elif cell.agent.name == "rabbit":
            prey = cell.agent
            self.energy += prey.energy * 0.2
            prey.die()

        self.energy -= 4

        if self.energy < 0:
            self.die()


@track_plot("Rabbits", "tab:blue")
@track_plot("Wolves", "tab:red")
class PredatorPrey(AgentModel):
    """
    Parameters (``setup``)
    ----------------------
    dim : int
        Width and height of the space. Default 30.
    rabbits, wolves : int
        Initial populations. Default 20 each.

    Bands: ``state`` (:data:`SOIL`, :data:`PASTURE`), ``count`` (steps as
    soil), ``who`` (:data:`EMPTY`, :data:`WOLF`, :data:`RABBIT`) and
    ``n_agents``.
    """

    #: Current populations (tracked for ``Chart``).
    rabbits: int = 0
    wolves: int = 0

    def setup(self, dim: int = 30, rabbits: int = 20, wolves: int = 20) -> None:
        self.dim = dim
        self.cs = CellSpace.grid(dim, state=np.int8(PASTURE), count=np.int16(0))
        self.cs.create_neighborhood()

        self.rabbit_society = Society(Rabbit, quantity=rabbits)
        self.wolf_society = Society(Wolf, quantity=wolves)
        self.place(self.rabbit_society, self.wolf_society, cellspace=self.cs)

        self.cs.add_derived_band("who", _who, dtype=np.int8)
        self._count()

    def regrow(self) -> None:
        """Cell rule, vectorized: soil turns back into pasture on its fourth step."""
        state = self.cs.backend.arrays["state"]
        count = self.cs.backend.arrays["count"]
        soil = state == SOIL
        count[soil] += 1
        back = soil & (count == 4)
        state[back] = PASTURE
        count[back] = 0

    def execute(self) -> None:
        self.regrow()
        self.rabbit_society.execute()
        self.wolf_society.execute()
        self._count()

    def _count(self) -> None:
        self.rabbits = len(self.rabbit_society)
        self.wolves = len(self.wolf_society)


def _who(cell) -> int:
    agent = cell.agent
    if agent is None:
        return EMPTY
    return WOLF if agent.name == "wolf" else RABBIT


__all__ = [
    "PredatorPrey",
    "Rabbit",
    "Wolf",
    "SOIL",
    "PASTURE",
    "EMPTY",
    "WOLF",
    "RABBIT",
    "STATE_COLORS",
    "STATE_LABELS",
    "WHO_COLORS",
    "WHO_LABELS",
]
