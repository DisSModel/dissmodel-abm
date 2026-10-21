"""
Schelling's segregation model.

Port of ``Schelling`` from TerraME's ``logo`` package (version 0.5.6,
``lua/Schelling.lua``).

Rules, as in the original
-------------------------
- Space of ``dim`` × ``dim`` cells with Moore neighborhood and **wrapped
  borders** (``createNeighborhood{wrap = true}``).
- ``floor(dim² · (1 − free_space))`` agents, each "brazil" or "germany" at
  random (``Random{"brazil", "germany"}``), placed at random, one per cell.
- An agent is unhappy when fewer than ``preference`` of its neighbors are
  agents of its own kind.
- **One** move per step: the group of unhappy agents is rebuilt, one of them
  is drawn at random and moves to a random empty cell anywhere in the space.

Defaults follow the code of the original: ``dim = 25``, ``preference = 3``,
``free_space = 0.05`` (a ``Choice{min = 0.05, max = 0.20}`` takes its
minimum as default; the help text of the original says 25) and
``finalTime = 500``.

``unhappy`` is the percentage of unhappy agents counted when the step
starts, before the move, which is what the original chart shows.
"""
from __future__ import annotations

import math

import numpy as np

from dissmodel.visualization import track_plot
from dissmodel_abm.core import Agent, AgentModel, CellSpace, Group, Society, Trajectory

#: Codes of the ``state`` band.
FREE, BRAZIL, GERMANY = 0, 1, 2
_CODES = {"brazil": BRAZIL, "germany": GERMANY}

COLORS = {FREE: "#fee08b", BRAZIL: "#1a9850", GERMANY: "#d73027"}
LABELS = {FREE: "free", BRAZIL: "brazil", GERMANY: "germany"}


class Resident(Agent):
    state: str = "brazil"

    def init(self) -> None:
        self.state = ("brazil", "germany")[int(self.rng.integers(2))]

    def is_unhappy(self, preference: int) -> bool:
        likeme = 0
        for neigh in self.cell.neighborhood:
            other = neigh.agent
            if other is not None and other.state == self.state:
                likeme += 1
        return likeme < preference


@track_plot("Unhappy", "tab:red")
class Schelling(AgentModel):
    """
    Parameters (``setup``)
    ----------------------
    dim : int
        Width and height of the space. Default 25.
    free_space : float
        Fraction of cells left empty. Default 0.05.
    preference : int
        Neighbors of the same kind an agent needs to be happy. Default 3.

    Bands: ``state`` (:data:`FREE`, :data:`BRAZIL`, :data:`GERMANY`) and
    ``n_agents``.
    """

    unhappy: float = 0.0

    def setup(self, dim: int = 25, free_space: float = 0.05, preference: int = 3) -> None:
        self.dim = dim
        self.free_space = free_space
        self.preference = preference

        self.cs = CellSpace.grid(dim)
        self.cs.create_neighborhood(wrap=True)

        self.society = Society(Resident, quantity=math.floor(dim * dim * (1.0 - free_space)))
        self.place(self.society, cellspace=self.cs)

        self.cs.add_derived_band(
            "state",
            lambda cell: _CODES[cell.agent.state] if cell.agent is not None else FREE,
            dtype=np.int8,
        )
        self.unhappy_agents = Group(self.society, select=lambda a: a.is_unhappy(self.preference))
        self.unhappy = self.percent_unhappy()

    def execute(self) -> None:
        self.unhappy_agents.filter()
        if len(self.unhappy_agents) > 0:
            agent = self.unhappy_agents.sample()
            empty_cells = Trajectory(self.cs, select=lambda cell: cell.is_empty())
            agent.move(empty_cells.sample())
        self.unhappy = 100.0 * len(self.unhappy_agents) / len(self.society)

    def percent_unhappy(self) -> float:
        """Percentage of agents unhappy right now."""
        n = sum(1 for a in self.society if a.is_unhappy(self.preference))
        return 100.0 * n / len(self.society)


__all__ = ["Schelling", "Resident", "FREE", "BRAZIL", "GERMANY", "COLORS", "LABELS"]
