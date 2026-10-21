"""
Agents that reproduce and die by age.

Port of ``LifeCycle`` from TerraME's ``logo`` package (version 0.5.6,
``lua/LifeCycle.lua``).

Rules, as in the original
-------------------------
Each agent starts with age 0 and, every step: ages by one; between ages 15
and 30, gives birth with probability 0.3 (the child starts in the parent's
cell); moves to a random empty neighbor, if any; after age 20, dies with
probability 0.05.

The child stays in the parent's cell when the parent finds no empty
neighbor, so a cell may hold two or more agents and the population is not
limited by the number of cells, exactly as in the code of version 0.5.6.

Note on the reference images of ``logo``: ``log/LifeCycle-*.bmp`` (2016)
were made with the earlier rule (``LogoAgent:breed``, replaced in June 2017),
which gave birth only when there was an empty neighbor and put the child
there. That rule caps the population at the number of cells, which is why
the reference chart levels off near 100 agents. Up to step 40, before the
cap matters, this port and the reference chart agree.

Defaults: ``quantity = 10``, ``dim = 10``, ``finalTime = 100``.
"""
from __future__ import annotations

import numpy as np

from dissmodel.visualization import track_plot
from dissmodel_abm.core import Agent, AgentModel, CellSpace, Society

#: Codes of the ``state`` band.
EMPTY, FULL = 0, 1
COLORS = {EMPTY: "#1a9850", FULL: "#000000"}
LABELS = {EMPTY: "empty", FULL: "full"}


class Person(Agent):
    age: int = 0

    def init(self) -> None:
        self.age = 0

    def execute(self) -> None:
        self.age += 1

        if 15 <= self.age <= 30 and self.rng.random() < 0.3:
            self.reproduce()

        self.walk_to_empty()

        if self.rng.random() < 0.05 and self.age >= 20:
            self.die()


@track_plot("Agents", "tab:blue")
class LifeCycle(AgentModel):
    """
    Parameters (``setup``)
    ----------------------
    dim : int
        Width and height of the space. Default 10.
    quantity : int
        Initial number of agents. Default 10.

    Bands: ``state`` (:data:`EMPTY`, :data:`FULL`) and ``n_agents``.
    """

    agents: int = 0

    def setup(self, dim: int = 10, quantity: int = 10) -> None:
        self.dim = dim
        self.society = Society(Person, quantity=quantity)
        self.cs = CellSpace.grid(dim)
        self.cs.create_neighborhood()
        self.place(self.society, cellspace=self.cs)
        self.cs.add_derived_band(
            "state", lambda cell: EMPTY if cell.is_empty() else FULL, dtype=np.int8
        )
        self.agents = len(self.society)

    def execute(self) -> None:
        self.society.execute()
        self.agents = len(self.society)


__all__ = ["LifeCycle", "Person", "EMPTY", "FULL", "COLORS", "LABELS"]
