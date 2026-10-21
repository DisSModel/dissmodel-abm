"""
A single agent moving around randomly.

Port of ``SingleAgent`` from TerraME's ``logo`` package (version 0.5.6,
``lua/SingleAgent.lua``). Each step, every agent moves to a random empty
neighbor (``walkToEmpty``).

TerraME defaults: ``quantity = 1``, ``dim = 15``, ``finalTime = 100``.
"""
from __future__ import annotations

from dissmodel_abm.core import Agent, AgentModel, CellSpace, Society


class Walker(Agent):
    def execute(self) -> None:
        self.walk_to_empty()


class SingleAgent(AgentModel):
    """
    Parameters (``setup``)
    ----------------------
    dim : int
        Width and height of the space. Default 15.
    quantity : int
        Number of agents. Default 1.

    Bands: ``n_agents`` (agents per cell).
    """

    def setup(self, dim: int = 15, quantity: int = 1) -> None:
        self.dim = dim
        self.cs = CellSpace.grid(dim)
        self.cs.create_neighborhood()
        self.society = Society(Walker, quantity=quantity)
        self.place(self.society, cellspace=self.cs)

    def execute(self) -> None:
        self.society.execute()


__all__ = ["SingleAgent", "Walker"]
