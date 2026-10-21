"""
``AgentModel``: a dissmodel ``Model`` that hosts societies and cell spaces.

It plays the role of TerraME's ``Model{init = ...}`` plus ``Environment`` and
``createPlacement``. The clock is dissmodel's own ``Environment``; there is no
``Timer``/``Event``: the model's ``execute()`` says what happens in a step::

    class PredatorPrey(AgentModel):
        def setup(self, dim=30):
            self.cs = CellSpace.grid(dim, state=PASTURE, count=0)
            self.cs.create_neighborhood()
            self.rabbits = Society(Rabbit, quantity=20)
            self.wolves = Society(Wolf, quantity=20)
            self.place(self.rabbits, self.wolves, cellspace=self.cs)

        def execute(self):
            self.rabbits.execute()
            self.wolves.execute()

    env = Environment(start_time=1, end_time=500)   # 500 steps, as TerraME
    PredatorPrey(seed=42)
    env.run()

Randomness
----------
``AgentModel(seed=...)`` creates one ``numpy.random.Generator``,
``self.rng``. Societies and cell spaces created during ``setup()`` use it,
and so do agents (``agent.rng``), so two runs with the same seed and the
same code give the same result. Code that draws numbers should use
``self.rng`` or ``agent.rng``, never ``numpy.random`` or ``random``.

Time
----
TerraME runs events at ``t = 1, ..., finalTime``. dissmodel's
``Environment(start_time=0, end_time=n)`` runs ``n + 1`` steps; use
``Environment(start_time=1, end_time=final_time)`` for the same count.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from dissmodel.core import Model

from . import _context
from .agent import Agent
from .cellspace import CellSpace
from .society import Society


class AgentModel(Model):
    """
    Base class for agent-based models.

    Parameters
    ----------
    seed : int or numpy.random.SeedSequence, optional
        Seed of the model's random generator. ``None`` draws fresh entropy
        (not reproducible); the seed actually used is kept in
        :attr:`seed_used` so that such a run can still be repeated.
    **kwargs
        ``step``, ``start_time``, ``end_time``, ``name`` (see
        ``dissmodel.core.Model``) and the keyword arguments of ``setup()``.

    Notes
    -----
    Subclasses that override ``post_execute`` should call
    ``super().post_execute()``, which recomputes derived cell bands.
    """

    rng: np.random.Generator
    seed: Any
    seed_used: int

    def __init__(self, *args: Any, seed: Any = None, **kwargs: Any) -> None:
        seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        self.seed = seed
        self.seed_used = int(seq.entropy)  # type: ignore[arg-type]
        self.rng = np.random.default_rng(seq)
        self._cellspaces: list[CellSpace] = []
        _context.push(self)
        try:
            super().__init__(*args, **kwargs)
        finally:
            _context.pop(self)
        self.refresh()

    # -- cell spaces ---------------------------------------------------------------

    def _register_cellspace(self, cs: CellSpace) -> None:
        if cs not in self._cellspaces:
            self._cellspaces.append(cs)

    def refresh(self) -> None:
        """Recompute the derived bands of the model's cell spaces."""
        for cs in self._cellspaces:
            cs.refresh()

    def post_execute(self) -> None:
        self.refresh()

    # -- placement -----------------------------------------------------------------------

    def place(
        self,
        *targets: Society | Agent,
        cellspace: CellSpace,
        strategy: str = "random",
        max: int = 1,  # noqa: A002 - same name as TerraME's argument
    ) -> None:
        """
        Put agents in cells (TerraME ``Environment:createPlacement``).

        Parameters
        ----------
        *targets : Society or Agent
            Agents to place; societies are taken in the order given.
        cellspace : CellSpace
            Where to place them. It is also registered with the model, so its
            derived bands are refreshed after each step.
        strategy : {"random", "uniform", "spread", "void"}
            - ``"random"`` (TerraME's default): agents and cells are shuffled
              and filled in order, ``max`` agents per cell.
            - ``"uniform"``: cells in row order, one agent each, starting
              again from the first cell when the cells run out.
            - ``"spread"``: each agent picks a random cell among those with
              fewer than ``max`` agents.
            - ``"void"``: no agent is placed; agents enter cells later with
              ``agent.enter(cell)``.
        max : int
            Maximum agents per cell for ``"random"`` and ``"spread"``.
            Default 1, as in TerraME.

        Raises
        ------
        ValueError
            If there are more agents than ``len(cellspace) * max``.
        """
        self._register_cellspace(cellspace)
        agents: list[Agent] = []
        for t in targets:
            if isinstance(t, Society):
                agents.extend(t)
            elif isinstance(t, Agent):
                agents.append(t)
            else:
                raise TypeError(f"Cannot place {type(t).__name__}; expected Society or Agent.")

        if strategy == "void":
            return

        # Work on flat cell indices, in row order, and create Cell objects
        # only for the cells that receive agents: placing 1000 agents costs
        # the same in a 50 x 50 space and in a 3000 x 3000 one.
        valid = cellspace._valid_idx
        n_cells = len(valid)
        if strategy in ("random", "spread"):
            if max < 1:
                raise ValueError("Argument 'max' must be positive.")
            if len(agents) > n_cells * max:
                raise ValueError("It is not possible to put such amount of agents in space.")

        if strategy == "random":
            cells = valid[self.rng.permutation(n_cells)]
            agents = [agents[i] for i in self.rng.permutation(len(agents))]
            for k, agent in enumerate(agents):
                agent.enter(cellspace._cell_at(int(cells[k // max])))
        elif strategy == "uniform":
            for k, agent in enumerate(agents):
                agent.enter(cellspace._cell_at(int(valid[k % n_cells])))
        elif strategy == "spread":
            agents = [agents[i] for i in self.rng.permutation(len(agents))]
            free = valid.copy()
            n_free = n_cells
            for agent in agents:
                i = int(self.rng.integers(n_free))
                cell = cellspace._cell_at(int(free[i]))
                agent.enter(cell)
                if len(cell.agents) >= max:
                    free[i] = free[n_free - 1]
                    n_free -= 1
        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}; use 'random', 'uniform', 'spread' or 'void'."
            )


__all__ = ["AgentModel"]
