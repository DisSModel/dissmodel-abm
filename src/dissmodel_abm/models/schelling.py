"""
dissmodel_abm/models/schelling.py
==================================
Schelling's segregation model, ported from TerraME's ``logo`` package
(https://www.terrame.org/package/logo/models/#schelling).

Original TerraME parameters
----------------------------
- ``dim``: x and y dimensions of space (default 25).
- ``finalTime``: final simulation time (default 500).
- ``freeSpace``: percentage of space left empty (default 25).
- ``preference``: minimum number of same-type neighbors that makes an
  agent satisfied (default 3).

Rule
----
Two agent types ("red" and "blue") are randomly distributed over a grid,
with a fraction of cells left empty. At each step, every agent counts how
many of its (Queen) neighbors are of the same type. If that count is
below ``preference``, the agent is unhappy and moves to a randomly chosen
empty cell. The simulation typically converges to a segregated
configuration even though agents only have a mild preference for
same-type neighbors.

dissmodel-abm mapping
----------------------
Unlike ``RandomWalkModel`` / ``PredatorPreyModel`` (point agents that
move freely in continuous space), Schelling is a *one-agent-per-cell*
model — exactly the kind of model the TerraME ``logo`` package targets
("Implements spatial agent-based models with at most one agent per
cell"). The model is written against :attr:`~dissmodel_abm.core.AgentModel.society`
(see :mod:`dissmodel_abm.core.society`) — agents are accessed as
:class:`~dissmodel_abm.core.Agent` objects (``agent.agent_type``,
``agent.grid_neighbors()``), never as raw GeoDataFrame rows. Internally,
:meth:`execute` still builds a plain ``{id: type}`` snapshot for the
per-step happiness evaluation — this is a performance optimization
local to the model (TerraME-equivalent of evaluating all agents against
the same "tick" state before committing moves), not something the
modeler needs to know about when reading ``agent.agent_type`` elsewhere.

Movement is implemented as swapping ``agent_type`` between an unhappy
agent's cell and a randomly chosen empty cell — the cell-grid analogue
of TerraME's ``Agent:move()`` to an empty Cell.

Live plotting
-------------
``satisfaction`` is registered via ``@track_plot``, so any
:class:`dissmodel.visualization.Chart` connected to the same
``Environment`` plots it automatically — no extra wiring needed, same
convention used by ``dissmodel-sysdyn``'s ``SIR`` model.
"""
from __future__ import annotations

import numpy as np
from libpysal.weights import Queen

from dissmodel.visualization import track_plot
from dissmodel_abm.core import AgentModel

EMPTY = -1


@track_plot("Satisfaction", "tab:green")
class SchellingModel(AgentModel):
    """
    Schelling's segregation model on a regular grid.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygon grid, e.g. from
        ``dissmodel.geo.vector.vector_grid(dimension=(dim, dim), resolution=1)``.
        An ``agent_type`` column is added if not present.
    free_space : float, optional
        Fraction of cells left empty, in ``[0, 1]``, by default 0.25
        (TerraME's ``freeSpace=25`` means 25%).
    preference : int, optional
        Minimum number of same-type neighbors required for an agent to
        be satisfied, by default 3.
    seed : int, optional
        Random seed for the initial distribution, by default None.

    Notes
    -----
    Each agent (cell) holds ``agent_type``:

    - ``-1`` : empty cell
    - ``0``  : agent of type "red"
    - ``1``  : agent of type "blue"

    Examples
    --------
    >>> from dissmodel.core import Environment
    >>> from dissmodel.geo.vector import vector_grid
    >>> from dissmodel_abm.models.schelling import SchellingModel
    >>> gdf = vector_grid(dimension=(10, 10), resolution=1)
    >>> env = Environment(end_time=20)
    >>> model = SchellingModel(gdf=gdf, free_space=0.25, preference=3, seed=0)
    >>> env.run()  # doctest: +SKIP
    """

    #: Fraction of agents currently satisfied (tracked for live plotting).
    satisfaction: float = 0.0

    def setup(
        self,
        free_space: float = 0.25,
        preference: int = 3,
        seed: int | None = None,
    ) -> None:
        self.free_space = free_space
        self.preference = preference

        self.create_neighborhood(strategy=Queen, use_index=True)

        rng = np.random.default_rng(seed)
        n = len(self.society)

        n_empty = int(round(n * free_space))
        n_occupied = n - n_empty
        n_red = n_occupied // 2
        n_blue = n_occupied - n_red

        types = np.array(
            [0] * n_red + [1] * n_blue + [EMPTY] * n_empty
        )
        rng.shuffle(types)

        # Pre-create the column with the right dtype (int), then assign
        # through Society so all subsequent reads/writes — here and in
        # execute() — go through the same Agent interface.
        self.gdf["agent_type"] = 0
        for agent, t in zip(self.society, types):
            agent.agent_type = int(t)

    def execute(self) -> None:
        society = self.society
        n = len(society)
        if n == 0:
            self.satisfaction = 1.0
            return

        # Snapshot {id: type} so neighbor counts during this step are
        # evaluated against a consistent "tick" state (moves made earlier
        # in the same step don't affect later agents' happiness check),
        # matching TerraME's per-tick evaluation semantics. This is an
        # internal performance detail of execute(); reading/writing
        # agent.agent_type anywhere else always goes through Society.
        type_map = {agent.id: agent.agent_type for agent in society}
        empty_idx = [idx for idx, t in type_map.items() if t == EMPTY]

        if not empty_idx:
            self.satisfaction = self.fraction_satisfied()
            return  # no empty cells, nothing can move

        rng = np.random.default_rng()
        order = list(type_map.keys())
        rng.shuffle(order)

        for idx in order:
            t = type_map[idx]
            if t == EMPTY:
                continue

            neighs = self.neighs_id(idx)
            if not neighs:
                continue

            same = sum(1 for nb in neighs if type_map.get(nb, EMPTY) == t)

            if same < self.preference and empty_idx:
                # Move to a randomly chosen empty cell.
                target = empty_idx.pop(rng.integers(len(empty_idx)))

                type_map[idx] = EMPTY
                type_map[target] = t

                empty_idx.append(idx)

        for agent in society:
            agent.agent_type = type_map[agent.id]

        # Updates the tracked attribute — picked up automatically by any
        # Chart connected to the same Environment (see @track_plot above).
        self.satisfaction = self.fraction_satisfied()

    # ------------------------------------------------------------------
    # Convenience metrics
    # ------------------------------------------------------------------

    def fraction_satisfied(self) -> float:
        """
        Return the fraction of occupied cells whose agent currently has
        at least ``preference`` same-type neighbors.

        Returns
        -------
        float
            Value in ``[0, 1]``. Returns ``1.0`` if there are no agents.
        """
        society = self.society
        occupied = society.select(lambda a: a.agent_type != EMPTY)

        if not occupied:
            return 1.0

        type_map = {agent.id: agent.agent_type for agent in society}

        satisfied = 0
        for agent in occupied:
            neighs = self.neighs_id(agent.id)
            same = sum(1 for nb in neighs if type_map.get(nb, EMPTY) == agent.agent_type)
            if same >= self.preference:
                satisfied += 1

        return satisfied / len(occupied)


__all__ = ["SchellingModel", "EMPTY"]
