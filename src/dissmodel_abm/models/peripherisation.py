"""
dissmodel_abm/models/peripherisation.py
=========================================
The Peripherisation Model (Barros and Alves Jr., 2003), from Joana Barros'
PhD thesis *"Urban Growth in Latin American Cities: Exploring urban
dynamics through agent-based simulation"* (UCL, 2004,
http://www.dpi.inpe.br/gilberto/cursos/st-society/barros-phd-thesis.pdf).

Originally implemented in StarLogo, later in FORTRAN, RePast (JAVA), and
in a TerraME/Lua dialect (the variant that prompted this port). This is
a from-scratch Python port of the *original* StarLogo rule set, as
described in the thesis and in Barros & Alves Jr. (2003) "Simulating
Rapid Urbanisation in Latin American Cities" — not a translation of the
Lua variant, which implements a different (region/density-based) rule
set built on top of the same general idea.

The model
---------
Population is divided into three economic groups, following the
pyramidal income-distribution model used in Latin American urban
studies:

- **red** (``group=0``)    — high income, a minority
- **yellow** (``group=1``) — middle income
- **blue** (``group=2``)   — low income, the majority

All agents share the same locational preference: they want to settle
close to infrastructure, which in the model is represented by proximity
to red (high-income) cells. What differs between the three groups is
the *economic power to displace others*:

- **red** can settle anywhere, evicting whoever already occupies that
  cell (the evicted agent re-enters the pool of agents looking for a
  place to settle).
- **yellow** can settle anywhere *except* on a red cell.
- **blue** can only settle on an empty cell.

Each pending agent performs a biased random walk of length ``steps``
(the model's central parameter) across the grid — favoring movement
toward the centroid of currently-occupied red cells — and then attempts
to settle on the cell it ends up on, according to its group's rule
above. If settlement fails (e.g. a blue agent landing on an occupied
cell), the agent stays in the pool and tries again next step.

This produces the core-periphery pattern described in the thesis: red
clusters near the seed, yellow forms around red, and blue is pushed to
the outer ring — the opposite of the classic Burgess concentric-ring
model, matching the inverted income gradient observed in Latin American
cities.

Parameters (paper's names in parentheses)
-------------------------------------------
- ``steps`` (``steps``): number of biased random-walk steps a pending
  agent takes before attempting to settle. Larger values pull
  settlement closer to existing red cells per attempt, but each step
  also costs one model tick per agent, so larger ``steps`` produces
  slower, more homogeneous growth; smaller ``steps`` produces faster,
  more spread-out growth (thesis section 7.4.1.1).
- ``proportions`` (``proportion of agents per economic group``): the
  (red, yellow, blue) split, e.g. ``(0.10, 0.40, 0.50)`` — the thesis'
  default pyramidal distribution.
- ``n_agents``: total number of agents to place over the course of the
  simulation.
- ``agents_per_step``: how many pending agents attempt to walk/settle
  each model tick (the thesis' StarLogo original effectively processes
  many agents per tick; this is exposed as a parameter for
  performance/visual-pacing control rather than being part of the
  original model).
- ``seed_cells``: initial red seed(s) the simulation grows from
  (defaults to a single central seed, the thesis' most common initial
  condition; see thesis Figure 6.9 / section 7.4.1.3 for alternatives
  such as multiple seeds, a path, or a colonial grid).

dissmodel-abm mapping
----------------------
Like ``SchellingModel``, this is a one-agent-per-cell model: ``self.gdf``
is a polygon grid from ``dissmodel.geo.vector.vector_grid``, and every
cell is an :class:`~dissmodel_abm.core.Agent` with a ``group`` attribute.
Pending (not-yet-settled) agents are tracked separately as a simple
in-memory queue — they are not yet cells, so they aren't
``Society`` members until they settle (mirrors the StarLogo/RePast
agent lifecycle, where an agent exists and walks before acquiring a
``patch``/``Cell``).

Live plotting
-------------
``red``, ``yellow``, ``blue``, and ``pending``
are registered via ``@track_plot``, matching the convention used by
``PredatorPreyModel`` and ``SchellingModel``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from libpysal.weights import Queen

from dissmodel.visualization import track_plot
from dissmodel_abm.core import AgentModel

EMPTY = -1
RED = 0      # high income
YELLOW = 1   # middle income
BLUE = 2     # low income

_GROUP_NAMES = {RED: "red", YELLOW: "yellow", BLUE: "blue"}


@track_plot("Red", "tab:red")
@track_plot("Yellow", "tab:olive")
@track_plot("Blue", "tab:blue")
@track_plot("Pending", "gray")
class PeripherisationModel(AgentModel):
    """
    The Peripherisation Model (Barros and Alves Jr., 2003).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygon grid, e.g. from
        ``dissmodel.geo.vector.vector_grid(dimension=(dim, dim), resolution=1)``.
        A ``group`` column is added if not present.
    steps : int, optional
        Number of biased random-walk steps a pending agent takes before
        attempting to settle, by default 2 (a thesis-typical value;
        thesis tests steps in {1, 2, 4, 8}).
    proportions : tuple of float, optional
        ``(red, yellow, blue)`` fractions of the agent population, by
        default ``(0.10, 0.40, 0.50)`` (the thesis' pyramidal default).
        Must sum to 1.0.
    n_agents : int, optional
        Total number of agents to place over the simulation, by default
        ``None``, meaning one agent per non-seed cell in the grid (fills
        the grid).
    agents_per_step : int, optional
        How many pending agents attempt to walk/settle each tick, by
        default 5.
    seed_cells : list of str, optional
        Cell ids to seed as red at setup, by default ``None``, meaning a
        single central seed (the thesis' most common initial condition).
    seed : int, optional
        Random seed, by default ``None``.

    Notes
    -----
    Each occupied cell holds ``group``:

    - ``-1`` : empty cell
    - ``0``  : red (high income)
    - ``1``  : yellow (middle income)
    - ``2``  : blue (low income)

    Examples
    --------
    >>> from dissmodel.core import Environment
    >>> from dissmodel.geo.vector import vector_grid
    >>> from dissmodel_abm.models.peripherisation import PeripherisationModel
    >>> gdf = vector_grid(dimension=(31, 31), resolution=1)
    >>> env = Environment(end_time=200)
    >>> model = PeripherisationModel(gdf=gdf, steps=2, n_agents=400, seed=0)
    >>> env.run()  # doctest: +SKIP
    """

    #: Current number of red (high-income) cells (tracked for live plotting).
    red: int = 0
    #: Current number of yellow (middle-income) cells (tracked for live plotting).
    yellow: int = 0
    #: Current number of blue (low-income) cells (tracked for live plotting).
    blue: int = 0
    #: Number of agents still trying to settle (tracked for live plotting).
    pending: int = 0

    def setup(
        self,
        steps: int = 2,
        proportions: tuple[float, float, float] = (0.10, 0.40, 0.50),
        n_agents: Optional[int] = None,
        agents_per_step: int = 5,
        seed_cells: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> None:
        if abs(sum(proportions) - 1.0) > 1e-6:
            raise ValueError(f"proportions must sum to 1.0, got {proportions}")

        self.steps = steps
        self.proportions = proportions
        self.agents_per_step = agents_per_step
        self._rng = np.random.default_rng(seed)

        self.create_neighborhood(strategy=Queen, use_index=True)

        if "group" not in self.gdf.columns:
            self.gdf["group"] = EMPTY
        else:
            self.gdf["group"] = EMPTY  # always start from a clean slate

        # Seed the initial red cell(s) (thesis: single central seed by
        # default; see thesis section 7.4.1.3 for alternative initial
        # conditions such as multiple seeds, a path, or a colonial grid).
        if seed_cells is None:
            seed_cells = [self._central_cell_id()]
        for cell_id in seed_cells:
            self.society[cell_id].group = RED

        if n_agents is None:
            n_agents = len(self.gdf) - len(seed_cells)

        n_red = round(n_agents * proportions[0])
        n_yellow = round(n_agents * proportions[1])
        n_blue = n_agents - n_red - n_yellow

        pending = [RED] * n_red + [YELLOW] * n_yellow + [BLUE] * n_blue
        self._rng.shuffle(pending)
        self._pending: list[int] = pending

        self._update_tracked_counts()

    def execute(self) -> None:
        if not self._pending:
            self._update_tracked_counts()
            return

        red_target = self._red_centroid()
        n = min(self.agents_per_step, len(self._pending))

        for _ in range(n):
            group = self._pending.pop(0)
            cell_id = self._walk(red_target)
            settled = self._try_settle(cell_id, group)
            if not settled:
                # Failed to settle (cell occupied by a group this agent
                # can't evict) — back of the queue to try again later.
                self._pending.append(group)

        self._update_tracked_counts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _central_cell_id(self) -> str:
        """Return the id of the grid cell closest to the overall centroid."""
        gdf = self.gdf
        overall = gdf.geometry.union_all().centroid
        dists = gdf.geometry.centroid.distance(overall)
        return gdf.index[dists.argmin()]

    def _red_centroid(self) -> Optional[tuple[float, float]]:
        """Centroid of all currently-occupied red cells, or None if there are none."""
        red = self.society.select(lambda a: a.group == RED)
        if not red:
            return None
        xs = [a.geometry.centroid.x for a in red]
        ys = [a.geometry.centroid.y for a in red]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _walk(self, target: Optional[tuple[float, float]]) -> str:
        """
        Perform a biased random walk of length ``self.steps`` starting
        from a random occupied cell's neighbor (agents enter the
        simulated space adjacent to already-settled cells, mirroring the
        Lua reference's ``random_localize`` choosing a cell near the
        existing settlement rather than anywhere on the grid), and
        return the id of the cell the walk ends on.

        Each step moves to a Queen-neighbor of the current cell, biased
        toward ``target`` (the centroid of red cells) via inverse-
        distance weighting when ``target`` is given; otherwise the walk
        is uniformly random.
        """
        current = self._entry_cell()

        for _ in range(self.steps):
            neighs = self.neighs_id(current)
            if not neighs:
                break
            if target is not None:
                dists = np.array([
                    np.hypot(
                        self.gdf.at[n, "geometry"].centroid.x - target[0],
                        self.gdf.at[n, "geometry"].centroid.y - target[1],
                    )
                    for n in neighs
                ])
                weights = 1.0 / (dists + 0.1)
                probs = weights / weights.sum()
                current = neighs[self._rng.choice(len(neighs), p=probs)]
            else:
                current = neighs[self._rng.integers(len(neighs))]

        return current

    def _entry_cell(self) -> str:
        """
        Pick a starting cell for a new agent's walk: a random neighbor of
        a random already-occupied cell, so agents enter adjacent to the
        growing settlement rather than appearing anywhere on the grid.
        Falls back to a uniformly random cell if nothing is occupied yet.
        """
        occupied = self.gdf.index[self.gdf["group"] != EMPTY]
        if len(occupied) == 0:
            return self.gdf.index[self._rng.integers(len(self.gdf))]

        anchor = occupied[self._rng.integers(len(occupied))]
        neighs = self.neighs_id(anchor)
        if not neighs:
            return anchor
        return neighs[self._rng.integers(len(neighs))]

    def _try_settle(self, cell_id: str, group: int) -> bool:
        """
        Attempt to settle ``group`` on ``cell_id``, following the
        thesis' eviction rules.

        If settlement displaces an occupant, the displaced agent's group
        is pushed back onto ``self._pending`` directly (it re-enters the
        pool to find a new place, exactly as in the thesis: "the latter
        is 'evicted' and must find another place to settle").

        Returns
        -------
        bool
            True if the agent settled successfully, False if it must
            remain pending and try again.
        """
        current = self.society[cell_id].group

        if group == RED:
            can_settle = True  # red can settle anywhere
        elif group == YELLOW:
            can_settle = current != RED  # anywhere except on red
        else:  # BLUE
            can_settle = current == EMPTY  # only on empty cells

        if not can_settle:
            return False

        if current != EMPTY:
            self._pending.append(current)  # evicted agent re-enters the pool

        self.society[cell_id].group = group
        return True

    def _update_tracked_counts(self) -> None:
        gdf = self.gdf
        self.red = int((gdf["group"] == RED).sum())
        self.yellow = int((gdf["group"] == YELLOW).sum())
        self.blue = int((gdf["group"] == BLUE).sum())
        self.pending = len(self._pending)

    # ------------------------------------------------------------------
    # Convenience metrics
    # ------------------------------------------------------------------

    def occupied_fraction(self) -> float:
        """Fraction of grid cells currently occupied by any group."""
        n = len(self.gdf)
        if n == 0:
            return 0.0
        return (self.red + self.yellow + self.blue) / n

    def is_done(self) -> bool:
        """Whether every agent has successfully settled."""
        return len(self._pending) == 0


__all__ = ["PeripherisationModel", "EMPTY", "RED", "YELLOW", "BLUE"]
