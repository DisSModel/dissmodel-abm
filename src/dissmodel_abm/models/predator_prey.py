"""
dissmodel_abm/models/predator_prey.py
======================================
Classic predator-prey (wolf-sheep) agent-based model.

Written entirely against :attr:`~dissmodel_abm.core.AgentModel.society`
(see :mod:`dissmodel_abm.core.society`) — the model never touches
``self.gdf`` directly. Demonstrates the full set of TerraME-inspired
operations exposed by :class:`~dissmodel_abm.core.Agent` /
:class:`~dissmodel_abm.core.Society`:

- ``agent.walk()``         — random movement       (TerraME ``Agent:walk``)
- ``agent.neighbors()``    — spatial neighbor query (TerraME neighborhood)
- ``agent.die()``          — remove an agent        (TerraME ``Agent:die``)
- ``agent.reproduce()``    — duplicate an agent      (TerraME ``Agent:reproduce``)

Agents have the following attributes:

- ``geometry`` : Point  — position
- ``kind``     : str    — ``"wolf"`` or ``"sheep"``
- ``energy``   : float  — depletes each step, replenished by eating

Rules
-----
1. Every agent takes one random walk step.
2. Every agent loses ``energy_loss`` energy (sheep first graze, if
   ``graze_gain`` is set).
3. Wolves within ``eat_radius`` of a sheep eat it: the sheep dies, the
   wolf gains ``energy_gain``.
4. Agents with ``energy <= 0`` die.
5. Agents with ``energy >= reproduce_threshold`` reproduce: a child is
   created at the same position with ``energy`` reset to
   ``reproduce_threshold / 2``.

Live plotting
-------------
``sheep`` and ``wolves`` are registered via ``@track_plot``,
so any :class:`dissmodel.visualization.Chart` connected to the same
``Environment`` plots them automatically — no extra wiring needed, same
convention used by ``dissmodel-sysdyn``'s ``SIR`` model.
"""
from __future__ import annotations

from dissmodel.visualization import track_plot
from dissmodel_abm.core import AgentModel
from dissmodel_abm.core.society import Agent


@track_plot("Sheep", "tab:blue")
@track_plot("Wolves", "tab:red")
class PredatorPreyModel(AgentModel):
    """
    Predator-prey (wolf-sheep) agent-based model.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame of agents with columns ``geometry``, ``kind``
        (``"wolf"`` or ``"sheep"``) and ``energy``.
    step_size : float, optional
        Maximum displacement per axis per step, by default 1.0.
    bounds : tuple of float, optional
        ``(minx, miny, maxx, maxy)``, by default ``(0, 0, 100, 100)``.
    eat_radius : float, optional
        Distance within which a wolf can eat a sheep, by default 1.0.
    energy_loss : float, optional
        Energy lost by every agent each step, by default 1.0.
    energy_gain : float, optional
        Energy gained by a wolf when it eats a sheep, by default 5.0.
    reproduce_threshold : float, optional
        Energy level at which an agent reproduces, by default 15.0.
    graze_gain : float, optional
        Energy regained by sheep each step (grazing), by default 0.0.
        Set to a small positive value (e.g. 0.5-1.0) to allow sheep
        populations to sustain themselves indefinitely.
    verbose : bool, optional
        If True, print population counts (sheep/wolves/total) at the
        end of each :meth:`execute` call, by default False.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>> from dissmodel.core import Environment
    >>> from dissmodel_abm.models.predator_prey import PredatorPreyModel
    >>> n_sheep, n_wolves = 30, 5
    >>> rng = np.random.default_rng(0)
    >>> xs = rng.uniform(0, 100, n_sheep + n_wolves)
    >>> ys = rng.uniform(0, 100, n_sheep + n_wolves)
    >>> gdf = gpd.GeoDataFrame({
    ...     "geometry": gpd.points_from_xy(xs, ys),
    ...     "kind": ["sheep"] * n_sheep + ["wolf"] * n_wolves,
    ...     "energy": [10.0] * n_sheep + [10.0] * n_wolves,
    ... })
    >>> env = Environment(end_time=10)
    >>> model = PredatorPreyModel(gdf=gdf)
    >>> env.run()  # doctest: +SKIP
    """

    #: Current number of sheep agents (tracked for live plotting).
    sheep: int = 0

    #: Current number of wolf agents (tracked for live plotting).
    wolves: int = 0

    def setup(
        self,
        step_size: float = 1.0,
        bounds: tuple[float, float, float, float] = (0, 0, 100, 100),
        eat_radius: float = 1.0,
        energy_loss: float = 1.0,
        energy_gain: float = 5.0,
        reproduce_threshold: float = 15.0,
        graze_gain: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.step_size = step_size
        self.bounds = bounds
        self.eat_radius = eat_radius
        self.energy_loss = energy_loss
        self.energy_gain = energy_gain
        self.reproduce_threshold = reproduce_threshold
        self.graze_gain = graze_gain
        self.verbose = verbose

        # One-time defaults for agents created without these columns.
        # (Touches self.gdf only here, at setup time, for column
        # initialization — execute() below uses self.society exclusively.)
        if "energy" not in self.gdf.columns:
            self.gdf["energy"] = 10.0
        if "kind" not in self.gdf.columns:
            self.gdf["kind"] = "sheep"

    def execute(self) -> None:
        society = self.society

        if len(society) == 0:
            self.sheep = 0
            self.wolves = 0
            if self.verbose:
                print(f"t={self.env.now():<4.0f} sheep=  0  wolves=  0  total=  0")
            return

        # 1. Movement
        for agent in society:
            agent.walk(step_size=self.step_size, bounds=self.bounds)

        # 2. Metabolism (sheep graze, everyone loses energy)
        for agent in society:
            if self.graze_gain and agent.kind == "sheep":
                agent.energy += self.graze_gain
            agent.energy -= self.energy_loss

        # 3. Predation: wolves eat nearby sheep
        self._predation_step(society)

        # 4. Death
        society.remove_if(lambda agent: agent.energy <= 0)

        # 5. Reproduction
        for agent in society.select(lambda a: a.energy >= self.reproduce_threshold):
            agent.reproduce(energy=self.reproduce_threshold / 2.0)

        # Update tracked counts — picked up automatically by any Chart
        # connected to the same Environment (see @track_plot above).
        self.sheep = society.count(lambda a: a.kind == "sheep")
        self.wolves = society.count(lambda a: a.kind == "wolf")

        if self.verbose:
            print(
                f"t={self.env.now():<4.0f} "
                f"sheep={self.sheep:3d}  wolves={self.wolves:3d}  "
                f"total={len(society):3d}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predation_step(self, society) -> None:
        """Wolves within ``eat_radius`` of a sheep eat it."""
        wolves = society.select(lambda a: a.kind == "wolf")
        if not wolves:
            return

        eaten: set = set()

        for wolf in wolves:
            sheep_nearby = [
                prey for prey in wolf.neighbors(self.eat_radius)
                if prey.id not in eaten and prey.kind == "sheep"
            ]
            if sheep_nearby:
                prey = sheep_nearby[0]
                eaten.add(prey.id)
                wolf.energy += self.energy_gain

        for prey_id in eaten:
            society.remove(prey_id)


__all__ = ["PredatorPreyModel"]
