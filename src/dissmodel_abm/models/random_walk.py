"""
dissmodel_abm/models/random_walk.py
====================================
Minimal agent-based model: agents perform an independent random walk
within a bounding box.

Equivalent to repeatedly calling TerraME's ``Agent:walk()`` for every
agent in a Society each time step. Uses
:meth:`~dissmodel_abm.core.Society.walk_all`, the vectorized form of
looping ``agent.walk(...)`` over every agent in :attr:`self.society
<dissmodel_abm.core.AgentModel.society>` — appropriate here since every
agent follows the exact same rule with no per-agent branching.
"""
from __future__ import annotations

from dissmodel_abm.core import AgentModel


class RandomWalkModel(AgentModel):
    """
    Each agent moves by a random offset in x and y every time step,
    staying within ``bounds``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame of agents with ``Point`` geometry.
    step_size : float, optional
        Maximum displacement per axis per step, by default 1.0.
    bounds : tuple of float, optional
        ``(minx, miny, maxx, maxy)``, by default ``(0, 0, 100, 100)``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> from dissmodel.core import Environment
    >>> from dissmodel_abm.models.random_walk import RandomWalkModel
    >>> gdf = gpd.GeoDataFrame(
    ...     {"geometry": [Point(50, 50) for _ in range(10)]}
    ... )
    >>> env = Environment(end_time=5)
    >>> model = RandomWalkModel(gdf=gdf, step_size=2.0)
    >>> env.run()  # doctest: +SKIP
    """

    def setup(
        self,
        step_size: float = 1.0,
        bounds: tuple[float, float, float, float] = (0, 0, 100, 100),
    ) -> None:
        self.step_size = step_size
        self.bounds = bounds

    def execute(self) -> None:
        self.society.walk_all(step_size=self.step_size, bounds=self.bounds)


__all__ = ["RandomWalkModel"]
