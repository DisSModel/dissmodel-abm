"""
dissmodel_abm/core/agent_model.py
==================================
Base class for agent-based models (ABM), compatible with dissmodel.

This module does NOT modify dissmodel's core. It provides ``AgentModel``,
a subclass of ``dissmodel.geo.vector.SpatialModel``, adding a small set of
TerraME-inspired operations (``walk``, ``die_if``, ``reproduce_if``,
``neighbors_within``, ``move_to``) that operate on ``self.gdf`` where each
*row* represents one agent (Point geometry + state columns).

Mapping to TerraME's Agent type (https://www.terrame.org/base/types/agent/)
-----------------------------------------------------------------------------
TerraME (Agent / Society)        dissmodel-abm (AgentModel)
--------------------------       ---------------------------------------
execute(self)                     execute()              (Model lifecycle)
init(self)                        setup()                (Model lifecycle)
Society (collection of Agents)    self.society            (Society over self.gdf)
Agent                              self.society[idx]       (Agent — proxy over one row)
placement / getCell()             agent.geometry           (Point)
move(cell) / walk()               agent.move_to() / agent.walk()
die()                              agent.die()
reproduce()                        agent.reproduce()
emptyNeighbor / walkToEmpty        agent.neighbors(radius) / agent.grid_neighbors()
addSocialNetwork / message         not provided here — use a separate graph
                                    (e.g. networkx) keyed by agent.id

Two ways to work with agents
------------------------------
1. **Recommended — Society / Agent** (this is the protective layer):
   work with individual agents as objects, without ever touching
   ``self.gdf`` directly. See :mod:`dissmodel_abm.core.society`.

       for agent in self.society:
           if agent.energy <= 0:
               agent.die()
           elif agent.energy >= 15:
               agent.reproduce(energy=5.0)

2. **Legacy — direct GeoDataFrame methods** (``walk``, ``die_if``,
   ``reproduce_if``, ``neighbors_within`` below): batch/vectorized
   operations on ``self.gdf``, kept for backward compatibility and for
   cases where vectorized performance matters more than per-agent
   readability. New models should prefer ``self.society``.

Usage
-----
    from dissmodel_abm.core import AgentModel

    class MyModel(AgentModel):
        def setup(self, step_size=1.0, bounds=(0, 0, 100, 100)):
            self.step_size = step_size
            self.bounds = bounds

        def execute(self):
            for agent in self.society:
                agent.walk(step_size=self.step_size, bounds=self.bounds)
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from dissmodel.geo.vector import SpatialModel

from .society import Society


class AgentModel(SpatialModel):
    """
    Base class for agent-based models.

    Subclass of :class:`dissmodel.geo.vector.SpatialModel`. Each row of
    ``self.gdf`` represents one agent: a ``Point`` geometry plus any number
    of state columns (energy, age, type, state, ...).

    No changes to dissmodel's core are required — this class only adds
    convenience methods on top of the existing ``gdf`` / ``Model``
    lifecycle (``setup`` / ``pre_execute`` / ``execute`` / ``post_execute``).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame of agents. Must have a ``Point`` geometry column.
    **kwargs :
        Forwarded to :class:`~dissmodel.geo.vector.SpatialModel`
        (``step``, ``start_time``, ``end_time``, ``name``, plus any
        extra kwargs consumed by ``setup()``).

    Attributes
    ----------
    society : Society
        Object-oriented view over ``self.gdf`` — the recommended way to
        read and mutate agents (``self.society[idx]``, ``for agent in
        self.society``, ``self.society.add(...)``). Always in sync with
        ``self.gdf``: both refer to the same underlying data.
    """

    @property
    def society(self) -> Society:
        """Society view over this model's agents (see :class:`Society`)."""
        # Created lazily so AgentModel subclasses don't need to call
        # super().setup() just to get a working `self.society`.
        if not hasattr(self, "_society"):
            self._society = Society(self)
        return self._society

    # ------------------------------------------------------------------
    # Movement (legacy, batch/vectorized — prefer self.society for new code)
    # ------------------------------------------------------------------

    def walk(
        self,
        step_size: float = 1.0,
        bounds: Optional[tuple[float, float, float, float]] = None,
        mask: Optional[pd.Series] = None,
    ) -> None:
        """
        Move agents by a random offset in x and y (TerraME ``Agent:walk``).

        .. note::
           Vectorized batch operation. For per-agent control, prefer
           ``agent.walk(...)`` via :attr:`society`, or
           :meth:`Society.walk_all` for the vectorized equivalent.

        Parameters
        ----------
        step_size : float, optional
            Maximum absolute displacement per axis, by default 1.0.
            Displacement is drawn from ``Uniform(-step_size, step_size)``.
        bounds : tuple of float, optional
            ``(minx, miny, maxx, maxy)``. If given, new positions are
            clipped to stay within these bounds.
        mask : pandas.Series of bool, optional
            If given, only agents where ``mask`` is True are moved.
        """
        n = len(self.gdf)
        if n == 0:
            return

        idx = self.gdf.index
        if mask is not None:
            idx = self.gdf.index[mask]
            n = len(idx)
            if n == 0:
                return

        dx = np.random.uniform(-step_size, step_size, n)
        dy = np.random.uniform(-step_size, step_size, n)

        geom = self.gdf.loc[idx, "geometry"]
        new_x = geom.x.to_numpy() + dx
        new_y = geom.y.to_numpy() + dy

        if bounds is not None:
            minx, miny, maxx, maxy = bounds
            new_x = np.clip(new_x, minx, maxx)
            new_y = np.clip(new_y, miny, maxy)

        self.gdf.loc[idx, "geometry"] = gpd.points_from_xy(new_x, new_y)

    def move_to(self, idx: Any, x: float, y: float) -> None:
        """
        Move a single agent to an absolute position (TerraME ``Agent:move``).

        Parameters
        ----------
        idx : any
            Index of the agent in ``self.gdf``.
        x, y : float
            New coordinates.
        """
        self.gdf.loc[idx, "geometry"] = gpd.points_from_xy([x], [y])[0]

    # ------------------------------------------------------------------
    # Life cycle: death and reproduction (legacy, batch — prefer
    # self.society / agent.die() / agent.reproduce() for new code)
    # ------------------------------------------------------------------

    def die_if(self, condition: Callable[[pd.Series], bool]) -> int:
        """
        Remove agents for which ``condition(row)`` is True
        (TerraME ``Agent:die``).

        .. note::
           Batch operation; resets the GeoDataFrame index. For per-agent
           control without index resets, prefer ``agent.die()`` via
           :attr:`society`, or :meth:`Society.remove_if`.

        Parameters
        ----------
        condition : callable
            Function receiving a row (``pandas.Series``) and returning
            ``bool``. Applied row-wise via ``DataFrame.apply``.

        Returns
        -------
        int
            Number of agents removed.
        """
        if len(self.gdf) == 0:
            return 0

        mask = self.gdf.apply(condition, axis=1).astype(bool)
        n_removed = int(mask.sum())

        if n_removed:
            self.gdf = self.gdf.loc[~mask].reset_index(drop=True)

        return n_removed

    def reproduce_if(
        self,
        condition: Callable[[pd.Series], bool],
        child_fn: Optional[Callable[[pd.Series], pd.Series]] = None,
    ) -> int:
        """
        Duplicate agents for which ``condition(row)`` is True
        (TerraME ``Agent:reproduce``).

        Parameters
        ----------
        condition : callable
            Function receiving a row and returning ``bool``: agents for
            which this is True will produce one child each, placed at the
            same location as the parent.
        child_fn : callable, optional
            Function receiving the *child* row (a copy of the parent row)
            and returning a modified row. Use this to reset attributes
            for the newborn (e.g. ``energy``, ``age``). If not given, the
            child is an exact copy of the parent.

        Returns
        -------
        int
            Number of new agents created.
        """
        if len(self.gdf) == 0:
            return 0

        mask = self.gdf.apply(condition, axis=1).astype(bool)
        n_new = int(mask.sum())
        if n_new == 0:
            return 0

        children = self.gdf.loc[mask].copy()

        if child_fn is not None:
            children = children.apply(child_fn, axis=1)

        self.gdf = gpd.GeoDataFrame(
            pd.concat([self.gdf, children], ignore_index=True),
            crs=self.gdf.crs,
        )

        return n_new

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def neighbors_within(self, idx: Any, radius: float) -> list[Any]:
        """
        Return the indices of agents within ``radius`` of agent ``idx``
        (TerraME ``Agent:emptyNeighbor`` / neighborhood-style queries).

        Uses the GeoDataFrame's spatial index for efficient lookup.

        Parameters
        ----------
        idx : any
            Index of the reference agent in ``self.gdf``.
        radius : float
            Search radius, in the same units as the GeoDataFrame's CRS.

        Returns
        -------
        list
            Indices of agents within ``radius``, excluding ``idx`` itself.
        """
        if idx not in self.gdf.index:
            return []

        geom = self.gdf.loc[idx, "geometry"]
        buf = geom.buffer(radius)

        sindex = self.gdf.sindex
        candidates = list(sindex.query(buf, predicate="intersects"))

        result = []
        for pos in candidates:
            cand_idx = self.gdf.index[pos]
            if cand_idx == idx:
                continue
            if self.gdf.loc[cand_idx, "geometry"].distance(geom) <= radius:
                result.append(cand_idx)

        return result

    def all_neighbors_within(self, radius: float) -> dict[Any, list[Any]]:
        """
        Return a ``{idx: [neighbor_idx, ...]}`` mapping for every agent
        within ``radius`` of each other.

        Convenience wrapper around :meth:`neighbors_within` for the whole
        GeoDataFrame. For large agent counts prefer a single spatial-join
        based approach if this becomes a bottleneck.

        Parameters
        ----------
        radius : float
            Search radius, in the same units as the GeoDataFrame's CRS.

        Returns
        -------
        dict
            Mapping from agent index to a list of neighbor indices.
        """
        return {idx: self.neighbors_within(idx, radius) for idx in self.gdf.index}


__all__ = ["AgentModel"]
