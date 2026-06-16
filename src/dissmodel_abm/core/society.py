"""
dissmodel_abm/core/society.py
================================
``Society`` / ``Agent``: a protective layer between the modeler and the
underlying spatial substrate (today a vector ``GeoDataFrame``; a raster
backend can be added later behind the same interface).

Why this exists
----------------
Without this layer, every model (``PredatorPreyModel``, ``SchellingModel``,
...) ends up reaching directly into ``self.gdf``: calling
``gdf.loc[idx, "energy"]``, ``gdf.drop(index=...)``,
``pd.concat([gdf, new_row])``, ``gdf.sindex.query(...)``, etc. That couples
every model's *science* code to one specific substrate (vector) and to
pandas/geopandas mechanics that have nothing to do with the model's
actual rules.

``Society`` and ``Agent`` hide all of that. The modeler writes:

    for agent in self.society:
        if agent.energy <= 0:
            agent.die()

instead of:

    self.gdf = self.gdf[self.gdf["energy"] > 0].reset_index(drop=True)

Design
------
- :class:`Agent` is a thin proxy over one row of the GeoDataFrame,
  identified by its index. Reading/writing an attribute
  (``agent.energy``, ``agent.energy = 5``) reads/writes the underlying
  cell directly — no copies, no separate state to keep in sync.
- :class:`Society` is the collection. It owns no data of its own: it
  reads and writes through the host model's ``gdf`` attribute (set via
  ``model.gdf = ...``), so ``model.gdf`` and ``society`` are always the
  same data, just accessed through different APIs. This keeps
  `Map`, `Chart`, `ModelExecutor`, and any code that still expects
  ``model.gdf`` working unmodified.
- The public API (``add``, ``remove``, ``select``, ``sample``, ``walk``,
  ``__iter__``, ``__getitem__``, ``Agent.neighbors``) is substrate-agnostic
  by design: a future raster-backed ``Society`` could implement the same
  methods over a NumPy array instead of a GeoDataFrame, without changing
  a single line in model code written against this interface.

Agents without a location
--------------------------
Following TerraME, where an ``Agent`` may exist without any
``placement`` (it gets one via ``Agent:enter()`` and loses it via
``Agent:leave()``), agents here may have ``geometry = None``. Use
``agent.has_location`` to check, ``agent.enter(x, y)`` to give a
location-less agent a position for the first time, and ``agent.leave()``
to remove it. Spatial methods (``move_to``, ``walk``, ``distance_to``,
``neighbors``) raise a clear ``RuntimeError`` if called on an agent with
no location, rather than failing with a cryptic AttributeError on
``None``.

Not yet covered (left for the vector-first iteration)
-------------------------------------------------------
- Raster substrate (planned next; this module focuses on vector).
- Social networks / messaging (TerraME ``addSocialNetwork`` / ``message``).
- State machines (TerraME ``State`` / ``Jump`` / ``Flow``).
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

import geopandas as gpd
import numpy as np
import pandas as pd


class Agent:
    """
    Proxy over a single agent (one row of the host GeoDataFrame).

    Attribute access reads/writes the underlying cell directly. There is
    no separate copy of state: ``agent.energy = 5`` immediately updates
    ``society.gdf.at[idx, "energy"]``.

    Parameters
    ----------
    society : Society
        The society this agent belongs to.
    idx : any
        Index of this agent's row in ``society.gdf``.

    Notes
    -----
    Do not store ``Agent`` instances across steps that may remove or
    reindex the underlying GeoDataFrame (e.g. after any ``die()``,
    ``remove()``, or ``add()`` call elsewhere in the same step). Always
    re-obtain agents via ``society[idx]`` or by iterating ``society``
    within the current step.
    """

    __slots__ = ("_society", "_idx")

    def __init__(self, society: "Society", idx: Any) -> None:
        object.__setattr__(self, "_society", society)
        object.__setattr__(self, "_idx", idx)

    # -- identity ---------------------------------------------------------

    @property
    def id(self) -> Any:
        """Index of this agent in the underlying GeoDataFrame."""
        return self._idx

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Agent):
            return self._society is other._society and self._idx == other._idx
        return NotImplemented

    def __hash__(self) -> int:
        return hash((id(self._society), self._idx))

    def __repr__(self) -> str:
        return f"Agent(id={self._idx!r})"

    # -- attribute access (proxied to the GeoDataFrame row) ---------------

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails, so this
        # never intercepts _society / _idx (declared via __slots__).
        gdf = self._society.gdf
        try:
            return gdf.at[self._idx, name]
        except KeyError as exc:
            raise AttributeError(
                f"Agent has no attribute '{name}' "
                f"(no column '{name}' in the underlying data)"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._society.gdf.at[self._idx, name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Like ``getattr``, but returns ``default`` instead of raising."""
        try:
            return self._society.gdf.at[self._idx, name]
        except KeyError:
            return default

    # -- location (TerraME placement: enter / leave / move) ---------------

    @property
    def has_location(self) -> bool:
        """
        Whether this agent currently has a position.

        Mirrors TerraME, where an Agent may exist without any
        ``placement`` until :meth:`enter` is called (or after
        :meth:`leave`).
        """
        geom = self._society.gdf.at[self._idx, "geometry"]
        return geom is not None and not (isinstance(geom, float) and pd.isna(geom))

    # -- movement -----------------------------------------------------------

    def move_to(self, x: Any, y: Optional[float] = None) -> None:
        """
        Move this agent to a new position (TerraME ``Agent:move``).

        Parameters
        ----------
        x : float or shapely.Geometry
            Either the x-coordinate of a new point position (when ``y``
            is also given), or a complete geometry (Point, Polygon, ...)
            to use as-is — the equivalent of TerraME's
            ``Agent:move(cell)``, which moves the agent into an
            arbitrary ``Cell`` geometry, not necessarily a point.
        y : float, optional
            The y-coordinate, when ``x`` is a coordinate rather than a
            geometry.

        Examples
        --------
        >>> agent.move_to(5, 3)            # move to point (5, 3)
        >>> agent.move_to(some_polygon)     # move into an arbitrary Cell geometry
        """
        if y is None:
            geom = x
        else:
            geom = gpd.points_from_xy([x], [y])[0]
        self._society.gdf.at[self._idx, "geometry"] = geom

    def enter(self, x: Any, y: Optional[float] = None) -> None:
        """
        Give this agent a position for the first time (TerraME
        ``Agent:enter``).

        Accepts the same arguments as :meth:`move_to` (coordinates or a
        complete geometry); provided separately so model code can
        express intent the same way TerraME does: ``enter`` for a
        location-less agent taking its first position, ``move_to`` for
        repositioning an agent that already has one.
        """
        self.move_to(x, y)

    def leave(self) -> None:
        """
        Remove this agent's position, without removing the agent from
        its society (TerraME ``Agent:leave``).

        After calling this, :attr:`has_location` is False and spatial
        methods (``move_to``, ``walk``, ``distance_to``, ``neighbors``)
        will raise until :meth:`enter` is called again.
        """
        self._society.gdf.at[self._idx, "geometry"] = None

    def _require_location(self, action: str) -> Any:
        geom = self._society.gdf.at[self._idx, "geometry"]
        if geom is None or (isinstance(geom, float) and pd.isna(geom)):
            raise RuntimeError(
                f"Agent {self._idx!r} has no location; cannot {action}. "
                f"Call agent.enter(x, y) first."
            )
        return geom

    def walk(
        self,
        step_size: float = 1.0,
        bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Move this agent by a random offset in x and y.

        Uses the geometry's centroid as the reference point, so this
        works whether the agent's geometry is a Point (point-agent
        models) or a Polygon/other shape — though for polygon agents,
        ``walk`` replaces the geometry with a Point at the new position;
        it is intended for point-agent models. For one-agent-per-cell
        polygon models, swap attributes between cells instead (see
        ``SchellingModel``).
        """
        geom = self._require_location("walk")
        centroid = geom.centroid
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        new_x, new_y = centroid.x + dx, centroid.y + dy
        if bounds is not None:
            minx, miny, maxx, maxy = bounds
            new_x = min(max(new_x, minx), maxx)
            new_y = min(max(new_y, miny), maxy)
        self.move_to(new_x, new_y)

    def distance_to(self, other: "Agent") -> float:
        """Euclidean distance to another agent. Both agents must have a location."""
        geom = self._require_location("compute distance")
        other_geom = other._require_location("compute distance")
        return geom.distance(other_geom)

    # -- spatial queries ------------------------------------------------

    def neighbors(self, radius: float) -> list["Agent"]:
        """
        Return the agents within ``radius`` of this agent (excluding
        itself), using the society's spatial index.

        Parameters
        ----------
        radius : float
            Search radius, in the same units as the GeoDataFrame's CRS.

        Raises
        ------
        RuntimeError
            If this agent has no location (see :attr:`has_location`).
        """
        self._require_location("query neighbors")
        return self._society.neighbors_within(self._idx, radius)

    def grid_neighbors(self) -> list["Agent"]:
        """
        Return the topological (grid) neighbors of this agent's cell,
        for one-agent-per-cell models that called
        ``model.create_neighborhood(...)``. Requires the host model to
        be a ``SpatialModel`` with a neighborhood already attached.
        """
        return self._society.grid_neighbors_of(self._idx)

    # -- lifecycle --------------------------------------------------------

    def die(self) -> None:
        """Remove this agent from its society."""
        self._society.remove(self._idx)

    def reproduce(self, **overrides: Any) -> "Agent":
        """
        Create a new agent at the same position as this one, copying all
        attributes, then applying ``overrides``.

        If this agent has no location (see :attr:`has_location`), the
        child is also created without one, unless ``geometry`` is given
        in ``overrides``.

        Parameters
        ----------
        **overrides :
            Attribute values to override on the child (e.g.
            ``energy=5.0``). ``geometry`` defaults to the parent's
            current position unless explicitly overridden.

        Returns
        -------
        Agent
            The newly created agent.
        """
        row = self._society.gdf.loc[self._idx].to_dict()
        row.update(overrides)
        return self._society.add(**row)


class Society:
    """
    Protective collection layer over a vector (GeoDataFrame) substrate.

    A ``Society`` does not own its data — it reads and writes through
    the host model's ``gdf`` attribute, so it is always in sync with
    whatever ``Map``, ``Chart``, or other dissmodel components see via
    ``model.gdf``.

    Parameters
    ----------
    model : dissmodel.geo.vector.SpatialModel
        The host model. Must expose a ``gdf`` attribute (a
        ``geopandas.GeoDataFrame``) and, for grid-neighbor queries, may
        optionally expose ``neighs_id`` (provided by ``SpatialModel``
        after calling ``create_neighborhood``).

    Examples
    --------
    >>> for agent in self.society:
    ...     if agent.energy <= 0:
    ...         agent.die()
    ...     elif agent.energy >= 15:
    ...         agent.reproduce(energy=5.0)
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    # -- substrate access (delegates to the host model) -------------------

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """The underlying GeoDataFrame (same object as ``model.gdf``)."""
        return self._model.gdf

    @gdf.setter
    def gdf(self, value: gpd.GeoDataFrame) -> None:
        self._model.gdf = value

    # -- collection protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self.gdf)

    def __iter__(self) -> Iterator[Agent]:
        # Snapshot the index list up front: safe to call agent.die() or
        # society.add() while iterating, since we don't re-read the
        # (possibly mutated) GeoDataFrame's index mid-loop.
        for idx in list(self.gdf.index):
            yield Agent(self, idx)

    def __getitem__(self, idx: Any) -> Agent:
        if idx not in self.gdf.index:
            raise KeyError(f"No agent with id {idx!r}")
        return Agent(self, idx)

    def __contains__(self, idx: Any) -> bool:
        return idx in self.gdf.index

    # -- lifecycle ----------------------------------------------------------

    def add(self, geometry: Any = None, **attrs: Any) -> Agent:
        """
        Add a new agent to the society.

        Parameters
        ----------
        geometry : shapely.Geometry, optional
            Position of the new agent. May be omitted (or explicitly
            ``None``) to create an agent with no location yet — mirrors
            TerraME, where an Agent can exist without a ``placement``
            until ``Agent:enter()`` is called. Call ``agent.enter(x, y)``
            later to give it a position.
        **attrs :
            Any other attributes to set on the new agent (columns will
            be created on first use if they don't already exist).

        Returns
        -------
        Agent
            The newly created agent.
        """
        attrs["geometry"] = geometry

        gdf = self.gdf
        next_idx = (max(gdf.index) + 1) if len(gdf) else 0

        # Mutate the existing GeoDataFrame in place (rather than
        # reassigning self.gdf to a new object) so that any external
        # reference to this same object — e.g. a dissmodel Map component
        # constructed with the original gdf — keeps seeing live updates.
        for col in attrs:
            if col not in gdf.columns:
                gdf[col] = None
        gdf.loc[next_idx] = attrs

        return Agent(self, next_idx)

    def remove(self, agent_or_idx: "Agent | Any") -> None:
        """
        Remove an agent from the society.

        Parameters
        ----------
        agent_or_idx : Agent or index
            The agent (or its index) to remove.
        """
        idx = agent_or_idx.id if isinstance(agent_or_idx, Agent) else agent_or_idx
        self.gdf.drop(index=idx, inplace=True)

    def remove_if(self, condition: Callable[[Agent], bool]) -> int:
        """
        Remove every agent for which ``condition(agent)`` is True.

        Returns
        -------
        int
            Number of agents removed.
        """
        to_remove = [agent.id for agent in self if condition(agent)]
        if to_remove:
            self.gdf.drop(index=to_remove, inplace=True)
        return len(to_remove)

    # -- selection ------------------------------------------------------

    def select(self, predicate: Callable[[Agent], bool]) -> list[Agent]:
        """Return the agents for which ``predicate(agent)`` is True."""
        return [agent for agent in self if predicate(agent)]

    def sample(self, n: int) -> list[Agent]:
        """Return ``n`` agents chosen uniformly at random, without replacement."""
        n = min(n, len(self))
        idxs = np.random.choice(self.gdf.index.to_numpy(), size=n, replace=False)
        return [Agent(self, idx) for idx in idxs]

    def count(self, predicate: Optional[Callable[[Agent], bool]] = None) -> int:
        """Count agents, optionally matching ``predicate``."""
        if predicate is None:
            return len(self)
        return sum(1 for agent in self if predicate(agent))

    # -- batch movement (vectorized, for performance) -----------------------

    def walk_all(
        self,
        step_size: float = 1.0,
        bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Move every agent by an independent random offset in x and y.

        Vectorized equivalent of calling ``agent.walk(...)`` on every
        agent; prefer this for large societies.
        """
        gdf = self.gdf
        n = len(gdf)
        if n == 0:
            return

        dx = np.random.uniform(-step_size, step_size, n)
        dy = np.random.uniform(-step_size, step_size, n)
        new_x = gdf.geometry.x.to_numpy() + dx
        new_y = gdf.geometry.y.to_numpy() + dy

        if bounds is not None:
            minx, miny, maxx, maxy = bounds
            new_x = np.clip(new_x, minx, maxx)
            new_y = np.clip(new_y, miny, maxy)

        gdf["geometry"] = gpd.points_from_xy(new_x, new_y)

    # -- spatial queries (point agents) --------------------------------------

    def neighbors_within(self, idx: Any, radius: float) -> list[Agent]:
        """
        Return the agents within ``radius`` of agent ``idx`` (excluding itself).

        Raises
        ------
        RuntimeError
            If agent ``idx`` has no location (``geometry`` is ``None``).
        """
        gdf = self.gdf
        if idx not in gdf.index:
            return []

        geom = gdf.at[idx, "geometry"]
        if geom is None or (isinstance(geom, float) and pd.isna(geom)):
            raise RuntimeError(
                f"Agent {idx!r} has no location; cannot query neighbors. "
                f"Call agent.enter(x, y) first."
            )
        buf = geom.buffer(radius)

        sindex = gdf.sindex
        candidates = list(sindex.query(buf, predicate="intersects"))

        result = []
        for pos in candidates:
            cand_idx = gdf.index[pos]
            if cand_idx == idx:
                continue
            cand_geom = gdf.at[cand_idx, "geometry"]
            # Location-less agents (geometry=None) can't be neighbors of
            # anyone and are silently skipped rather than raising, since
            # they were never the subject of this query.
            if cand_geom is None:
                continue
            if cand_geom.distance(geom) <= radius:
                result.append(Agent(self, cand_idx))

        return result

    # -- topological queries (one-agent-per-cell models) ---------------------

    def grid_neighbors_of(self, idx: Any) -> list[Agent]:
        """
        Return the topological (grid) neighbors of cell ``idx``.

        Requires the host model to have called
        ``model.create_neighborhood(...)`` beforehand (provided by
        ``dissmodel.geo.vector.SpatialModel``).
        """
        neighs_id = getattr(self._model, "neighs_id", None)
        if neighs_id is None:
            raise RuntimeError(
                "Host model does not support grid neighborhoods. "
                "Call model.create_neighborhood(...) first."
            )
        return [Agent(self, nb_idx) for nb_idx in neighs_id(idx)]


__all__ = ["Society", "Agent"]
